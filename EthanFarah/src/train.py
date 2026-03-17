"""Training loop: REINFORCE/MCTS + distilroberta + hybrid/sparse reward."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from src.algo.base import Algorithm
from src.config import Config, parse_args
from src.data.squad import load_squad
from src.data.token_alignment import TokenAligner, map_mask
from src.data.tokenization import get_tokenizer
from src.data.types import CompressedPrompt, Episode, Prompt, RewardInput
from src.env.chunking import ChunkConfig, merge_chunk_actions
from src.env.compression_env import CompressionEnv
from src.llm.frozen_llm import FrozenLLM
from src.llm.kl_cache import KLCache
from src.policy.base import Policy
from src.reward.hybrid import HybridReward
from src.reward.metrics import compute_task_score
from src.tracking.checkpointing import CheckpointState, load_checkpoint, save_checkpoint
from src.tracking.wandb_logger import WandbLogger


def _transfer_obs_to_device(obs: "Observation", device: str) -> None:
    """Move observation tensors to the target device in-place."""
    from src.env.spaces import Observation

    obs.token_ids = obs.token_ids.to(device)
    obs.attention_mask = obs.attention_mask.to(device)
    obs.position_ids = obs.position_ids.to(device)


def collect_episode(
    env: CompressionEnv,
    policy: Policy,
    prompt: Prompt,
    device: str = "cpu",
    greedy: bool = False,
) -> Episode:
    """Run one full episode: reset env, loop policy, merge log_probs."""
    obs = env.reset(prompt)
    chunk_log_probs: list[torch.Tensor] = []
    done = False

    while not done:
        _transfer_obs_to_device(obs, device)
        with torch.no_grad():
            if greedy:
                actions, log_probs = policy.act_greedy(obs)
            else:
                actions, log_probs = policy.act(obs)
        if actions.shape[0] != 1:
            raise ValueError(
                f"collect_episode requires batch=1, got {actions.shape[0]}"
            )
        chunk_log_probs.append(log_probs[0].cpu())
        obs, _reward, done, _info = env.step(actions[0].cpu())

    episode = env.get_episode()
    episode.log_probs = merge_chunk_actions(chunk_log_probs, env.chunk_config)
    return episode


_EMPTY_COMPRESSION_PENALTY = -0.5


def _prepend_bos(token_ids: Tensor, frozen_llm: FrozenLLM) -> Tensor:
    """Prepend BOS token if the tokenizer has a distinct one and it's not already first."""
    bos_id = frozen_llm.tokenizer.bos_token_id
    if bos_id is None or bos_id == frozen_llm.tokenizer.eos_token_id:
        return token_ids
    if token_ids.numel() > 0 and token_ids[0].item() == bos_id:
        return token_ids
    return torch.cat([torch.tensor([bos_id], device=token_ids.device), token_ids])
_CACHE_MAX_SIZE = 1024
_original_output_cache: dict[str, str] = {}


def _get_original_output(prompt_text: str, frozen_llm: FrozenLLM) -> str:
    if prompt_text in _original_output_cache:
        return _original_output_cache[prompt_text]
    original_ids = frozen_llm.tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=True,
        truncation=True, max_length=frozen_llm.tokenizer.model_max_length,
    )["input_ids"]
    output = frozen_llm.generate(original_ids)
    if len(_original_output_cache) >= _CACHE_MAX_SIZE:
        _original_output_cache.pop(next(iter(_original_output_cache)))
    _original_output_cache[prompt_text] = output
    return output


@dataclass
class ScoreResult:
    """Output of score_episode."""

    llm_output: str | None
    original_llm_output: str | None
    sparse_reward: float = 0.0
    dense_reward: float = 0.0


def _get_llm_space_mask(episode: Episode) -> Tensor:
    """Convert policy-space keep_mask to LLM-space for KL computation."""
    if episode.prompt.alignment is not None:
        return map_mask(episode.compressed.keep_mask, episode.prompt.alignment).bool()
    return episode.compressed.keep_mask.bool()


def _is_empty_compression(compressed: CompressedPrompt) -> bool:
    """Check if all tokens are masked (nothing for the LLM to attend to)."""
    return compressed.token_ids.shape[0] == 0


def score_episode(
    episode: Episode,
    reward_fn: HybridReward,
    frozen_llm: FrozenLLM,
    kl_cache: KLCache,
    sparse_only: bool = False,
) -> ScoreResult:
    """Compute reward and write it into episode in-place.

    When sparse_only=True (PCRL mode), dense rewards are set to zero
    and only the terminal sparse reward is used.
    """
    if _is_empty_compression(episode.compressed):
        episode.terminal_reward = _EMPTY_COMPRESSION_PENALTY
        return ScoreResult(None, None)

    try:
        kl_cache.cache_full_prompt(episode.prompt)
        original_llm_output = _get_original_output(episode.prompt.text, frozen_llm)
        if not original_llm_output.strip():
            episode.terminal_reward = 0.0
            return ScoreResult(None, original_llm_output)

        compressed_ids = _prepend_bos(episode.compressed.token_ids, frozen_llm)
        llm_output = frozen_llm.generate(compressed_ids.unsqueeze(0))
        reward_input = RewardInput(
            original=episode.prompt,
            compressed=episode.compressed,
            llm_output=llm_output,
            original_llm_output=original_llm_output,
        )

        if sparse_only:
            seq_len = episode.actions.shape[0]
            episode.rewards = torch.zeros(seq_len, dtype=torch.float)
            terminal = reward_fn._sparse_reward.compute(reward_input)
            sparse_val = terminal.item()

            dense_ce = 0.0
            if reward_fn.config.dense_ce_coeff > 0 and original_llm_output.strip():
                target_ids = frozen_llm.tokenizer(
                    original_llm_output, return_tensors="pt", add_special_tokens=False,
                )["input_ids"]
                if target_ids.shape[1] > 0:
                    ce = frozen_llm.teacher_forced_ce(compressed_ids.unsqueeze(0), target_ids)
                    dense_ce = -reward_fn.config.dense_ce_coeff * ce

            episode.terminal_reward = sparse_val + dense_ce
            result = ScoreResult(llm_output, original_llm_output, sparse_val, dense_ce)
        else:
            dense_reward = reward_fn.compute(reward_input)
            seq_len = episode.actions.shape[0]
            shared_reward = dense_reward.sum().item() / seq_len
            episode.rewards = torch.full((seq_len,), shared_reward, dtype=torch.float)
            terminal = reward_fn.terminal_scalar()
            episode.terminal_reward = terminal.item() if terminal is not None else 0.0
            result = ScoreResult(
                llm_output, original_llm_output,
                sparse_reward=episode.terminal_reward,
                dense_reward=episode.rewards.sum().item(),
            )

        return result
    finally:
        kl_cache.clear()


def _compute_actual_f1(llm_output: str, metadata: dict[str, Any]) -> float | None:
    if not metadata.get("answer_texts"):
        return None
    return compute_task_score(llm_output, metadata)


@dataclass
class Components:
    """Bundle of initialized training components."""

    env: CompressionEnv
    policy: Policy
    algorithm: Algorithm
    reward_fn: HybridReward
    train_prompts: list[Prompt]
    val_prompts: list[Prompt]
    frozen_llm: FrozenLLM
    kl_cache: KLCache


_POLICY_MODEL_MAP = {
    "distilroberta": {"distilroberta-base"},
}


def _build_aligner(config: Config, llm_tokenizer: "PreTrainedTokenizerFast") -> TokenAligner | None:
    """Create TokenAligner for dual-tokenizer mode, or None for single-tokenizer."""
    if config.policy_type != "distilroberta":
        return None
    policy_tokenizer = get_tokenizer(config.policy_model_name)
    return TokenAligner(policy_tokenizer, llm_tokenizer)


def init_components(config: Config) -> Components:
    """Create all training components from config."""
    if config.k_samples < 1:
        raise ValueError(f"k_samples must be >= 1, got {config.k_samples}")
    if config.kl_coeff > 0:
        raise ValueError(
            "kl_coeff > 0 is incompatible with text bridge mode "
            "(KL penalty requires aligned token IDs between original and compressed)"
        )
    valid = _POLICY_MODEL_MAP.get(config.policy_type, set())
    if valid and config.policy_model_name not in valid:
        raise ValueError(
            f"policy_model_name={config.policy_model_name!r} is not compatible "
            f"with policy_type={config.policy_type!r}. Valid: {valid}"
        )
    frozen_llm = FrozenLLM(config.llm, device=config.device)
    kl_cache = KLCache(frozen_llm)
    reward_fn = HybridReward(config, kl_cache)

    chunk_config = ChunkConfig(
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
    llm_tokenizer = get_tokenizer(config.llm.model_name)
    policy_tokenizer = get_tokenizer(config.policy_model_name)
    aligner = _build_aligner(config, llm_tokenizer)
    env = CompressionEnv(
        chunk_config=chunk_config,
        target_ratio=config.target_compression_ratio,
        policy_tokenizer=policy_tokenizer,
        llm_tokenizer=llm_tokenizer,
    )

    vocab_size = llm_tokenizer.vocab_size
    policy = _build_policy(config, vocab_size)
    policy = policy.to(config.device)
    algorithm = _build_algorithm(config, policy)

    train_cap = config.max_train_samples if config.max_train_samples is not None else config.num_episodes
    train_prompts = load_squad(
        split=config.train_split,
        max_samples=train_cap,
        model_name=config.llm.model_name,
        max_length=config.max_prompt_tokens,
        aligner=aligner,
    )
    val_prompts = load_squad(
        split=config.val_split,
        max_samples=config.max_val_samples,
        model_name=config.llm.model_name,
        max_length=config.max_prompt_tokens,
        aligner=aligner,
    )
    if not train_prompts:
        raise ValueError("No training prompts loaded")
    if not val_prompts:
        raise ValueError("No validation prompts loaded")

    return Components(
        env=env, policy=policy, algorithm=algorithm,
        reward_fn=reward_fn, train_prompts=train_prompts,
        val_prompts=val_prompts, frozen_llm=frozen_llm,
        kl_cache=kl_cache,
    )


def _load_policy_checkpoint(policy: Policy, path: str) -> None:
    """Load pre-trained head weights from a checkpoint file."""
    ckpt = torch.load(Path(path), weights_only=True)
    if "head" not in ckpt:
        raise ValueError(f"Checkpoint at {path} missing 'head' key. Keys: {list(ckpt.keys())}")
    policy.head.load_state_dict(ckpt["head"])
    sys.stderr.write(f"Loaded pre-trained policy head from {path}\n")


def _build_policy(config: Config, vocab_size: int) -> Policy:
    if config.policy_type != "distilroberta":
        raise ValueError(f"Unknown policy_type: {config.policy_type}")
    from src.policy.distilroberta import DistilRoBERTaPolicy
    policy = DistilRoBERTaPolicy(config, vocab_size=vocab_size)
    if config.policy_checkpoint is not None:
        _load_policy_checkpoint(policy, config.policy_checkpoint)
    return policy


def _build_algorithm(config: Config, policy: Policy) -> Algorithm:
    if config.algorithm_type == "reinforce":
        from src.algo.reinforce_simple import SimpleREINFORCE
        return SimpleREINFORCE(policy, config)
    if config.algorithm_type == "mcts":
        from src.algo.mcts import MCTSAlgorithm
        from src.policy.distilroberta import DistilRoBERTaPolicy
        if isinstance(policy, DistilRoBERTaPolicy):
            policy.enable_value_head()
        return MCTSAlgorithm(policy, config)
    raise ValueError(f"Unknown algorithm_type: {config.algorithm_type}")


@dataclass
class _LogContext:
    """Bundle of per-episode data needed for logging."""

    ep_idx: int
    episode: Episode
    metrics: dict[str, float]
    config: Config
    score_result: ScoreResult
    prompt: Prompt
    logger: WandbLogger | None


def _write_debug_log(ctx: _LogContext) -> None:
    """Append detailed episode info to debug log file."""
    if ctx.config.debug_log is None:
        return
    tokenizer = get_tokenizer(ctx.config.llm.model_name)
    compressed_text = tokenizer.decode(ctx.episode.compressed.token_ids.tolist())
    faith = ""
    if ctx.score_result.llm_output and ctx.score_result.original_llm_output:
        from src.reward.metrics import compute_rouge
        rouge = compute_rouge(ctx.score_result.llm_output, ctx.score_result.original_llm_output)
        faith = f"{rouge['rougeL']:.4f}"
    lines = [
        f"=== Episode {ctx.ep_idx + 1} ===",
        f"Compression ratio: {ctx.episode.compressed.compression_ratio:.4f}",
        f"Faithfulness ROUGE-L: {faith}",
        f"Reward: {ctx.score_result.sparse_reward + ctx.score_result.dense_reward:.4f}",
        f"Original prompt (first 500 chars):\n{ctx.prompt.text[:500]}",
        f"Compressed text (first 500 chars):\n{compressed_text[:500]}",
        f"Original LLM output:\n{ctx.score_result.original_llm_output or '(none)'}",
        f"Compressed LLM output:\n{ctx.score_result.llm_output or '(none)'}",
        "-" * 60,
        "",
    ]
    path = Path(ctx.config.debug_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _log_episode(ctx: _LogContext) -> None:
    """Build log dict, print summary, push to W&B."""
    sparse_reward = ctx.score_result.sparse_reward
    dense_reward = ctx.score_result.dense_reward
    total_reward = sparse_reward + dense_reward

    log_dict: dict[str, float] = {
        **{k: v for k, v in ctx.metrics.items() if not isinstance(v, list)},
        "reward/total": total_reward,
        "reward/sparse": sparse_reward,
        "reward/dense": dense_reward,
        "compression_ratio": ctx.episode.compressed.compression_ratio,
    }
    if "score_std" in ctx.metrics:
        log_dict["train/score_std"] = ctx.metrics["score_std"]

    if ctx.score_result.llm_output is not None:
        actual_f1 = _compute_actual_f1(ctx.score_result.llm_output, ctx.prompt.metadata)
        if actual_f1 is not None:
            log_dict["actual_f1"] = actual_f1
        if ctx.score_result.original_llm_output is not None:
            from src.reward.metrics import compute_rouge
            rouge = compute_rouge(ctx.score_result.llm_output, ctx.score_result.original_llm_output)
            log_dict["faithfulness_rougeL"] = rouge["rougeL"]

    faith_str = f" | faith={log_dict['faithfulness_rougeL']:.3f}" if "faithfulness_rougeL" in log_dict else ""
    grad_str = ""
    if "grad_norm_pre" in ctx.metrics:
        grad_str = (
            f" | gnorm={ctx.metrics['grad_norm_pre']:.3f}→{ctx.metrics['grad_norm_post']:.3f}"
            f" clip={ctx.metrics['clip_ratio']:.2f}x"
        )
    adv_str = ""
    if "advantage_mean" in ctx.metrics:
        nz = int(ctx.metrics.get("num_nonzero_adv", 0))
        adv_str = f" | adv={ctx.metrics['advantage_mean']:.4f} ({nz}/{ctx.config.k_samples} nonzero)"
    sample_str = ""
    if "sample_rewards" in ctx.metrics:
        rews = ctx.metrics["sample_rewards"]
        if isinstance(rews, list):
            sample_str = f" | R=[{','.join(f'{r:.3f}' for r in rews)}] base={ctx.metrics.get('baseline_reward', 0):.3f}"
    print(
        f"ep {ctx.ep_idx + 1}/{ctx.config.num_episodes} | "
        f"reward={total_reward:.4f} (dense={dense_reward:.4f} sparse={sparse_reward:.4f}) | "
        f"ratio={ctx.episode.compressed.compression_ratio:.4f}"
        f"{faith_str}"
        f" | ent={ctx.metrics.get('entropy', 0):.6f}"
        f" | sstd={ctx.metrics.get('score_std', 0):.6f}"
        f"{grad_str}{adv_str}{sample_str}",
        file=sys.stderr,
    )
    if ctx.logger is not None:
        ctx.logger.log_metrics(log_dict, step=ctx.ep_idx + 1, commit=False)
        ctx.logger.log_episode(ctx.episode, step=ctx.ep_idx + 1, commit=True)
    _write_debug_log(ctx)


def _run_eval(
    c: Components, config: Config, ep_idx: int, logger: WandbLogger | None,
) -> float:
    """Run validation eval and return task_score."""
    from src.evaluate import EvalContext, evaluate

    ctx = EvalContext(
        env=c.env, reward_fn=c.reward_fn,
        frozen_llm=c.frozen_llm, kl_cache=c.kl_cache,
    )
    sparse_only = _is_sparse_only(config)
    eval_metrics = evaluate(c.policy, c.val_prompts, ctx, device=config.device, sparse_only=sparse_only)
    step = ep_idx + 1
    print(
        f"  [eval @ {step}] "
        + " | ".join(f"{k}={v:.4f}" for k, v in eval_metrics.items()),
        file=sys.stderr,
    )
    if logger is not None:
        logger.log_metrics(
            {f"eval/{k}": v for k, v in eval_metrics.items()},
            step=step,
        )
    return eval_metrics.get("task_score", 0.0)


def _ckpt_dir(config: Config) -> Path:
    return Path(config.output_dir) / config.experiment_name / "checkpoints"


def _is_sparse_only(config: Config) -> bool:
    return config.algorithm_type in {"reinforce", "mcts"}


def _update_mcts_value_fn(algorithm: "MCTSAlgorithm", episode: Episode, score: ScoreResult) -> None:
    """Update MCTS heuristic value EMA with observed faithfulness."""
    ratio = episode.compressed.compression_ratio
    if ratio < 1.0 and score.sparse_reward > 0:
        faithfulness = min(score.sparse_reward / (1.0 - ratio), 1.0)
    else:
        faithfulness = 0.0
    algorithm.update_value_fn(faithfulness)


def train(config: Config) -> None:
    """Main training loop with k-sample SCST baseline."""
    from src.algo.mcts import MCTSAlgorithm
    c = init_components(config)
    logger = WandbLogger(config) if config.wandb_project else None
    sparse_only = _is_sparse_only(config)

    start_ep = 0
    if config.resume_from:
        ckpt_state = CheckpointState(c.policy, c.algorithm, step=0)
        start_ep = load_checkpoint(ckpt_state, Path(config.resume_from))

    best_task_score = -float("inf")
    ckpt_base = _ckpt_dir(config)
    episode_buffer: list[Episode] = []
    score_buffer: list[ScoreResult] = []
    prompt_buffer: list[Prompt] = []

    try:
        for ep_idx in range(start_ep, config.num_episodes):
            dataset_len = len(c.train_prompts)
            if config.shuffle and ep_idx % dataset_len == 0:
                epoch = ep_idx // dataset_len
                random.Random(config.seed + epoch).shuffle(c.train_prompts)
            prompt = c.train_prompts[ep_idx % dataset_len]

            # Greedy rollout for SCST baseline
            greedy_ep = collect_episode(c.env, c.policy, prompt, device=config.device, greedy=True)
            score_episode(greedy_ep, c.reward_fn, c.frozen_llm, c.kl_cache, sparse_only=sparse_only)
            baseline_reward = greedy_ep.terminal_reward

            # k sampled rollouts
            for _ in range(config.k_samples):
                algo_ep = c.algorithm.collect_episode(c.env, prompt)
                ep = algo_ep if algo_ep is not None else collect_episode(
                    c.env, c.policy, prompt, device=config.device,
                )
                sr = score_episode(ep, c.reward_fn, c.frozen_llm, c.kl_cache, sparse_only=sparse_only)
                ep.baseline_reward = baseline_reward
                episode_buffer.append(ep)
                score_buffer.append(sr)
                prompt_buffer.append(prompt)
                if isinstance(c.algorithm, MCTSAlgorithm):
                    _update_mcts_value_fn(c.algorithm, ep, sr)

            # Only update when buffer is full
            buffer_target = config.k_samples * config.accumulation_steps
            if len(episode_buffer) >= buffer_target:
                metrics = c.algorithm.update(episode_buffer)

                # Log using best sample (highest reward) as representative
                best_idx = max(range(len(episode_buffer)), key=lambda i: episode_buffer[i].terminal_reward)
                if (ep_idx + 1) % config.log_every == 0:
                    _log_episode(_LogContext(
                        ep_idx=ep_idx, episode=episode_buffer[best_idx], metrics=metrics,
                        config=config, score_result=score_buffer[best_idx],
                        prompt=prompt_buffer[best_idx], logger=logger,
                    ))

                episode_buffer = []
                score_buffer = []
                prompt_buffer = []
            else:
                metrics = None

            if metrics is not None and (ep_idx + 1) % config.checkpoint_every == 0:
                save_checkpoint(
                    CheckpointState(c.policy, c.algorithm, ep_idx + 1),
                    ckpt_base / f"step_{ep_idx + 1}.pt",
                )

            if (ep_idx + 1) % config.eval_every == 0:
                task_score = _run_eval(c, config, ep_idx, logger)
                if task_score > best_task_score:
                    best_task_score = task_score
                    save_checkpoint(
                        CheckpointState(c.policy, c.algorithm, ep_idx + 1),
                        ckpt_base / "best.pt",
                    )
                    print(
                        f"  [best] new best task_score={task_score:.4f} @ step {ep_idx + 1}",
                        file=sys.stderr,
                    )
    finally:
        if logger is not None:
            logger.finish()


def main() -> None:
    """CLI entry point: parse args, set seeds, call train."""
    import random
    import numpy as np
    from dotenv import load_dotenv
    load_dotenv()

    config = parse_args()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train(config)


if __name__ == "__main__":
    main()
