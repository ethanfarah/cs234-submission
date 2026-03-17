"""Training loop with factory functions for creating components."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import wandb
except ModuleNotFoundError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]

from src.algo.base import Algorithm
from src.config import (
    AlgoConfig,
    AlgoType,
    DataConfig,
    ExperimentConfig,
    MinRatioEnforcementMode,
    PolicyArch,
    PolicyConfig,
    RewardConfig,
    SparseRewardMode,
    TerminalRewardDistribution,
    RewardType,
    TrainConfig,
    parse_args,
)
from src.data.meetingbank import load_meetingbank
from src.data.squad import load_squad
from src.data.types import Episode, Prompt, RewardInput
from src.env.chunking import ChunkConfig, merge_chunk_actions
from src.env.compression_env import CompressionEnv
from src.policy.base import Policy
from src.reward.base import RewardFunction
from src.reward.kl_dense import KLDenseReward
from src.reward.metrics import compute_f1
from src.reward.sparse import SparseReward
from src.tracking.checkpointing import CheckpointState, load_checkpoint, save_checkpoint
from src.tracking.wandb_logger import WandbLogger

if TYPE_CHECKING:
    from src.llm.frozen_llm import FrozenLLM
    from src.llm.kl_cache import KLCache


def create_policy(config: PolicyConfig) -> Policy:
    """Factory that returns the right policy based on config.arch."""
    if config.arch == PolicyArch.DISTILBERT:
        from src.policy.distilbert import DistilBERTPolicy
        return DistilBERTPolicy(config)
    if config.arch == PolicyArch.TINYLLAMA:
        from src.policy.tinyllama import TinyLlamaPolicy
        return TinyLlamaPolicy(config)
    if config.arch == PolicyArch.CUSTOM_TRANSFORMER:
        from src.policy.custom_transformer import CustomTransformerPolicy
        return CustomTransformerPolicy(config)
    raise ValueError(f"Unknown policy arch: {config.arch}")


def create_algorithm(policy: Policy, config: AlgoConfig) -> Algorithm:
    """Factory that returns the right algorithm based on config.algo_type."""
    if config.algo_type == AlgoType.REINFORCE:
        from src.algo.reinforce import REINFORCE
        return REINFORCE(policy, config)
    if config.algo_type == AlgoType.PPO:
        from src.algo.ppo import PPO
        return PPO(policy, config)
    if config.algo_type == AlgoType.DQN:
        from src.algo.dqn import DQN
        return DQN(policy, config)
    if config.algo_type == AlgoType.BANDIT:
        from src.algo.bandit import ContextualBandit
        return ContextualBandit(policy, config)
    raise ValueError(f"Unknown algo type: {config.algo_type}")


def create_reward(
    config: RewardConfig,
    kl_cache: KLCache | None = None,
) -> RewardFunction:
    """Factory that returns the right reward function based on config."""
    if config.reward_type == RewardType.SPARSE:
        return SparseReward(config)
    if config.reward_type in (RewardType.KL_DENSE, RewardType.HYBRID):
        if kl_cache is None:
            raise ValueError(
                f"{config.reward_type.value} reward requires kl_cache"
            )
        if config.reward_type == RewardType.KL_DENSE:
            from src.reward.kl_dense import KLDenseReward
            return KLDenseReward(config, kl_cache)
        from src.reward.hybrid import HybridReward
        return HybridReward(config, kl_cache)
    if config.reward_type == RewardType.LEARNED:
        from src.reward.learned import LearnedReward
        return LearnedReward(config)
    raise ValueError(f"Unknown reward type: {config.reward_type}")


def load_data(
    config: DataConfig,
    tokenizer_model: str | None = None,
) -> tuple[list[Prompt], list[Prompt]]:
    """Load train/val data based on config.dataset.

    tokenizer_model: if set, used for tokenization (e.g. match LLM for smoke tests).
    When None, loaders use their default (Llama for SQuAD/MeetingBank).
    """
    if config.dataset == "squad":
        loader = load_squad
    elif config.dataset == "meetingbank":
        loader = load_meetingbank
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}")

    base_kwargs: dict[str, Any] = dict(max_length=config.max_prompt_tokens)
    if tokenizer_model is not None:
        base_kwargs["model_name"] = tokenizer_model

    train_kwargs = {**base_kwargs}
    if config.dataset == "squad" and config.answerable_only:
        train_kwargs["answerable_only"] = True

    train_prompts = loader(
        split=config.train_split,
        max_samples=config.max_train_samples,
        **train_kwargs,
    )
    val_prompts = loader(
        split=config.val_split,
        max_samples=config.max_val_samples,
        **base_kwargs,
    )
    return train_prompts, val_prompts


def collect_episode(
    env: CompressionEnv,
    policy: Policy,
    prompt: Prompt,
) -> Episode:
    """Run one full episode: reset env, loop policy, merge log_probs.

    The policy processes each chunk, producing per-token keep/drop decisions.
    Chunk-level log_probs are merged using the same overlap-dedup logic as
    actions (earlier chunk wins).

    Returned log_probs are detached (collected under torch.no_grad).
    Algorithms that need gradients through log_probs (e.g. REINFORCE)
    must recompute them via policy.evaluate_actions() at update time.
    """
    device = next(policy.parameters()).device
    obs = env.reset(prompt)
    obs = obs.to(device)
    chunk_log_probs: list[torch.Tensor] = []
    final_info: dict[str, float] = {}
    done = False

    while not done:
        with torch.no_grad():
            actions, log_probs = policy.act(obs)
        if actions.shape[0] != 1:
            raise ValueError(
                f"collect_episode requires batch=1, got {actions.shape[0]}"
            )
        chunk_log_probs.append(log_probs[0])
        obs, _reward, done, info = env.step(actions[0])
        if done:
            final_info = {k: float(v) for k, v in info.items() if isinstance(v, (int, float))}
        obs = obs.to(device)

    episode = env.get_episode()
    # merge_chunk_actions is dtype-agnostic; overlap-dedup applies to log_probs too
    episode.log_probs = merge_chunk_actions(chunk_log_probs, env.chunk_config)
    episode.info.update(final_info)
    return episode


def collect_episode_topk(
    env: CompressionEnv,
    policy: Policy,
    prompt: Prompt,
    target_ratio: float = 0.5,
) -> Episode:
    """Like collect_episode but uses deterministic top-K token selection."""
    device = next(policy.parameters()).device
    obs = env.reset(prompt)
    obs = obs.to(device)
    chunk_log_probs: list[torch.Tensor] = []
    final_info: dict[str, float] = {}
    done = False

    while not done:
        with torch.no_grad():
            actions, log_probs = policy.act_topk(obs, target_ratio)
        if actions.shape[0] != 1:
            raise ValueError(
                f"collect_episode_topk requires batch=1, got {actions.shape[0]}"
            )
        chunk_log_probs.append(log_probs[0])
        obs, _reward, done, info = env.step(actions[0])
        if done:
            final_info = {k: float(v) for k, v in info.items() if isinstance(v, (int, float))}
        obs = obs.to(device)

    episode = env.get_episode()
    episode.log_probs = merge_chunk_actions(chunk_log_probs, env.chunk_config)
    episode.info.update(final_info)
    return episode


def _empty_penalty(reward_fn: RewardFunction) -> float:
    """Scale empty-compression penalty by reward type.

    Must be significantly worse than any real compression to prevent
    policies (especially PPO) from converging on "drop everything".

    KL_DENSE per-token rewards with kl_coeff=0.01 produce episode
    rewards in [-3, -1]. Penalty = -500 * kl_coeff = -5.0 at default,
    well below the worst real compression (~-3.2).

    SparseReward and HybridReward terminal rewards are in [0, 1];
    -10.0 keeps "drop everything" well below any real outcome (e.g. additive
    at ratio=0 gives task_score - penalty, so empty must be worse).
    """
    if isinstance(reward_fn, KLDenseReward):
        return -500.0 * reward_fn.config.kl_coeff
    return -10.0


def _keepall_penalty(reward_fn: RewardFunction) -> float | None:
    """Penalty for keeping all tokens (ratio > 0.99).

    Only applies to KLDenseReward, which has no natural compression
    incentive (keeping all tokens yields reward=0). Sparse and hybrid
    rewards can still generate valid LLM output at high ratios, so
    they should compute rewards normally.

    Scale mirrors _empty_penalty: -500 * kl_coeff puts the penalty well
    below worst real compression (~-3.2 at kl_coeff=0.01).

    Returns None for reward types that don't need this penalty.
    """
    if isinstance(reward_fn, KLDenseReward):
        return -500.0 * reward_fn.config.kl_coeff
    return None


def score_episode(
    episode: Episode,
    reward_fn: RewardFunction,
    frozen_llm: FrozenLLM | None,
    kl_cache: KLCache | None = None,
) -> str | None:
    """Compute reward and write it into episode in-place.

    For dense rewards: caches full-prompt logits, computes per-token reward,
    scatters into (seq_len,) via keep_mask.
    For sparse rewards: generates LLM output, computes scalar terminal reward.

    Short-circuits with a penalty if the policy drops all tokens (empty
    compression) or keeps all tokens under KLDenseReward (no compression
    incentive). Sparse/hybrid rewards at high ratios proceed normally.

    Returns the LLM output string (for reward hacking monitoring), or None if
    the episode was short-circuited by a penalty.
    """
    if frozen_llm is None:
        raise ValueError(
            "frozen_llm is required to score episodes; "
            "ensure reward type is configured with LLM access"
        )

    # token_ids is 1-D (compressed_len,); shape[0] is the token count
    if episode.compressed.token_ids.shape[0] == 0:
        episode.terminal_reward = _empty_penalty(reward_fn)
        return None

    if episode.compressed.compression_ratio > 0.99:
        penalty = _keepall_penalty(reward_fn)
        if penalty is not None:
            episode.terminal_reward = penalty
            return None

    status = "completed"
    error_msg: str | None = None
    try:
        if kl_cache is not None:
            kl_cache.cache_full_prompt(episode.prompt)
        llm_output = None
        if not isinstance(reward_fn, KLDenseReward):
            llm_output = frozen_llm.generate(
                episode.compressed.token_ids.unsqueeze(0),
            )
        reward_input = RewardInput(
            original=episode.prompt,
            compressed=episode.compressed,
            llm_output=llm_output,
        )
        reward = reward_fn.compute(reward_input)
        _apply_reward(episode, reward, reward_fn)
        return llm_output
    finally:
        if kl_cache is not None:
            kl_cache.clear()


def _apply_reward(
    episode: Episode,
    reward: torch.Tensor,
    reward_fn: RewardFunction,
) -> None:
    """Write reward into episode fields.

    Dense: scatter (compressed_len,) into (seq_len,) via keep_mask.
    Sparse: set terminal_reward scalar.
    """
    if reward_fn.is_dense():
        full_rewards = torch.zeros_like(episode.actions, dtype=torch.float)
        full_rewards[episode.compressed.keep_mask.bool()] = reward.float()
        episode.rewards = full_rewards
        terminal = reward_fn.terminal_scalar()
        if terminal is not None:
            episode.terminal_reward = terminal.item()
        else:
            episode.terminal_reward = 0.0
    else:
        episode.terminal_reward = reward.item()


def _compute_actual_f1(llm_output: str, metadata: dict[str, Any]) -> float | None:
    """Compute F1 between LLM output and ground-truth answers.

    Used to monitor reward hacking: if the learned reward increases but
    actual_f1 stays flat or drops, the policy is exploiting the reward model.

    Returns None for unanswerable questions (empty answer_texts) to avoid
    polluting the metric with systematic zeros.
    """
    answer_texts: list[str] = metadata.get("answer_texts", [])
    if not answer_texts:
        return None
    return max(compute_f1(llm_output, ans) for ans in answer_texts)


@dataclass
class Components:
    """Bundle of initialized training components."""

    env: CompressionEnv
    policy: Policy
    algorithm: Algorithm
    reward_fn: RewardFunction
    train_prompts: list[Prompt]
    val_prompts: list[Prompt]
    frozen_llm: FrozenLLM | None
    kl_cache: KLCache | None


def _create_llm_components(
    config: ExperimentConfig,
) -> tuple[FrozenLLM | None, KLCache | None]:
    """Create FrozenLLM and KLCache based on reward type."""
    needs_llm = config.reward.reward_type in (
        RewardType.SPARSE, RewardType.KL_DENSE, RewardType.HYBRID, RewardType.LEARNED,
    )
    if not needs_llm:
        return None, None

    from src.llm.frozen_llm import FrozenLLM
    frozen_llm = FrozenLLM(config.llm, device=config.train.device)

    kl_cache = None
    if config.reward.reward_type in (RewardType.KL_DENSE, RewardType.HYBRID):
        from src.llm.kl_cache import KLCache
        kl_cache = KLCache(frozen_llm)

    return frozen_llm, kl_cache


def init_components(config: ExperimentConfig) -> Components:
    """Create all training components from config."""
    frozen_llm, kl_cache = _create_llm_components(config)
    reward_fn = create_reward(config.reward, kl_cache=kl_cache)
    chunk_config = ChunkConfig(
        chunk_size=config.data.chunk_size,
        overlap=config.data.chunk_overlap,
    )
    env = CompressionEnv(
        chunk_config=chunk_config,
        target_ratio=config.reward.target_compression_ratio,
        min_ratio=config.reward.min_compression_ratio,
        min_ratio_mode=config.reward.min_ratio_enforcement_mode,
        min_ratio_soft_fraction=config.reward.min_ratio_soft_fraction,
        min_ratio_selection_strategy=config.reward.min_ratio_selection_strategy,
    )
    policy = create_policy(config.policy)
    policy = policy.to(config.train.device)
    algorithm = create_algorithm(policy, config.algo)
    train_prompts, val_prompts = load_data(config.data, tokenizer_model=config.llm.model_name)
    if not train_prompts:
        raise ValueError("No training prompts loaded; check DataConfig")
    if not val_prompts:
        raise ValueError("No validation prompts loaded; check DataConfig")

    return Components(
        env=env, policy=policy, algorithm=algorithm,
        reward_fn=reward_fn, train_prompts=train_prompts,
        val_prompts=val_prompts, frozen_llm=frozen_llm,
        kl_cache=kl_cache,
    )


@dataclass
class _LogContext:
    """Bundle of per-episode data needed for logging."""

    ep_idx: int
    episode: Episode
    metrics: dict[str, float]
    config: ExperimentConfig
    llm_output: str | None
    prompt: Prompt
    logger: WandbLogger | None


def _log_episode(ctx: _LogContext) -> None:
    """Build log dict, print summary, push to W&B."""
    reward_scalar = ctx.episode.terminal_reward
    if ctx.config.reward.reward_type in (RewardType.KL_DENSE, RewardType.HYBRID):
        reward_scalar = ctx.episode.rewards.sum().item() + ctx.episode.terminal_reward
    log_dict: dict[str, float] = {
        **ctx.metrics,
        "reward": reward_scalar,
        "compression_ratio": ctx.episode.compressed.compression_ratio,
    }
    if (
        ctx.config.reward.reward_type == RewardType.LEARNED
        and ctx.llm_output is not None
    ):
        actual_f1 = _compute_actual_f1(ctx.llm_output, ctx.prompt.metadata)
        if actual_f1 is not None:
            log_dict["actual_f1"] = actual_f1

    metric_str = " | ".join(f"{k}={v:.4f}" for k, v in ctx.metrics.items())
    actual_f1_str = f" | actual_f1={log_dict['actual_f1']:.4f}" if "actual_f1" in log_dict else ""
    print(
        f"ep {ctx.ep_idx + 1}/{ctx.config.train.num_episodes} | "
        f"reward={reward_scalar:.4f} | "
        f"ratio={ctx.episode.compressed.compression_ratio:.4f}"
        f"{actual_f1_str}"
        f"{' | ' + metric_str if metric_str else ''}",
        file=sys.stderr,
    )
    if ctx.logger is not None:
        ctx.logger.log_metrics(log_dict, step=ctx.ep_idx + 1, commit=False)
        ctx.logger.log_episode(ctx.episode, step=ctx.ep_idx + 1, commit=True)
    elif wandb is not None and getattr(wandb, "run", None) is not None:
        wandb.log(log_dict, step=ctx.ep_idx + 1)




def _to_primitive(value: Any) -> Any:
    """Convert config/dataclass values to JSON-serializable primitives."""
    if is_dataclass(value):
        return {k: _to_primitive(v) for k, v in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _to_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_primitive(v) for v in value]
    return value


def _git_commit() -> str | None:
    """Return current git commit hash if available."""
    status = "completed"
    error_msg: str | None = None
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _prepare_run_artifacts(config: ExperimentConfig) -> dict[str, Any]:
    """Persist resolved config and run-start manifest records."""
    run_id = f"{config.experiment_name}-seed{config.train.seed}-{int(datetime.now(tz=timezone.utc).timestamp())}"
    if config.train.output_dir == "outputs":
        config.train.output_dir = f"outputs/{config.experiment_name}/seed_{config.train.seed}"
    out_dir = Path(config.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved = _to_primitive(config)
    runtime = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "hostname": platform.node(),
        "git_commit": _git_commit(),
    }
    resolved_payload = {
        "run_id": run_id,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "config": resolved,
        "runtime": runtime,
    }
    _write_json(out_dir / "resolved_config.json", resolved_payload)

    run_start = {
        "event": "run_start",
        "run_id": run_id,
        "experiment_name": config.experiment_name,
        "seed": config.train.seed,
        "output_dir": str(out_dir),
        "started_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    _append_jsonl(out_dir / "run_manifest.jsonl", run_start)
    _append_jsonl(Path("outputs") / "run_manifest.jsonl", run_start)
    return {"run_id": run_id, "output_dir": out_dir}


def _finalize_run_artifacts(run_ctx: dict[str, Any], status: str, error: str | None) -> None:
    """Append run-end manifest records."""
    event = {
        "event": "run_end",
        "run_id": run_ctx["run_id"],
        "status": status,
        "ended_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if error is not None:
        event["error"] = error
    _append_jsonl(Path(run_ctx["output_dir"]) / "run_manifest.jsonl", event)
    _append_jsonl(Path("outputs") / "run_manifest.jsonl", event)

def _set_seeds(seed: int) -> None:
    """Initialize all RNG seeds for reproducibility."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _validate_config(config: ExperimentConfig) -> None:
    """Fail fast on invalid or unstable training settings."""
    reward = config.reward
    train = config.train
    algo = config.algo

    if not 0.0 <= reward.min_compression_ratio <= 1.0:
        raise ValueError("reward.min_compression_ratio must be in [0, 1]")
    if not 0.0 <= reward.quality_threshold <= 1.0:
        raise ValueError("reward.quality_threshold must be in [0, 1]")
    if reward.initial_quality_threshold is not None and not 0.0 <= reward.initial_quality_threshold <= 1.0:
        raise ValueError("reward.initial_quality_threshold must be in [0, 1]")
    if reward.threshold_warmup_episodes < 0:
        raise ValueError("reward.threshold_warmup_episodes must be >= 0")
    if reward.min_ratio_soft_fraction < 0.0 or reward.min_ratio_soft_fraction > 1.0:
        raise ValueError("reward.min_ratio_soft_fraction must be in [0, 1]")
    if train.update_batch_episodes < 1:
        raise ValueError("train.update_batch_episodes must be >= 1")
    if algo.num_epochs < 1:
        raise ValueError("algo.num_epochs must be >= 1")
    if (
        reward.min_ratio_enforcement_mode == MinRatioEnforcementMode.OFF
        and reward.min_compression_ratio > 0.0
    ):
        print(
            "warning: min_ratio_enforcement_mode=OFF; min_compression_ratio will only be enforced by episode filtering",
            file=sys.stderr,
        )
    if (
        algo.algo_type != AlgoType.PPO
        and algo.terminal_reward_distribution != TerminalRewardDistribution.LAST_KEPT
    ):
        raise ValueError(
            "algo.terminal_reward_distribution currently applies only to PPO; use LAST_KEPT for other algorithms"
        )
    if (
        config.policy.arch == PolicyArch.TINYLLAMA
        and config.policy.pretrained_name != config.llm.model_name
    ):
        raise ValueError(
            "For TINYLLAMA policy, policy.pretrained_name must match llm.model_name so token IDs map to meaningful embeddings."
        )


def _effective_quality_threshold(config: ExperimentConfig, ep_idx: int) -> float | None:
    """Return scheduled threshold for sparse-threshold runs, else None."""
    reward = config.reward
    if reward.sparse_reward_mode != SparseRewardMode.THRESHOLD:
        return None
    if reward.threshold_warmup_episodes <= 0 or reward.initial_quality_threshold is None:
        return reward.quality_threshold
    progress = min(1.0, (ep_idx + 1) / reward.threshold_warmup_episodes)
    start = reward.initial_quality_threshold
    end = reward.quality_threshold
    return start + (end - start) * progress


def _reward_component_metrics(config: ExperimentConfig, episode: Episode) -> dict[str, float]:
    """Extract reward component diagnostics for sparse rewards."""
    if config.reward.reward_type != RewardType.SPARSE:
        return {}
    ratio = episode.compressed.compression_ratio
    mode = config.reward.sparse_reward_mode
    if mode == SparseRewardMode.ADDITIVE:
        penalty = config.reward.compression_penalty * abs(
            ratio - config.reward.target_compression_ratio
        )
        scale = max(config.reward.task_score_scale, 1e-8)
        task_est = (episode.terminal_reward + penalty - config.reward.reward_bias) / scale
        return {
            "sparse_penalty_component": penalty,
            "sparse_task_score_estimate": task_est,
        }
    if mode == SparseRewardMode.MULTIPLICATIVE:
        return {"sparse_keep_discount": 1.0 - ratio}
    return {}


def _threshold_observability(reward_fn: RewardFunction) -> dict[str, float]:
    """Collect threshold-mode diagnostics when available."""
    if not isinstance(reward_fn, SparseReward):
        return {}
    metrics: dict[str, float] = {}
    if reward_fn.last_task_score is not None:
        metrics["task_score"] = reward_fn.last_task_score
    if reward_fn.last_threshold_pass is not None:
        metrics["threshold_pass"] = float(reward_fn.last_threshold_pass)
    if reward_fn.last_effective_threshold is not None:
        metrics["effective_quality_threshold"] = reward_fn.last_effective_threshold
    return metrics


@dataclass
class _EarlyStopMonitor:
    """Stops training when the policy has collapsed (low entropy + high ratio)."""

    window: int
    entropy_min: float
    ratio_max: float
    _counter: int = field(default=0, init=False, repr=False)

    @staticmethod
    def from_config(cfg: TrainConfig) -> _EarlyStopMonitor:
        return _EarlyStopMonitor(
            window=cfg.early_stop_window,
            entropy_min=cfg.early_stop_entropy_min,
            ratio_max=cfg.early_stop_ratio_max,
        )

    def check(self, entropy: float, ratio: float) -> bool:
        if self.window <= 0:
            return False
        if entropy < self.entropy_min and ratio > self.ratio_max:
            self._counter += 1
        else:
            self._counter = 0
        return self._counter >= self.window


def train(config: ExperimentConfig) -> None:
    """Main training loop.

    Creates components, collects episodes, computes rewards, updates policy.
    Seeds RNGs from config.train.seed for reproducibility.
    Logs to W&B (if wandb_project is set) and saves checkpoints at configured intervals.
    """
    _validate_config(config)
    _set_seeds(config.train.seed)
    run_ctx = _prepare_run_artifacts(config)
    c = init_components(config)
    logger = WandbLogger(config) if config.train.wandb_project else None

    start_ep = 0
    if config.train.resume_from:
        ckpt_state = CheckpointState(c.algorithm, step=0)  # step ignored on load
        start_ep = load_checkpoint(ckpt_state, Path(config.train.resume_from))

    early_stop = _EarlyStopMonitor.from_config(config.train)

    status = "completed"
    error_msg: str | None = None
    try:
        pending_batch: list[Episode] = []
        pending_ratios: list[float] = []
        pending_forced_counts: list[float] = []
        pending_threshold_passes: list[float] = []
        prompt_index = 0
        for ep_idx in range(start_ep, config.train.num_episodes):
            if config.train.sample_with_replacement:
                prompt = random.choice(c.train_prompts)
            else:
                prompt = c.train_prompts[prompt_index]
                prompt_index = (prompt_index + 1) % len(c.train_prompts)
            episode = c.algorithm.collect_episode(c.env, prompt)
            if episode is None:
                episode = collect_episode(c.env, c.policy, prompt)
            ratio = episode.compressed.compression_ratio
            scheduled_threshold = _effective_quality_threshold(config, ep_idx)
            if isinstance(c.reward_fn, SparseReward):
                c.reward_fn.current_quality_threshold = scheduled_threshold
            llm_output = score_episode(episode, c.reward_fn, c.frozen_llm, c.kl_cache)
            threshold_metrics = _threshold_observability(c.reward_fn)
            episode.info.update(threshold_metrics)
            pending_batch.append(episode)
            pending_ratios.append(ratio)
            pending_forced_counts.append(float(episode.info.get("forced_keep_count", 0.0)))
            if "threshold_pass" in threshold_metrics:
                pending_threshold_passes.append(threshold_metrics["threshold_pass"])
            should_update = (
                len(pending_batch) >= config.train.update_batch_episodes
                or ep_idx + 1 == config.train.num_episodes
            )
            if not should_update:
                continue
            metrics = c.algorithm.update(pending_batch)
            metrics["batch_size_episodes"] = float(len(pending_batch))
            metrics["batch_mean_ratio"] = sum(pending_ratios) / len(pending_ratios)
            metrics["batch_mean_forced_keeps"] = (
                sum(pending_forced_counts) / len(pending_forced_counts)
            )
            if pending_threshold_passes:
                metrics["batch_threshold_pass_rate"] = (
                    sum(pending_threshold_passes) / len(pending_threshold_passes)
                )
            if "task_score" in episode.info:
                metrics["task_score"] = float(episode.info["task_score"])
            if "threshold_pass" in episode.info:
                metrics["threshold_pass"] = float(episode.info["threshold_pass"])
            if "effective_quality_threshold" in episode.info:
                metrics["effective_quality_threshold"] = float(
                    episode.info["effective_quality_threshold"]
                )
            metrics.update(_reward_component_metrics(config, episode))
            pending_batch.clear()
            pending_ratios.clear()
            pending_forced_counts.clear()
            pending_threshold_passes.clear()

            entropy = metrics.get("entropy", 1.0)  # high default → never triggers early stop
            if early_stop.check(entropy, ratio):
                _log_episode(_LogContext(
                    ep_idx=ep_idx, episode=episode, metrics=metrics,
                    config=config, llm_output=llm_output,
                    prompt=prompt, logger=logger,
                ))
                print(
                    f"early stop at ep {ep_idx + 1}: "
                    f"entropy={entropy:.4f} ratio={ratio:.4f}",
                    file=sys.stderr,
                )
                break

            if (ep_idx + 1) % config.train.log_every == 0:
                _log_episode(_LogContext(
                    ep_idx=ep_idx, episode=episode, metrics=metrics,
                    config=config, llm_output=llm_output,
                    prompt=prompt, logger=logger,
                ))

            if (ep_idx + 1) % config.train.checkpoint_every == 0:
                save_checkpoint(
                    CheckpointState(c.algorithm, ep_idx + 1, metadata={
                        "run_id": run_ctx["run_id"],
                        "experiment_name": config.experiment_name,
                        "seed": config.train.seed,
                        "resolved_config_path": str(Path(config.train.output_dir) / "resolved_config.json"),
                        "git_commit": _git_commit(),
                    }),
                    Path(config.train.output_dir) / "checkpoints" / f"step_{ep_idx + 1}.pt",
                )
    except Exception as exc:
        status = "failed"
        error_msg = str(exc)
        raise
    finally:
        if logger is not None:
            logger.finish()
        _finalize_run_artifacts(run_ctx, status, error_msg)


def main() -> None:
    """CLI entry point: parse args and call train."""
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
