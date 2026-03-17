from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

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
    PolicyArch,
    PolicyConfig,
    RewardConfig,
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
    if config.algo_type == AlgoType.REINFORCE:
        from src.algo.reinforce import REINFORCE
        return REINFORCE(policy, config)
    if config.algo_type == AlgoType.A2C:
        from src.algo.a2c import A2C
        return A2C(policy, config)
    if config.algo_type == AlgoType.PPO:
        from src.algo.ppo import PPO
        return PPO(policy, config)
    if config.algo_type == AlgoType.DQN:
        from src.algo.dqn import DQN
        return DQN(policy, config)
    if config.algo_type == AlgoType.BANDIT:
        from src.algo.bandit import ContextualBandit
        return ContextualBandit(policy, config)
    if config.algo_type == AlgoType.GRPO:
        from src.algo.grpo import GRPO
        return GRPO(policy, config)


def create_reward(
    config: RewardConfig,
    kl_cache: KLCache | None = None,
) -> RewardFunction:
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


def load_data(
    config: DataConfig,
    tokenizer_model: str | None = None,
) -> tuple[list[Prompt], list[Prompt]]:
    if config.dataset == "squad":
        loader = load_squad
    elif config.dataset == "meetingbank":
        loader = load_meetingbank

    base_kwargs: dict[str, Any] = dict(max_length=config.max_prompt_tokens)
    if tokenizer_model is not None:
        base_kwargs["model_name"] = tokenizer_model

    train_prompts = loader(
        split=config.train_split,
        max_samples=config.max_train_samples,
        **base_kwargs,
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
    device = next(policy.parameters()).device
    obs = env.reset(prompt)
    obs = obs.to(device)
    chunk_log_probs: list[torch.Tensor] = []
    done = False

    while not done:
        with torch.no_grad():
            actions, log_probs = policy.act(obs)
        chunk_log_probs.append(log_probs[0])
        obs, _reward, done, _info = env.step(actions[0])
        obs = obs.to(device)

    episode = env.get_episode()
    episode.log_probs = merge_chunk_actions(chunk_log_probs, env.chunk_config)
    return episode


def _empty_penalty(reward_fn: RewardFunction) -> float:
    if isinstance(reward_fn, KLDenseReward):
        return -500.0 * reward_fn.config.kl_coeff
    return -5.0


def _keepall_penalty(reward_fn: RewardFunction) -> float | None:
    if isinstance(reward_fn, KLDenseReward):
        return -500.0 * reward_fn.config.kl_coeff
    return None


def score_episode(
    episode: Episode,
    reward_fn: RewardFunction,
    frozen_llm: FrozenLLM | None,
    kl_cache: KLCache | None = None,
) -> str | None:
    if episode.compressed.token_ids.shape[0] == 0:
        episode.terminal_reward = _empty_penalty(reward_fn)
        return None

    if episode.compressed.compression_ratio > 0.99:
        penalty = _keepall_penalty(reward_fn)
        if penalty is not None:
            episode.terminal_reward = penalty
            return None

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
    answer_texts: list[str] = metadata.get("answer_texts", [])
    if not answer_texts:
        return None
    return max(compute_f1(llm_output, ans) for ans in answer_texts)


@dataclass
class Components:
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
    frozen_llm, kl_cache = _create_llm_components(config)
    reward_fn = create_reward(config.reward, kl_cache=kl_cache)
    chunk_config = ChunkConfig(
        chunk_size=config.data.chunk_size,
        overlap=config.data.chunk_overlap,
    )
    env = CompressionEnv(
        chunk_config=chunk_config,
        target_ratio=config.reward.target_compression_ratio,
    )
    policy = create_policy(config.policy)
    policy = policy.to(config.train.device)
    algorithm = create_algorithm(policy, config.algo)
    train_prompts, val_prompts = load_data(config.data, tokenizer_model=config.llm.model_name)
    return Components(
        env=env, policy=policy, algorithm=algorithm,
        reward_fn=reward_fn, train_prompts=train_prompts,
        val_prompts=val_prompts, frozen_llm=frozen_llm,
        kl_cache=kl_cache,
    )


@dataclass
class _LogContext:
    ep_idx: int
    episode: Episode
    metrics: dict[str, float]
    config: ExperimentConfig
    llm_output: str | None
    prompt: Prompt
    logger: WandbLogger | None


def _log_episode(ctx: _LogContext) -> None:
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


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class _EarlyStopMonitor:
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
    _set_seeds(config.train.seed)
    c = init_components(config)
    logger = WandbLogger(config) if config.train.wandb_project else None

    start_ep = 0
    if config.train.resume_from:
        ckpt_state = CheckpointState(c.algorithm, step=0)  # step ignored on load
        start_ep = load_checkpoint(ckpt_state, Path(config.train.resume_from))

    early_stop = _EarlyStopMonitor.from_config(config.train)

    try:
        for ep_idx in range(start_ep, config.train.num_episodes):
            prompt = random.choice(c.train_prompts)
            episode = c.algorithm.collect_episode(c.env, prompt)
            if episode is None:
                episode = collect_episode(c.env, c.policy, prompt)
            llm_output = score_episode(episode, c.reward_fn, c.frozen_llm, c.kl_cache)
            metrics = c.algorithm.update([episode])

            entropy = metrics.get("entropy", 1.0)  
            ratio = episode.compressed.compression_ratio
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
                    CheckpointState(c.algorithm, ep_idx + 1),
                    Path(config.train.output_dir) / "checkpoints" / f"step_{ep_idx + 1}.pt",
                )
    finally:
        if logger is not None:
            logger.finish()


def main() -> None:
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
