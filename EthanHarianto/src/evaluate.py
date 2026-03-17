"""Evaluation loop and Pareto curve generation."""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover
    np = None  # type: ignore[assignment]

from src.data.types import Prompt
from src.policy.base import Policy
from src.reward.sparse import _compute_task_score
from src.train import collect_episode, collect_episode_topk, init_components, score_episode
from src.tracking.checkpointing import CheckpointState, load_checkpoint

if TYPE_CHECKING:
    from src.env.compression_env import CompressionEnv
    from src.llm.frozen_llm import FrozenLLM
    from src.llm.kl_cache import KLCache
    from src.reward.base import RewardFunction


@dataclass
class EvalContext:
    """Bundle of components needed for evaluation."""

    env: CompressionEnv
    reward_fn: RewardFunction
    frozen_llm: FrozenLLM | None
    kl_cache: KLCache | None


@dataclass
class SweepConfig:
    """Evaluation context plus sweep-specific parameters."""

    ctx: EvalContext
    ratios: list[float]




def _set_eval_seed(seed: int) -> None:
    """Set RNG seeds for reproducible evaluation."""
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _eval_prompt(ctx: EvalContext, policy: Policy, prompt: Prompt, topk: bool = False, target_ratio: float = 0.5) -> dict[str, float]:
    """Run one episode and return per-prompt metrics."""
    if topk:
        episode = collect_episode_topk(ctx.env, policy, prompt, target_ratio=target_ratio)
    else:
        episode = collect_episode(ctx.env, policy, prompt)
    llm_output = score_episode(episode, ctx.reward_fn, ctx.frozen_llm, ctx.kl_cache)

    # Always compute downstream task score using generated output, independent of
    # training reward type. KL-dense training may skip generation in score_episode.
    task_score = 0.0
    if llm_output is None and ctx.frozen_llm is not None and episode.compressed.token_ids.shape[0] > 0:
        llm_output = ctx.frozen_llm.generate(episode.compressed.token_ids.unsqueeze(0))
    if llm_output is not None:
        task_score = _compute_task_score(llm_output, prompt.metadata)

    return {
        "terminal_reward": episode.terminal_reward,
        "compression_ratio": episode.compressed.compression_ratio,
        "task_score": task_score,
    }


def evaluate(
    policy: Policy,
    prompts: list[Prompt],
    ctx: EvalContext,
    topk: bool = False,
    target_ratio: float = 0.5,
) -> dict[str, float]:
    """Run policy on validation set and return aggregate metrics."""
    policy.eval()
    try:
        totals: dict[str, float] = {
            "terminal_reward": 0.0, "compression_ratio": 0.0, "task_score": 0.0,
        }
        with torch.no_grad():
            for prompt in prompts:
                metrics = _eval_prompt(ctx, policy, prompt, topk=topk, target_ratio=target_ratio)
                for k, v in metrics.items():
                    totals[k] += v
        n = max(len(prompts), 1)
        return {k: v / n for k, v in totals.items()}
    finally:
        policy.train()


def sweep_compression_ratios(
    policy: Policy,
    prompts: list[Prompt],
    sweep: SweepConfig,
) -> dict[float, dict[str, float]]:
    """Evaluate at multiple target ratios for Pareto curves."""
    original_ratio = sweep.ctx.env._target_ratio
    results: dict[float, dict[str, float]] = {}

    try:
        for ratio in sweep.ratios:
            sweep.ctx.env._target_ratio = ratio
            results[ratio] = evaluate(policy, prompts, sweep.ctx)
    finally:
        sweep.ctx.env._target_ratio = original_ratio

    return results


def _parse_eval_args() -> tuple[str, str, int, bool, float]:
    """Parse eval-specific args; strips them from argv for parse_args()."""
    parser = argparse.ArgumentParser(description="Evaluate a trained policy")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--eval-split", default="validation", help="Split to evaluate (train/validation/test)")
    parser.add_argument("--eval-seeds", type=int, default=1, help="Number of seeds to average during evaluation")
    parser.add_argument("--topk", action="store_true", help="Use deterministic top-K selection instead of sampling")
    parser.add_argument("--eval-ratio", type=float, default=0.0, help="Override target ratio for eval (0=use config)")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args.checkpoint, args.eval_split, max(1, args.eval_seeds), args.topk, args.eval_ratio


def _print_results(metrics: dict[str, float], step: int) -> None:
    """Print evaluation results table to stderr."""
    print(f"\n=== Evaluation Results (step {step}) ===", file=sys.stderr)
    for key, val in metrics.items():
        print(f"  {key:25s} {val:.4f}", file=sys.stderr)


def _normalize_enum_keys(d: dict, path: str = "") -> None:
    """In-place normalize enum string values to uppercase for OmegaConf validation."""
    enum_keys = {
        "algo_type", "arch", "reward_type", "sparse_reward_mode", "hybrid_mode",
        "kl_direction", "min_ratio_enforcement_mode", "min_ratio_selection_strategy",
        "terminal_reward_distribution", "baseline_type",
    }
    for k, v in list(d.items()):
        key_path = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            _normalize_enum_keys(v, key_path)
        elif k in enum_keys and isinstance(v, str):
            d[k] = v.upper()


def _load_config_for_checkpoint(checkpoint_path: str):
    """Load config from run's resolved_config.json if present, else parse_args()."""
    from src.config import ExperimentConfig, parse_args
    from omegaconf import OmegaConf

    cp = Path(checkpoint_path).resolve()
    run_dir = cp.parent.parent if cp.name.startswith("step_") else cp.parent
    resolved = run_dir / "resolved_config.json"
    if resolved.exists():
        import json
        data = json.loads(resolved.read_text())
        cfg_dict = data.get("config", data)
        _normalize_enum_keys(cfg_dict)
        base = OmegaConf.structured(ExperimentConfig)
        override = OmegaConf.create(cfg_dict)
        merged = OmegaConf.merge(base, override)
        return OmegaConf.to_object(merged)
    return parse_args()


def main() -> None:
    """CLI entry point for standalone evaluation."""
    checkpoint_path, eval_split, eval_seeds, use_topk, eval_ratio = _parse_eval_args()

    config = _load_config_for_checkpoint(checkpoint_path)
    if eval_split and eval_split != "train":
        config.data.val_split = eval_split

    c = init_components(config)
    step = load_checkpoint(
        CheckpointState(c.algorithm, 0),
        Path(checkpoint_path),
    )

    if c.frozen_llm is None:
        raise ValueError("evaluate requires a frozen LLM; check reward_type config")

    prompts = c.train_prompts if eval_split == "train" else c.val_prompts
    ctx = EvalContext(
        env=c.env, reward_fn=c.reward_fn,
        frozen_llm=c.frozen_llm, kl_cache=c.kl_cache,
    )

    target_ratio = eval_ratio if eval_ratio > 0 else config.reward.target_compression_ratio
    if use_topk:
        print(f"Using top-K deterministic selection at ratio={target_ratio}", file=sys.stderr)

    per_seed: list[dict[str, float]] = []
    for i in range(eval_seeds):
        _set_eval_seed(config.train.seed + i)
        per_seed.append(evaluate(c.policy, prompts, ctx, topk=use_topk, target_ratio=target_ratio))

    keys = per_seed[0].keys()
    metrics = {k: sum(m[k] for m in per_seed) / len(per_seed) for k in keys}
    metrics_std = {
        f"{k}_std": (sum((m[k] - metrics[k]) ** 2 for m in per_seed) / len(per_seed)) ** 0.5
        for k in keys
    }
    metrics = {**metrics, **metrics_std, "eval_seeds": float(eval_seeds)}

    _print_results(metrics, step)

    if config.train.wandb_project:
        from src.tracking.wandb_logger import WandbLogger
        logger = WandbLogger(config)
        logger.log_metrics({f"eval/{k}": v for k, v in metrics.items()}, step=step)
        logger.finish()


if __name__ == "__main__":
    main()
