"""Evaluation loop and Pareto curve generation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import Config
from src.data.types import Prompt
from src.policy.base import Policy
from src.reward.hybrid import HybridReward
from src.reward.metrics import compute_rouge, compute_task_score
from src.train import _get_original_output, collect_episode, init_components, score_episode
from src.tracking.checkpointing import CheckpointState, load_checkpoint

from src.env.compression_env import CompressionEnv
from src.llm.frozen_llm import FrozenLLM
from src.llm.kl_cache import KLCache


@dataclass
class EvalContext:
    """Bundle of components needed for evaluation."""

    env: CompressionEnv
    reward_fn: HybridReward
    frozen_llm: FrozenLLM
    kl_cache: KLCache


@dataclass
class SweepConfig:
    """Evaluation context plus sweep-specific parameters."""

    ctx: EvalContext
    ratios: list[float]


def _eval_prompt(
    ctx: EvalContext,
    policy: Policy,
    prompt: Prompt,
    device: str = "cpu",
    sparse_only: bool = False,
) -> dict[str, float]:
    """Run one greedy episode and return per-prompt metrics."""
    episode = collect_episode(ctx.env, policy, prompt, device=device, greedy=True)
    result = score_episode(episode, ctx.reward_fn, ctx.frozen_llm, ctx.kl_cache, sparse_only=sparse_only)

    task_score = 0.0
    if result.llm_output is not None:
        task_score = compute_task_score(result.llm_output, prompt.metadata)

    faithfulness = 0.0
    if result.llm_output and result.original_llm_output:
        faithfulness = compute_rouge(result.llm_output, result.original_llm_output)["rougeL"]

    uncompressed_f1 = 0.0
    if result.original_llm_output is not None:
        uncompressed_f1 = compute_task_score(result.original_llm_output, prompt.metadata)

    return {
        "terminal_reward": episode.terminal_reward,
        "compression_ratio": episode.compressed.compression_ratio,
        "task_score": task_score,
        "faithfulness_rougeL": faithfulness,
        "uncompressed_f1": uncompressed_f1,
    }


def evaluate(
    policy: Policy,
    prompts: list[Prompt],
    ctx: EvalContext,
    device: str = "cpu",
    sparse_only: bool = False,
) -> dict[str, float]:
    """Run policy on validation set and return aggregate metrics."""
    policy.eval()
    try:
        totals: dict[str, float] = {
            "terminal_reward": 0.0, "compression_ratio": 0.0, "task_score": 0.0,
            "faithfulness_rougeL": 0.0, "uncompressed_f1": 0.0,
        }
        with torch.no_grad():
            for prompt in prompts:
                metrics = _eval_prompt(ctx, policy, prompt, device=device, sparse_only=sparse_only)
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
    device: str = "cpu",
) -> dict[float, dict[str, float]]:
    """Evaluate at multiple target ratios for Pareto curves."""
    original_ratio = sweep.ctx.env.target_ratio
    results: dict[float, dict[str, float]] = {}

    try:
        for ratio in sweep.ratios:
            sweep.ctx.env.target_ratio = ratio
            results[ratio] = evaluate(policy, prompts, sweep.ctx, device=device)
    finally:
        sweep.ctx.env.target_ratio = original_ratio

    return results


@dataclass
class _EvalArgs:
    checkpoint: str
    sweep_ratios: list[float] | None


def _parse_eval_args() -> _EvalArgs:
    """Parse eval-specific args; strips them from sys.argv for parse_args()."""
    parser = argparse.ArgumentParser(description="Evaluate a trained policy")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint .pt file"
    )
    parser.add_argument(
        "--sweep-ratios", type=float, nargs="+", default=None,
        help="Compression ratios for Pareto sweep (e.g. 0.3 0.4 0.5 0.6 0.7)",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return _EvalArgs(checkpoint=args.checkpoint, sweep_ratios=args.sweep_ratios)


def _print_results(metrics: dict[str, float], step: int) -> None:
    """Print evaluation results table to stderr."""
    print(f"\n=== Evaluation Results (step {step}) ===", file=sys.stderr)
    for key, val in metrics.items():
        print(f"  {key:25s} {val:.4f}", file=sys.stderr)


def _print_sweep_results(
    results: dict[float, dict[str, float]], step: int,
) -> None:
    """Print Pareto sweep results as a table."""
    print(f"\n=== Pareto Sweep Results (step {step}) ===", file=sys.stderr)
    header = f"  {'ratio':>8s}  {'task_score':>12s}  {'terminal_reward':>16s}  {'compression_ratio':>18s}"
    print(header, file=sys.stderr)
    print(f"  {'─' * len(header.strip())}", file=sys.stderr)
    for ratio in sorted(results):
        m = results[ratio]
        print(
            f"  {ratio:8.2f}  {m['task_score']:12.4f}  "
            f"{m['terminal_reward']:16.4f}  {m['compression_ratio']:18.4f}",
            file=sys.stderr,
        )


def main() -> None:
    """CLI entry point for standalone evaluation."""
    eval_args = _parse_eval_args()

    from src.config import parse_args
    config = parse_args()
    torch.manual_seed(config.seed)

    c = init_components(config)
    step = load_checkpoint(
        CheckpointState(c.policy, c.algorithm, 0),
        Path(eval_args.checkpoint),
    )

    ctx = EvalContext(
        env=c.env, reward_fn=c.reward_fn,
        frozen_llm=c.frozen_llm, kl_cache=c.kl_cache,
    )

    if eval_args.sweep_ratios:
        sweep = SweepConfig(ctx=ctx, ratios=eval_args.sweep_ratios)
        results = sweep_compression_ratios(c.policy, c.val_prompts, sweep, device=config.device)
        _print_sweep_results(results, step)
        if config.wandb_project:
            from src.tracking.wandb_logger import WandbLogger
            logger = WandbLogger(config)
            for ratio, metrics in results.items():
                logger.log_metrics(
                    {f"sweep/{k}": v for k, v in metrics.items()},
                    step=int(ratio * 100),
                )
            logger.finish()
    else:
        metrics = evaluate(c.policy, c.val_prompts, ctx, device=config.device)
        _print_results(metrics, step)
        if config.wandb_project:
            from src.tracking.wandb_logger import WandbLogger
            logger = WandbLogger(config)
            logger.log_metrics({f"eval/{k}": v for k, v in metrics.items()}, step=step)
            logger.finish()


if __name__ == "__main__":
    main()
