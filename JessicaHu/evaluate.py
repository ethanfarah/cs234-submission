"""Evaluation loop and Pareto curve generation."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.data.types import Prompt
from src.policy.base import Policy
from src.reward.metrics import compute_f1
from src.reward.sparse import _compute_task_score
from src.train import collect_episode, init_components, score_episode
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


def _eval_prompt(ctx: EvalContext, policy: Policy, prompt: Prompt) -> dict[str, float]:
    """Run one episode and return per-prompt metrics."""
    episode = collect_episode(ctx.env, policy, prompt)
    llm_output = score_episode(episode, ctx.reward_fn, ctx.frozen_llm, ctx.kl_cache)

    # task_score is 0.0 for empty-compression episodes (llm_output=None)
    # because the policy produced no output for the LLM to work with
    task_score = 0.0
    f1 = 0.0
    if llm_output is not None:
        task_score = _compute_task_score(llm_output, prompt.metadata)
        answer_texts: list[str] = prompt.metadata.get("answer_texts", [])
        if answer_texts:
            f1 = max(compute_f1(llm_output, ans) for ans in answer_texts)

    return {
        "terminal_reward": episode.terminal_reward,
        "compression_ratio": episode.compressed.compression_ratio,
        "task_score": task_score,
        "f1": f1,
    }


def evaluate(
    policy: Policy,
    prompts: list[Prompt],
    ctx: EvalContext,
) -> dict[str, float]:
    """Run policy on validation set and return aggregate metrics."""
    policy.eval()
    try:
        totals: dict[str, float] = {
            "terminal_reward": 0.0,
            "compression_ratio": 0.0,
            "task_score": 0.0,
            "f1": 0.0,
        }
        with torch.no_grad():
            for prompt in prompts:
                metrics = _eval_prompt(ctx, policy, prompt)
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


def _parse_eval_args() -> str:
    """Parse --checkpoint from argv; strips it from sys.argv for parse_args()."""
    parser = argparse.ArgumentParser(description="Evaluate a trained policy")
    parser.add_argument(
        "--checkpoint", required=True, help="Path to checkpoint .pt file"
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return args.checkpoint


def _print_results(metrics: dict[str, float], step: int) -> None:
    """Print evaluation results table to stderr."""
    print(f"\n=== Evaluation Results (step {step}) ===", file=sys.stderr)
    for key, val in metrics.items():
        print(f"  {key:25s} {val:.4f}", file=sys.stderr)


def main() -> None:
    """CLI entry point for standalone evaluation."""
    checkpoint_path = _parse_eval_args()

    from src.config import parse_args
    config = parse_args()
    torch.manual_seed(config.train.seed)

    c = init_components(config)
    step = load_checkpoint(
        CheckpointState(c.algorithm, 0),
        Path(checkpoint_path),
    )

    if c.frozen_llm is None:
        raise ValueError("evaluate requires a frozen LLM; check reward_type config")

    ctx = EvalContext(
        env=c.env, reward_fn=c.reward_fn,
        frozen_llm=c.frozen_llm, kl_cache=c.kl_cache,
    )
    metrics = evaluate(c.policy, c.val_prompts, ctx)
    _print_results(metrics, step)

    if config.train.wandb_project:
        from src.tracking.wandb_logger import WandbLogger
        logger = WandbLogger(config)
        logger.log_metrics({f"eval/{k}": v for k, v in metrics.items()}, step=step)
        logger.finish()


if __name__ == "__main__":
    main()
