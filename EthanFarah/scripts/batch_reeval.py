"""Re-evaluate saved checkpoints with greedy policy and max_new_tokens=16.

Initializes components once, then swaps checkpoint weights for each eval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from src.config import Config, parse_args
from src.evaluate import EvalContext, evaluate
from src.tracking.checkpointing import CheckpointState, load_checkpoint
from src.train import Components, init_components


def _parse_batch_args() -> list[Path]:
    """Parse --checkpoint args, return list of .pt paths."""
    parser = argparse.ArgumentParser(description="Batch re-evaluate checkpoints")
    parser.add_argument(
        "--checkpoint", type=str, nargs="+", required=True,
        help="One or more checkpoint .pt paths",
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    paths = [Path(p) for p in args.checkpoint]
    valid = [p for p in paths if p.exists()]
    missing = [p for p in paths if not p.exists()]
    for p in missing:
        print(f"WARNING: skipping {p} (not found)", file=sys.stderr)
    if not valid:
        raise FileNotFoundError("No valid checkpoint paths found")
    return valid


def _eval_checkpoint(
    path: Path, c: Components, config: Config,
) -> dict[str, float]:
    """Load weights from checkpoint and run greedy eval."""
    step = load_checkpoint(
        CheckpointState(c.policy, c.algorithm, 0), path,
    )
    ctx = EvalContext(
        env=c.env, reward_fn=c.reward_fn,
        frozen_llm=c.frozen_llm, kl_cache=c.kl_cache,
    )
    print(f"  evaluating {path.parent.parent.name} (step {step})...", file=sys.stderr)
    return evaluate(c.policy, c.val_prompts, ctx, device=config.device, sparse_only=True)


def _print_summary(results: list[tuple[Path, dict[str, float]]]) -> None:
    """Print results table to stderr."""
    header = (
        f"{'checkpoint':<45s}  {'faith_RL':>8s}  {'task_f1':>8s}  "
        f"{'uncomp_f1':>9s}  {'reward':>8s}  {'ratio':>8s}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}", file=sys.stderr)
    print(header, file=sys.stderr)
    print(sep, file=sys.stderr)
    for path, m in results:
        name = path.parent.parent.name
        print(
            f"{name:<45s}  {m.get('faithfulness_rougeL', 0):8.4f}  "
            f"{m['task_score']:8.4f}  {m.get('uncompressed_f1', 0):9.4f}  "
            f"{m['terminal_reward']:8.4f}  {m['compression_ratio']:8.4f}",
            file=sys.stderr,
        )
    print(sep, file=sys.stderr)


def main() -> None:
    """Init once, eval all checkpoints, print summary."""
    paths = _parse_batch_args()
    config = parse_args()
    config.llm.max_new_tokens = 16
    torch.manual_seed(config.seed)

    print(f"Initializing components (LLM + dataset)...", file=sys.stderr)
    c = init_components(config)
    print(f"Ready. Evaluating {len(paths)} checkpoint(s).\n", file=sys.stderr)

    results: list[tuple[Path, dict[str, float]]] = []
    for path in paths:
        metrics = _eval_checkpoint(path, c, config)
        results.append((path, metrics))

    _print_summary(results)


if __name__ == "__main__":
    main()
