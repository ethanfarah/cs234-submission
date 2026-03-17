"""Orchestrate Phase 2 experiments: main training, ablation, and sweep."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

SEEDS = [42, 123, 456]
SWEEP_RATIOS = [0.3, 0.4, 0.5, 0.6, 0.7]
SWEEP_SEED = 42
OUTPUT_DIR = "outputs"


@dataclass
class TrainExperiment:
    """A single training run configuration."""

    experiment_name: str
    seed: int
    num_episodes: int = 5000
    eval_every: int = 250
    checkpoint_every: int = 500
    max_val_samples: int = 50
    extra_flags: list[str] = field(default_factory=list)


@dataclass
class SweepExperiment:
    """A compression ratio sweep evaluation."""

    checkpoint: str
    ratios: list[float]
    experiment_name: str


def _train_args(exp: TrainExperiment) -> list[str]:
    return [
        sys.executable, "-m", "src.train",
        "--experiment-name", exp.experiment_name,
        "--seed", str(exp.seed),
        "--num-episodes", str(exp.num_episodes),
        "--eval-every", str(exp.eval_every),
        "--checkpoint-every", str(exp.checkpoint_every),
        "--max-val-samples", str(exp.max_val_samples),
        "--wandb-project", "prompt-compression",
        *exp.extra_flags,
    ]


def _sweep_args(exp: SweepExperiment) -> list[str]:
    return [
        sys.executable, "-m", "src.evaluate",
        "--checkpoint", exp.checkpoint,
        "--sweep-ratios", *[str(r) for r in exp.ratios],
        "--experiment-name", exp.experiment_name,
    ]


def _run_command(args: list[str], dry_run: bool) -> None:
    cmd_str = " ".join(args)
    if dry_run:
        print(f"[DRY RUN] {cmd_str}", file=sys.stderr)
        return
    print(f"[RUN] {cmd_str}", file=sys.stderr)
    subprocess.run(args, check=True)


def _print_header(title: str) -> None:
    border = "=" * 60
    print(f"\n{border}", file=sys.stderr)
    print(f"  {title}", file=sys.stderr)
    print(f"{border}\n", file=sys.stderr)


def build_main_experiments() -> list[TrainExperiment]:
    return [
        TrainExperiment(
            experiment_name=f"bandit-hybrid-s{seed}",
            seed=seed,
        )
        for seed in SEEDS
    ]


def build_ablation_experiments() -> list[TrainExperiment]:
    return [
        TrainExperiment(
            experiment_name=f"bandit-hybrid-rc-s{seed}",
            seed=seed,
            extra_flags=["--ratio-conditioned"],
        )
        for seed in SEEDS
    ]


def build_sweep_experiment() -> SweepExperiment:
    base_name = f"bandit-hybrid-s{SWEEP_SEED}"
    checkpoint = str(Path(OUTPUT_DIR) / base_name / "checkpoints" / "best.pt")
    return SweepExperiment(
        checkpoint=checkpoint,
        ratios=SWEEP_RATIOS,
        experiment_name=f"{base_name}-sweep",
    )


def run_main(dry_run: bool) -> None:
    _print_header("Group 1: Main Training (3 seeds)")
    for exp in build_main_experiments():
        _run_command(_train_args(exp), dry_run)


def run_ablation(dry_run: bool) -> None:
    _print_header("Group 2: Ratio Conditioning Ablation (3 seeds)")
    for exp in build_ablation_experiments():
        _run_command(_train_args(exp), dry_run)


def run_sweep(dry_run: bool) -> None:
    _print_header("Group 3: Compression Ratio Sweep (seed 42)")
    exp = build_sweep_experiment()
    if not dry_run and not Path(exp.checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {exp.checkpoint}. Run --group main first."
        )
    _run_command(_sweep_args(exp), dry_run)


GROUP_RUNNERS = {
    "main": run_main,
    "ablation": run_ablation,
    "sweep": run_sweep,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 experiments")
    parser.add_argument(
        "--group",
        choices=["main", "ablation", "sweep", "all"],
        default="all",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.group == "all":
        for runner in GROUP_RUNNERS.values():
            runner(args.dry_run)
    else:
        GROUP_RUNNERS[args.group](args.dry_run)


if __name__ == "__main__":
    main()
