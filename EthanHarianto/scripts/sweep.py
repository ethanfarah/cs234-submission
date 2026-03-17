"""Prioritized W&B sweep launcher for the 3-axis experiment grid.

Runs configs in a budget-aware order: fastest reward types first,
most-promising algo/policy combos first within each tier.  Each tier
can have a different episode count so hybrid (2 LLM passes/ep) doesn't
burn the entire compute budget.

Each run executes in a **subprocess** so a CUDA crash in one config
doesn't poison the GPU state for subsequent runs.

Bandit + sparse is excluded (bandit requires dense per-token rewards).
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import wandb

from src.config import AlgoType, PolicyArch, RewardType, load_config
from src.train import train

_SWEEP_PROJECT = "prompt-compression-sweep"
_SWEEP_GROUP = "round6-prioritized"

_ARCH_PRETRAINED = {
    PolicyArch.DISTILBERT: "distilbert-base-uncased",
    PolicyArch.TINYLLAMA: "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

_ALGO_CONFIGS = {
    "bandit": "configs/algo/bandit.yaml",
    "ppo": "configs/algo/ppo.yaml",
    "reinforce": "configs/algo/reinforce.yaml",
    "dqn": "configs/algo/dqn.yaml",
}

_REWARD_CONFIGS = {
    "hybrid": "configs/reward/hybrid.yaml",
    "kl_dense": "configs/reward/kl_dense.yaml",
    "sparse": "configs/reward/sparse.yaml",
}

_ALGO_PRIORITY = ["bandit", "ppo", "reinforce", "dqn"]
_POLICY_PRIORITY = ["custom_transformer", "distilbert", "tinyllama"]


@dataclass
class RunSpec:
    algo: str
    policy: str
    reward: str
    episodes: int
    tier: int
    seed: int


def build_run_queue(seeds: list[int]) -> list[RunSpec]:
    """Build prioritized run queue across 4 tiers (~12 day budget).

    Tier 1: KL-dense (all 12) at 10K — ~0.5-1.5 sec/ep, ~28h total
    Tier 2: Hybrid (top 4: {bandit,ppo} x {CT,DB}) at 3K — ~16 sec/ep, ~53h
    Tier 3: Sparse (valid 9: no bandit) at 3K — ~10 sec/ep, ~50h
    Tier 4: Hybrid (remaining 8) at 2K — ~16-25 sec/ep, ~80h
    Budget: ~211h ≈ 8.8 days (+ model load overhead ~20h) ≈ 12 days
    """
    queue: list[RunSpec] = []

    for algo in _ALGO_PRIORITY:
        for policy in _POLICY_PRIORITY:
            for seed in seeds:
                queue.append(RunSpec(algo, policy, "kl_dense", 10_000, 1, seed))

    tier2_algos = ["bandit", "ppo"]
    tier2_policies = ["custom_transformer", "distilbert"]
    for algo in tier2_algos:
        for policy in tier2_policies:
            for seed in seeds:
                queue.append(RunSpec(algo, policy, "hybrid", 3_000, 2, seed))

    sparse_algos = ["ppo", "reinforce", "dqn"]
    for algo in sparse_algos:
        for policy in _POLICY_PRIORITY:
            for seed in seeds:
                queue.append(RunSpec(algo, policy, "sparse", 3_000, 3, seed))

    tier2_set = {(a, p) for a in tier2_algos for p in tier2_policies}
    for algo in _ALGO_PRIORITY:
        for policy in _POLICY_PRIORITY:
            if (algo, policy) not in tier2_set:
                for seed in seeds:
                    queue.append(RunSpec(algo, policy, "hybrid", 2_000, 4, seed))

    return queue


def _run_single(spec: RunSpec) -> None:
    """Execute a single training run with W&B tracking (in-process)."""
    run_name = f"{spec.algo}_{spec.policy}_{spec.reward}_seed{spec.seed}"

    configs_to_load = ["configs/base.yaml"]
    for cfg_map, key in [(_ALGO_CONFIGS, spec.algo), (_REWARD_CONFIGS, spec.reward)]:
        yaml_path = cfg_map[key]
        if not Path(yaml_path).exists():
            raise FileNotFoundError(f"Config not found: {yaml_path}")
        configs_to_load.append(yaml_path)

    config = load_config(configs_to_load)
    config.algo.algo_type = AlgoType(spec.algo)
    config.policy.arch = PolicyArch(spec.policy)
    pretrained = _ARCH_PRETRAINED.get(config.policy.arch)
    if pretrained is not None:
        config.policy.pretrained_name = pretrained
    config.reward.reward_type = RewardType(spec.reward)
    config.train.num_episodes = spec.episodes
    config.train.early_stop_window = 500  # ~5% of max run; catches collapse early
    config.train.seed = spec.seed
    config.train.wandb_project = None
    config.train.output_dir = f"outputs/{run_name}"
    config.experiment_name = run_name

    wandb.init(
        project=_SWEEP_PROJECT,
        group=_SWEEP_GROUP,
        name=run_name,
        config={
            "algo_type": spec.algo,
            "policy_arch": spec.policy,
            "reward_type": spec.reward,
            "num_episodes": spec.episodes,
            "tier": spec.tier,
            "seed": spec.seed,
        },
        tags=[f"tier{spec.tier}", spec.algo, spec.policy, spec.reward, f"seed{spec.seed}"],
    )
    try:
        train(config)
    finally:
        wandb.finish()


def _run_as_subprocess(spec: RunSpec, run_idx: int, total: int, seeds: list[int]) -> bool:
    """Spawn a subprocess for one run. Returns True on success."""
    run_name = f"{spec.algo}_{spec.policy}_{spec.reward}_seed{spec.seed}"
    print(
        f"\n{'='*60}\n"
        f"[{run_idx}/{total}] tier={spec.tier} | {run_name} | "
        f"{spec.episodes} eps\n"
        f"{'='*60}",
        file=sys.stderr,
    )
    result = subprocess.run(
        [sys.executable, "-m", "scripts.sweep", "--single", str(run_idx - 1), "--seeds", *[str(s) for s in seeds]],
        env={"PYTHONUNBUFFERED": "1", **__import__("os").environ},
    )
    if result.returncode != 0:
        print(f"\nRUN FAILED: {run_name} (exit code {result.returncode})", file=sys.stderr)
        return False
    return True


def main() -> None:
    """Launch prioritized sweep with subprocess isolation.

    Usage:
      python -m scripts.sweep [--skip N]     # orchestrator mode
      python -m scripts.sweep --single IDX   # worker mode (internal)
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", type=int, default=0, help="Skip first N runs")
    parser.add_argument("--single", type=int, default=-1, help="Run single config by index (worker mode)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Seeds to run for each config")
    args = parser.parse_args()

    queue = build_run_queue(args.seeds)
    total = len(queue)

    # Worker mode: run one config and exit
    if args.single >= 0:
        _run_single(queue[args.single])
        return

    # Orchestrator mode: spawn subprocesses
    print(f"Prioritized sweep: {total} runs across 4 tiers", file=sys.stderr)
    for tier in range(1, 5):
        tier_runs = [r for r in queue if r.tier == tier]
        tier_eps = sum(r.episodes for r in tier_runs)
        print(f"  Tier {tier}: {len(tier_runs)} runs, {tier_eps:,} total episodes", file=sys.stderr)

    failed: list[str] = []
    for i, spec in enumerate(queue, 1):
        if i <= args.skip:
            name = f"{spec.algo}_{spec.policy}_{spec.reward}"
            print(f"[{i}/{total}] SKIPPED {name}", file=sys.stderr)
            continue
        ok = _run_as_subprocess(spec, i, total, args.seeds)
        if not ok:
            failed.append(f"{spec.algo}_{spec.policy}_{spec.reward}_seed{spec.seed}")

    succeeded = total - args.skip - len(failed)
    print(f"\nSweep complete: {succeeded}/{total - args.skip} succeeded", file=sys.stderr)
    if failed:
        print(f"Failed runs: {failed}", file=sys.stderr)


if __name__ == "__main__":
    main()
