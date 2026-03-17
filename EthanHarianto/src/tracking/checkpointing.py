"""Checkpoint save/load for training state and reproducibility metadata."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from src.algo.base import Algorithm


@dataclass
class CheckpointState:
    """Bundle of objects needed for checkpoint save/load.

    Policy state is stored inside algo.state_dict() — no separate field needed.
    """

    algo: Algorithm
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)


def _collect_rng_state() -> dict[str, Any]:
    """Capture RNG state for reproducible resume."""
    return {
        "python_random": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore RNG state if available in checkpoint."""
    if not state:
        return
    py_state = state.get("python_random")
    if py_state is not None:
        random.setstate(py_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def save_checkpoint(state: CheckpointState, path: Path) -> None:
    """Save algorithm state, step, metadata, and RNG state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algo": state.algo.state_dict(),
            "step": state.step,
            "metadata": state.metadata,
            "rng_state": _collect_rng_state(),
        },
        path,
    )


def load_checkpoint(state: CheckpointState, path: Path) -> int:
    """Load checkpoint from disk into state objects. Returns the saved step.

    Policy state is restored via algo.load_state_dict(), which internally
    calls policy.load_state_dict().
    """
    # weights_only=False to support custom objects in algo state_dict
    # (e.g. LR schedulers, replay buffers). Only load checkpoints from trusted sources.
    ckpt = torch.load(path, weights_only=False)
    state.algo.load_state_dict(ckpt["algo"])
    state.metadata.clear()
    state.metadata.update(ckpt.get("metadata", {}))
    _restore_rng_state(ckpt.get("rng_state"))
    return int(ckpt["step"])
