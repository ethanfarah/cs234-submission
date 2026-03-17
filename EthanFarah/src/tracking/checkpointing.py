"""Checkpoint save/load for training state."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.algo.base import Algorithm
    from src.policy.base import Policy


@dataclass
class CheckpointState:
    """Bundle of objects needed for checkpoint save/load."""

    policy: Policy
    algo: Algorithm
    step: int


def save_checkpoint(state: CheckpointState, path: Path) -> None:
    """Save policy, algorithm, and step to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": state.policy.state_dict(),
            "algo": state.algo.state_dict(),
            "step": state.step,
        },
        path,
    )


def load_checkpoint(state: CheckpointState, path: Path) -> int:
    """Load checkpoint from disk into state objects. Returns the saved step."""
    # weights_only=False to support custom objects in algo state_dict.
    # Only load checkpoints from trusted sources.
    ckpt = torch.load(path, weights_only=False)
    state.policy.load_state_dict(ckpt["policy"], strict=False)
    state.algo.load_state_dict(ckpt["algo"])
    return int(ckpt["step"])
