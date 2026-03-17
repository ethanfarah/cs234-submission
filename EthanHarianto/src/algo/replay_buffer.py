"""Experience replay buffer for DQN."""

from __future__ import annotations

from typing import Any

import random
from dataclasses import dataclass

from torch import Tensor

from src.env.spaces import Observation


@dataclass
class Transition:
    """A single (s, a, r, s', done) transition."""

    observation: Observation
    action: Tensor
    reward: Tensor
    next_observation: Observation
    done: bool


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer with O(1) random access.

    Uses a list-based circular buffer instead of deque so that indexed
    access for sampling is truly O(1).
    """

    def __init__(self, capacity: int) -> None:
        self._buffer: list[Transition] = []
        self._capacity = capacity
        self._pos = 0

    def push(self, transition: Transition) -> None:
        """Add a transition, overwriting the oldest if at capacity."""
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._pos] = transition
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample a random batch of transitions (O(k) with O(1) access)."""
        if batch_size > len(self):
            raise ValueError(
                f"batch_size ({batch_size}) > buffer length ({len(self)})"
            )
        return random.sample(self._buffer, batch_size)

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)

    def state_dict(self) -> dict[str, Any]:
        """Serialize buffer contents for checkpointing."""
        return {
            "transitions": list(self._buffer),
            "pos": self._pos,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore buffer contents from checkpoint."""
        self._buffer = list(state["transitions"])
        self._pos = state.get("pos", len(self._buffer) % self._capacity)
