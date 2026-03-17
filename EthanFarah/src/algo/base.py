"""Abstract algorithm interface for RL training."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from src.data.types import Episode

if TYPE_CHECKING:
    from src.data.types import Prompt
    from src.env.compression_env import CompressionEnv


class Algorithm(ABC):
    """Base class for all RL algorithms.

    An algorithm takes collected episodes and performs one or more
    gradient updates on the policy (and optionally a value function).
    """

    @abstractmethod
    def update(self, episodes: list[Episode]) -> dict[str, float]:
        """Perform a training update given collected episodes.

        Args:
            episodes: List of completed episodes with actions, rewards, etc.

        Returns:
            Dictionary of training metrics (loss, entropy, etc.) for logging.
        """
        ...

    def collect_episode(
        self, env: CompressionEnv, prompt: Prompt,
    ) -> Episode | None:
        """Override for algorithm-specific episode collection.

        Returns None to use the default collect_episode path.
        """
        return None

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return serializable state (optimizer, buffers, etc.) for checkpointing."""
        ...

    @abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state from a previously saved state_dict."""
        ...
