"""Abstract reward function interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from src.data.types import RewardInput


class RewardFunction(ABC):
    """Base class for all reward functions.

    A reward function scores the quality of a compression decision,
    either densely (per-chunk/per-token) or sparsely (terminal only).
    """

    @abstractmethod
    def compute(self, reward_input: RewardInput) -> torch.Tensor:
        """Compute reward for a compression.

        Args:
            reward_input: Contains original prompt, compressed prompt, and
                optional llm_output (for sparse rewards) or logits tensors
                (for dense KL-based rewards).

        Returns:
            Scalar reward tensor (for sparse) or per-token tensor (for dense).
        """
        ...

    @abstractmethod
    def is_dense(self) -> bool:
        """Whether this reward function provides per-step dense rewards."""
        ...

    def terminal_scalar(self) -> torch.Tensor | None:
        """Return a scalar terminal reward if this reward tracks one separately.

        Returns None for reward functions that do not produce a separate terminal
        scalar (e.g., SparseReward). KLDenseReward overrides this to add a
        compression bonus; HybridReward overrides to expose its sparse component.
        """
        return None
