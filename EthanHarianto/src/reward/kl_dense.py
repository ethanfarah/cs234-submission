"""KL-divergence dense per-token reward."""

from __future__ import annotations

import torch
from torch import Tensor

from src.config import RewardConfig
from src.data.types import RewardInput
from src.llm.kl_cache import KLCache
from src.reward.base import RewardFunction


class KLDenseReward(RewardFunction):
    """Dense per-token reward: negative KL divergence scaled by kl_coeff.

    When compression_bonus > 0, a terminal scalar bonus proportional to
    compression (1 - ratio) is added, giving the policy an incentive to
    actually compress rather than keeping all tokens (reward=0).
    """

    def __init__(self, config: RewardConfig, kl_cache: KLCache) -> None:
        self.config = config
        self.kl_cache = kl_cache
        self._last_ratio: float | None = None

    def compute(self, reward_input: RewardInput) -> Tensor:
        compressed = reward_input.compressed
        self._last_ratio = compressed.compression_ratio
        kl = self.kl_cache.compute_kl(
            compressed.token_ids.unsqueeze(0),
            compressed.keep_mask.bool(),
            direction=self.config.kl_direction,
        )
        return -self.config.kl_coeff * kl

    def terminal_scalar(self) -> Tensor | None:
        """Compression bonus proportional to (1 - ratio).

        Must be called after compute() which sets _last_ratio.
        Returns None if compression_bonus is disabled (0.0).
        """
        if self.config.compression_bonus == 0.0:
            return None
        if self._last_ratio is None:
            raise RuntimeError(
                "terminal_scalar() called before compute(); "
                "_last_ratio is not set"
            )
        bonus = self.config.compression_bonus * (1.0 - self._last_ratio)
        return torch.tensor(bonus)

    def is_dense(self) -> bool:
        return True
