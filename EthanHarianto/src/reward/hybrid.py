"""Hybrid reward: per-token KL signal + terminal task reward."""

from __future__ import annotations

import torch
from torch import Tensor

from src.config import HybridMode, RewardConfig
from src.data.types import RewardInput
from src.llm.kl_cache import KLCache
from src.reward.base import RewardFunction
from src.reward.kl_dense import KLDenseReward
from src.reward.sparse import SparseReward


class HybridReward(RewardFunction):
    """Weighted or threshold combination of dense KL and sparse terminal reward.

    Dense KL signal is returned by compute(); the sparse terminal flows
    exclusively through terminal_scalar() → episode.terminal_reward.
    """

    def __init__(self, config: RewardConfig, kl_cache: KLCache) -> None:
        self.config = config
        self._kl_reward = KLDenseReward(config, kl_cache)
        self._sparse_reward = SparseReward(config)
        self.last_sparse: Tensor | None = None

    def compute(self, reward_input: RewardInput) -> Tensor:
        self.last_sparse = None
        dense = self._kl_reward.compute(reward_input)
        self.last_sparse = self._sparse_reward.compute(reward_input)

        if self.config.hybrid_mode == HybridMode.THRESHOLD:
            dense = self._apply_threshold(dense)
        elif self.config.hybrid_mode != HybridMode.WEIGHTED:
            raise ValueError(f"Unknown HybridMode: {self.config.hybrid_mode}")

        return dense

    def terminal_scalar(self) -> Tensor | None:
        # Intentionally returns sparse component only — the inner
        # KLDenseReward.terminal_scalar() compression bonus is not
        # added because hybrid already gets compression incentive
        # from the sparse reward's ratio scaling.
        return self.last_sparse

    def _apply_threshold(self, kl_reward: Tensor) -> Tensor:
        """Replace continuous KL with binary penalty for exceeding threshold."""
        if self.config.kl_coeff == 0.0:
            raise ValueError("kl_coeff must be non-zero in threshold mode")
        raw_kl = -kl_reward / self.config.kl_coeff
        exceeds = raw_kl > self.config.threshold_tau
        penalty = torch.full_like(kl_reward, -self.config.threshold_penalty)
        return torch.where(exceeds, penalty, torch.zeros_like(kl_reward))

    def is_dense(self) -> bool:
        return True
