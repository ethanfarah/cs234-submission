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

        return dense

    def terminal_scalar(self) -> Tensor | None:
        return self.last_sparse

    def _apply_threshold(self, kl_reward: Tensor) -> Tensor:
        raw_kl = -kl_reward / self.config.kl_coeff
        exceeds = raw_kl > self.config.threshold_tau
        penalty = torch.full_like(kl_reward, -self.config.threshold_penalty)
        return torch.where(exceeds, penalty, torch.zeros_like(kl_reward))

    def is_dense(self) -> bool:
        return True
