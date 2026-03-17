"""Sparse terminal reward: PCRL-style gated faithfulness reward."""

from __future__ import annotations

import torch
from torch import Tensor

from src.config import Config
from src.data.types import RewardInput
from src.reward.metrics import compute_f1, compute_rouge


class SparseReward:
    """Sparse terminal reward with four modes.

    continuous:      faithfulness * (1 - ratio)
    multiplicative:  (1 - ratio) if faithfulness >= threshold, else -penalty
    soft_gated:      sigmoid(10 * (faithfulness - threshold)) * (1 - ratio)
    harmonic:        2 * faithfulness * (1 - ratio) / (faithfulness + (1 - ratio))
    """

    _VALID_MODES = {"continuous", "multiplicative", "soft_gated", "harmonic"}

    def __init__(self, config: Config) -> None:
        if config.sparse_reward_mode not in self._VALID_MODES:
            raise ValueError(
                f"Unknown sparse_reward_mode: {config.sparse_reward_mode!r}. "
                f"Valid: {self._VALID_MODES}"
            )
        self.mode = config.sparse_reward_mode
        self.quality_threshold = config.quality_threshold
        self.failure_penalty = config.failure_penalty
        self.faithfulness_metric = config.faithfulness_metric

    def _compute_faithfulness(self, prediction: str, reference: str) -> float:
        if self.faithfulness_metric == "rougeL":
            return compute_rouge(prediction, reference)["rougeL"]
        return compute_f1(prediction, reference)

    def compute(self, reward_input: RewardInput) -> Tensor:
        if reward_input.llm_output is None:
            raise ValueError("SparseReward requires llm_output; got None")
        if reward_input.original_llm_output is None:
            raise ValueError("SparseReward requires original_llm_output; got None")

        ratio = reward_input.compressed.compression_ratio
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"compression_ratio must be in [0, 1], got {ratio}")

        faithfulness = self._compute_faithfulness(
            reward_input.llm_output, reward_input.original_llm_output,
        )
        if self.mode == "continuous":
            reward = faithfulness * (1.0 - ratio)
        elif self.mode == "multiplicative":
            if faithfulness >= self.quality_threshold:
                reward = (1.0 - ratio)
            else:
                reward = -self.failure_penalty
        elif self.mode == "soft_gated":
            gate = torch.sigmoid(torch.tensor(10.0 * (faithfulness - self.quality_threshold)))
            reward = gate.item() * (1.0 - ratio)
        elif self.mode == "harmonic":
            denom = faithfulness + (1.0 - ratio)
            reward = 2.0 * faithfulness * (1.0 - ratio) / denom if denom > 0 else 0.0
        else:
            raise ValueError(f"Unknown sparse_reward_mode: {self.mode}")
        return torch.tensor(reward, dtype=torch.float32)

    def is_dense(self) -> bool:
        return False
