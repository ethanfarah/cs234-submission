"""Sparse terminal reward based on F1/ROUGE with configurable reward modes."""

from __future__ import annotations

import torch
from torch import Tensor

from src.config import RewardConfig, SparseRewardMode
from src.data.types import RewardInput
from src.reward.base import RewardFunction
from src.reward.metrics import compute_f1, compute_rouge


def _compute_task_score(llm_output: str, metadata: dict) -> float:
    """Extract task score from LLM output: F1 for QA, ROUGE-L for summarization."""
    if "answer_texts" in metadata:
        if not metadata["answer_texts"]:
            return 0.0
        return max(compute_f1(llm_output, ans) for ans in metadata["answer_texts"])
    if "reference_summary" in metadata:
        return compute_rouge(llm_output, metadata["reference_summary"])["rougeL"]
    raise ValueError(f"Unknown task: metadata keys = {list(metadata.keys())}")


def _reward_multiplicative(task_score: float, ratio: float) -> float:
    """task_score * (1 - ratio). Zero reward at ratio=1.0."""
    return task_score * (1.0 - ratio)


def _reward_threshold(task_score: float, ratio: float, config: RewardConfig) -> float:
    """PCRL-style: (1-ratio) if quality >= threshold, else -penalty.

    Quality above the threshold is saturated — no additional signal beyond the gate.
    """
    if task_score >= config.quality_threshold:
        return 1.0 - ratio
    return -config.failure_penalty


def _reward_additive(task_score: float, ratio: float, config: RewardConfig) -> float:
    """Original: task_score - penalty * |ratio - target|."""
    penalty = config.compression_penalty * abs(ratio - config.target_compression_ratio)
    return task_score - penalty


class SparseReward(RewardFunction):
    """Sparse terminal reward dispatching on SparseRewardMode."""

    def __init__(self, config: RewardConfig) -> None:
        self.config = config

    def compute(self, reward_input: RewardInput) -> Tensor:
        """Compute sparse reward based on configured mode."""
        if reward_input.llm_output is None:
            raise ValueError("SparseReward requires llm_output; got None")

        ratio = reward_input.compressed.compression_ratio
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"compression_ratio must be in [0, 1], got {ratio}")

        task_score = _compute_task_score(
            reward_input.llm_output, reward_input.original.metadata
        )
        mode = self.config.sparse_reward_mode

        if mode == SparseRewardMode.MULTIPLICATIVE:
            reward = _reward_multiplicative(task_score, ratio)
        elif mode == SparseRewardMode.THRESHOLD:
            reward = _reward_threshold(task_score, ratio, self.config)
        elif mode == SparseRewardMode.ADDITIVE:
            reward = _reward_additive(task_score, ratio, self.config)
        else:
            raise ValueError(f"Unknown sparse_reward_mode: {mode}")

        return torch.tensor(reward, dtype=torch.float32)

    def is_dense(self) -> bool:
        return False
