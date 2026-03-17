"""Sparse terminal reward based on F1/ROUGE with configurable reward modes."""

from __future__ import annotations

import re

import torch
from torch import Tensor

from src.config import RewardConfig, SparseRewardMode
from src.data.types import RewardInput
from src.reward.base import RewardFunction
from src.reward.metrics import compute_f1, compute_rouge, normalize_answer


_NO_ANSWER_PATTERNS = (
    "no answer",
    "cannot answer",
    "can't answer",
    "not answerable",
    "unanswerable",
    "insufficient information",
    "unknown",
)


def _answer_candidates(llm_output: str) -> list[str]:
    """Build short-answer candidates from free-form generation."""
    text = llm_output.strip()
    if not text:
        return [""]
    candidates: list[str] = [text]

    first_line = text.splitlines()[0].strip()
    if first_line:
        candidates.append(first_line)

    # Common answer prefixes used by instruction-following LMs.
    for match in re.finditer(r"(?i)\b(answer|final answer)\s*:\s*(.+)", text):
        tail = match.group(2).strip()
        if tail:
            candidates.append(tail.splitlines()[0].strip())

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique


def _is_no_answer_prediction(llm_output: str) -> bool:
    normalized = normalize_answer(llm_output)
    return any(pat in normalized for pat in _NO_ANSWER_PATTERNS)


def _compute_task_score(llm_output: str, metadata: dict) -> float:
    """Extract task score from LLM output: F1 for QA, ROUGE-L for summarization."""
    if "answer_texts" in metadata:
        answer_texts = metadata["answer_texts"]
        if not answer_texts:
            # SQuAD v2 unanswerable: reward explicit abstention behavior.
            return 1.0 if _is_no_answer_prediction(llm_output) else 0.0
        return max(
            compute_f1(candidate, ans)
            for candidate in _answer_candidates(llm_output)
            for ans in answer_texts
        )
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
    """Scaled task_score - penalty * |ratio - target| + bias. Use reward_bias so reported reward stays positive."""
    scale = getattr(config, "task_score_scale", 1.0)
    penalty = config.compression_penalty * abs(ratio - config.target_compression_ratio)
    bias = getattr(config, "reward_bias", 0.0)
    return scale * task_score - penalty + bias


class SparseReward(RewardFunction):
    """Sparse terminal reward dispatching on SparseRewardMode."""

    def __init__(self, config: RewardConfig) -> None:
        self.config = config
        self.current_quality_threshold: float | None = None
        self.last_task_score: float | None = None
        self.last_threshold_pass: bool | None = None
        self.last_effective_threshold: float | None = None

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
        self.last_task_score = task_score
        mode = self.config.sparse_reward_mode

        if mode == SparseRewardMode.MULTIPLICATIVE:
            reward = _reward_multiplicative(task_score, ratio)
            self.last_threshold_pass = None
            self.last_effective_threshold = None
        elif mode == SparseRewardMode.THRESHOLD:
            effective_threshold = (
                self.current_quality_threshold
                if self.current_quality_threshold is not None
                else self.config.quality_threshold
            )
            effective_cfg = RewardConfig(**vars(self.config))
            effective_cfg.quality_threshold = effective_threshold
            reward = _reward_threshold(task_score, ratio, effective_cfg)
            self.last_threshold_pass = task_score >= effective_threshold
            self.last_effective_threshold = effective_threshold
        elif mode == SparseRewardMode.ADDITIVE:
            reward = _reward_additive(task_score, ratio, self.config)
            self.last_threshold_pass = None
            self.last_effective_threshold = None
        else:
            raise ValueError(f"Unknown sparse_reward_mode: {mode}")

        return torch.tensor(reward, dtype=torch.float32)

    def is_dense(self) -> bool:
        return False
