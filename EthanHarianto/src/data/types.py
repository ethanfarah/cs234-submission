"""Core data types shared across the codebase."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Prompt:
    """A full prompt before compression."""

    token_ids: torch.Tensor          # (seq_len,)
    attention_mask: torch.Tensor     # (seq_len,)
    text: str
    metadata: dict = field(default_factory=dict)
    # metadata holds task-specific info:
    #   SQuAD: {"answer_texts": [...], "answer_starts": [...], "is_answerable": bool}
    #   MeetingBank: {"reference_summary": "..."}


@dataclass
class CompressedPrompt:
    """A prompt after token-level compression."""

    token_ids: torch.Tensor          # (compressed_len,)
    keep_mask: torch.Tensor          # (original_seq_len,) binary mask
    compression_ratio: float         # compressed_len / original_len


@dataclass
class RewardInput:
    """Inputs for reward computation.

    Sparse rewards use llm_output. Dense rewards use the KL cache directly.
    """

    original: Prompt
    compressed: CompressedPrompt
    llm_output: str | None = None


@dataclass
class Episode:
    """A full episode of compression decisions + outcomes."""

    prompt: Prompt
    actions: torch.Tensor            # (seq_len,) enforced binary keep/drop
    log_probs: torch.Tensor          # (seq_len,) log prob of each action
    rewards: torch.Tensor            # (seq_len,) per-step rewards (dense) or zeros + terminal
    # None for REINFORCE (no value baseline); populated with critic estimates for PPO/actor-critic
    values: torch.Tensor | None      # (seq_len,) value estimates
    compressed: CompressedPrompt
    terminal_reward: float
    sampled_actions: torch.Tensor | None = None  # (seq_len,) raw policy-sampled actions before env enforcement
    chunk_boundaries: list[int] = field(default_factory=list)
    chunk_size: int = 0
    overlap: int = 0
    target_compression_ratio: float = 0.0
    info: dict[str, float] = field(default_factory=dict)
    forced_keep_mask: torch.Tensor | None = None  # (seq_len,) bool mask where env flipped drop->keep