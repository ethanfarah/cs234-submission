"""Core data types shared across the codebase."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class Prompt:
    """A full prompt before compression."""

    token_ids: torch.Tensor          # (seq_len,) — policy tokenizer space
    attention_mask: torch.Tensor     # (seq_len,) — policy tokenizer space
    text: str
    metadata: dict = field(default_factory=dict)
    # metadata holds task-specific info:
    #   SQuAD: {"answer_texts": [...], "answer_starts": [...], "is_answerable": bool}

    # Dual-tokenizer fields (populated when policy_tokenizer != llm_tokenizer)
    llm_token_ids: torch.Tensor | None = None       # GPT-2 token_ids
    llm_attention_mask: torch.Tensor | None = None   # GPT-2 attention_mask
    alignment: Any | None = None                      # used by KL computation in train.py


@dataclass
class CompressedPrompt:
    """A prompt after token-level compression."""

    token_ids: torch.Tensor          # (compressed_len,) — only kept tokens in LLM space
    keep_mask: torch.Tensor          # (policy_seq_len,) binary mask in policy space
    compression_ratio: float         # fraction of tokens kept


@dataclass
class RewardInput:
    """Inputs for reward computation, supporting both sparse and dense rewards.

    Sparse rewards (e.g., task accuracy) use llm_output.
    Dense rewards (e.g., KL-based) use kl_cache directly.
    """

    original: Prompt
    compressed: CompressedPrompt
    llm_output: str | None = None
    original_llm_output: str | None = None         # LLM output on uncompressed prompt (for faithfulness)


@dataclass
class Episode:
    """A full episode of compression decisions + outcomes."""

    prompt: Prompt
    actions: torch.Tensor            # (seq_len,) binary keep/drop
    log_probs: torch.Tensor          # (seq_len,) log prob of each action
    rewards: torch.Tensor            # (seq_len,) per-step rewards (dense) or zeros + terminal
    compressed: CompressedPrompt
    terminal_reward: float
    chunk_boundaries: list[int] = field(default_factory=list)
    chunk_size: int = 0
    overlap: int = 0
    target_compression_ratio: float = 0.0
    baseline_reward: float = 0.0
