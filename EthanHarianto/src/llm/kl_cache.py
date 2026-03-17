"""KL computation with full-prompt logit caching."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.config import KLDirection
from src.data.types import Prompt
from src.llm.frozen_llm import FrozenLLM


class KLCache:
    """Caches full-prompt logits and computes KL divergence against compressed prompts.

    Call cache_full_prompt once per episode, then compute_kl per chunk.
    Clear between episodes.
    """

    def __init__(self, llm: FrozenLLM) -> None:
        self.llm = llm
        self._cached_logits: Tensor | None = None

    def cache_full_prompt(self, prompt: Prompt) -> None:
        ids = prompt.token_ids.unsqueeze(0)
        self._cached_logits = self.llm.get_logits(ids).squeeze(0)

    def compute_kl(
        self,
        compressed_ids: Tensor,
        keep_mask: Tensor,
        direction: KLDirection = KLDirection.FORWARD,
    ) -> Tensor:
        """Compute per-token KL divergence over kept positions.

        Args:
            compressed_ids: shape (1, compressed_len) — batch size must be 1.
            keep_mask: shape (seq_len,) bool — must match cached sequence length.
            direction: FORWARD = KL(full || compressed), REVERSE = KL(compressed || full).

        Returns:
            Per-token KL divergence, shape (compressed_len,).
        """
        if self._cached_logits is None:
            raise RuntimeError(
                "No cached logits — call cache_full_prompt before compute_kl"
            )
        if compressed_ids.dim() != 2 or compressed_ids.shape[0] != 1:
            raise ValueError(
                f"compressed_ids must be 2-D with batch size 1, "
                f"got shape {compressed_ids.shape}"
            )
        if keep_mask.dtype != torch.bool:
            raise ValueError(
                f"keep_mask must be a boolean tensor, got dtype {keep_mask.dtype}"
            )
        if keep_mask.dim() != 1:
            raise ValueError(
                f"keep_mask must be 1-D, got shape {keep_mask.shape}"
            )
        if keep_mask.shape[0] != self._cached_logits.shape[0]:
            raise ValueError(
                f"keep_mask length {keep_mask.shape[0]} does not match "
                f"cached sequence length {self._cached_logits.shape[0]}"
            )

        comp_logits = self.llm.get_logits(compressed_ids).squeeze(0)
        full_logits = self._cached_logits[keep_mask]

        if full_logits.shape[0] != comp_logits.shape[0]:
            raise ValueError(
                f"Mask selected {full_logits.shape[0]} positions but "
                f"compressed_ids has {comp_logits.shape[0]} tokens"
            )

        full_log_p = F.log_softmax(full_logits, dim=-1)
        comp_log_p = F.log_softmax(comp_logits, dim=-1)
        if direction == KLDirection.FORWARD:
            kl = F.kl_div(comp_log_p, full_log_p, reduction="none", log_target=True)
        elif direction == KLDirection.REVERSE:
            kl = F.kl_div(full_log_p, comp_log_p, reduction="none", log_target=True)
        else:
            raise ValueError(f"Unknown KLDirection: {direction}")
        return kl.sum(dim=-1)

    def clear(self) -> None:
        """Clear cached logits between episodes."""
        self._cached_logits = None
