"""Observation dataclass for the compression environment."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Observation:
    """Observation passed to the policy at each chunk step.

    Tensors are unbatched (single observation). Batching is handled at the
    dataloader / algorithm level, not here.
    """

    token_ids: torch.Tensor           # (chunk_len,) current chunk tokens
    attention_mask: torch.Tensor      # (chunk_len,)
    position_ids: torch.Tensor        # (chunk_len,) absolute positions in full prompt
    compression_ratio_so_far: float   # tokens kept / tokens seen so far
    target_compression_ratio: float   # desired final compression ratio
    chunk_index: int                  # which chunk we're on (0-indexed)
    total_chunks: int                 # total number of chunks in this prompt
