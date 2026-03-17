"""Streaming chunk logic for no-look-ahead processing."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from src.data.types import Prompt
from src.env.spaces import Observation


@dataclass
class ChunkConfig:
    """Configuration for overlapping chunk windows."""

    chunk_size: int = 128
    overlap: int = 16

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {self.chunk_size}")
        if not (0 <= self.overlap < self.chunk_size):
            raise ValueError(
                f"overlap must be in [0, chunk_size), got {self.overlap}"
            )


def compute_chunks(
    prompt: Prompt,
    config: ChunkConfig,
    target_ratio: float = 0.0,
) -> list[Observation]:
    """Split a prompt into overlapping chunks as Observations.

    Chunks advance by (chunk_size - overlap) tokens with the last chunk
    potentially shorter than chunk_size.

    Args:
        prompt: The full prompt to chunk.
        config: Chunk size and overlap settings.
        target_ratio: Target compression ratio set on each Observation.

    Returns:
        List of Observation objects, one per chunk.
    """
    seq_len = prompt.token_ids.shape[0]
    if seq_len == 0:
        return []

    stride = config.chunk_size - config.overlap
    starts = list(range(0, seq_len, stride))

    # Drop trailing start whose chunk would contribute nothing after
    # overlap skipping (e.g. seq_len=113, stride=112 → tail chunk of 1
    # token is fully inside the previous chunk's overlap zone).
    while len(starts) > 1:
        tail_len = min(starts[-1] + config.chunk_size, seq_len) - starts[-1]
        if tail_len <= config.overlap:
            starts.pop()
        else:
            break

    total_chunks = len(starts)
    chunks = []
    for i, start in enumerate(starts):
        end = min(start + config.chunk_size, seq_len)
        chunks.append(
            Observation(
                token_ids=prompt.token_ids[start:end],
                attention_mask=prompt.attention_mask[start:end],
                position_ids=torch.arange(start, end),
                compression_ratio_so_far=0.0,   # env overwrites at serve time
                target_compression_ratio=target_ratio,
                chunk_index=i,
                total_chunks=total_chunks,
            )
        )
    return chunks


def merge_chunk_actions(chunk_actions: list[Tensor], config: ChunkConfig) -> Tensor:
    """Merge per-chunk action tensors back to a full-prompt action tensor.

    Overlap regions are deduplicated by keeping the action from the
    earlier chunk ("earlier chunk wins" — committed decisions are final).
    Each chunk after the first skips its leading `overlap` tokens, which
    were already committed by the previous chunk. The first chunk
    contributes all its tokens. Result: exactly seq_len elements.

    Args:
        chunk_actions: List of (chunk_len,) binary action tensors.
        config: Chunk size and overlap settings.

    Returns:
        A single (seq_len,) binary action tensor for the full prompt.
    """
    if not chunk_actions:
        return torch.tensor([], dtype=torch.long)

    parts = []
    for i, actions in enumerate(chunk_actions):
        skip = config.overlap if i > 0 else 0
        parts.append(actions[skip:])
    return torch.cat(parts)
