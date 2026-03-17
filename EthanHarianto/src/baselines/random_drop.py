"""Random token dropping baseline."""

from __future__ import annotations

import torch

from src.data.types import CompressedPrompt, Prompt


def random_drop(
    prompt: Prompt,
    keep_ratio: float,
    seed: int | None = None,
) -> CompressedPrompt:
    """Randomly keep tokens at given ratio.

    Args:
        prompt: Full prompt to compress.
        keep_ratio: Fraction of tokens to keep in [0, 1].
        seed: Optional random seed for reproducibility.

    Returns:
        CompressedPrompt with randomly selected tokens.
    """
    if not (0.0 <= keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in [0, 1], got {keep_ratio}")

    n = prompt.token_ids.shape[0]
    gen = torch.Generator()
    if seed is not None:
        gen.manual_seed(seed)

    keep_mask = torch.bernoulli(torch.full((n,), keep_ratio), generator=gen).long()
    kept_ids = prompt.token_ids[keep_mask.bool()]
    ratio = kept_ids.shape[0] / max(n, 1)
    return CompressedPrompt(token_ids=kept_ids, keep_mask=keep_mask, compression_ratio=ratio)
