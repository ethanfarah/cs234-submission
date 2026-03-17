"""Ratio-conditioning adapter mixin for compression policies."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class RatioConditioner(nn.Module):
    """Maps a scalar target_ratio to a hidden_dim vector and fuses it with token features.

    Uses a small 2-layer MLP with GELU: scalar -> hidden_dim.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, features: Tensor, target_ratio: float) -> Tensor:
        """Fuse ratio embedding with token features.

        Designed for batch=1 (single observation per forward call), consistent
        with CustomTransformerPolicy. A scalar target_ratio is used rather than
        a per-sample tensor because the environment produces one ratio per episode.

        Args:
            features: Token features of shape (1, seq_len, hidden_dim).
            target_ratio: Compression target scalar in [0, 1].

        Returns:
            Fused features of shape (1, seq_len, hidden_dim).
        """
        ratio = torch.tensor([[target_ratio]], dtype=features.dtype, device=features.device)
        embedding = self.mlp(ratio)               # (1, hidden_dim)
        return features + embedding.unsqueeze(1)  # broadcast: (1, 1, hidden_dim) -> (1, seq_len, hidden_dim)
