"""Lightweight custom Transformer policy trained from scratch."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.config import PolicyConfig
from src.env.spaces import Observation
from src.policy.base import Policy
from src.policy.ratio_conditioning import RatioConditioner

_VOCAB_SIZE = 128_256  # Llama-3.1-8B tiktoken vocab (Llama 3/3.1 exact); NOT 32k (that's Llama 1/2)
_MAX_SEQ_LEN = 1024    # matches DataConfig.max_prompt_tokens


class CustomTransformerPolicy(Policy):
    """Custom Transformer policy built from scratch.

    Supports both causal and bidirectional modes via config.causal.
    Uses config: hidden_dim, num_layers, num_heads, dropout.
    Optionally uses RatioConditioner based on config.ratio_conditioned.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(_VOCAB_SIZE, config.hidden_dim)
        nn.init.normal_(self.token_embedding.weight, std=0.02)  # GPT-2/Llama standard embedding init
        # Zero-init is intentional: the model learns positional signal from gradients.
        self.pos_encoding = nn.Parameter(torch.zeros(_MAX_SEQ_LEN, config.hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_dim),
        )
        self.head_norm = nn.LayerNorm(config.hidden_dim)
        self.head = nn.Linear(config.hidden_dim, 2)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.ratio_conditioner = (
            RatioConditioner(config.hidden_dim) if config.ratio_conditioned else None
        )

    def forward(self, obs: Observation) -> Tensor:
        """Return per-token keep/drop logits.

        Expects an unbatched observation (obs.token_ids is 1-D). Batching is
        handled at the algorithm level via evaluate_actions, which is always
        called with batch=1 observations from collect_episode.

        obs.attention_mask is intentionally unused: ChunkConfig never produces
        padding, so all positions are always valid.

        Args:
            obs: Observation with token_ids and target_compression_ratio.

        Returns:
            Tensor of shape (1, seq_len, 2) -- logits for [drop, keep].
        """
        if obs.token_ids.dim() != 1:
            raise ValueError(f"expected unbatched obs (token_ids must be 1-D), got {obs.token_ids.dim()}-D")
        token_ids = obs.token_ids.unsqueeze(0)   # (1, seq_len)
        seq_len = token_ids.shape[1]
        x = self.token_embedding(token_ids)       # (1, seq_len, hidden_dim)
        x = x + self.pos_encoding[:seq_len]       # broadcast over batch

        if self.config.causal:
            mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
            x = self.transformer(x, mask=mask, is_causal=True)  # is_causal hints FlashAttention
        else:
            x = self.transformer(x)

        if self.ratio_conditioner is not None:
            x = self.ratio_conditioner(x, obs.target_compression_ratio)

        x = self.head_norm(x)
        return self.head(x)  # (1, seq_len, 2)
