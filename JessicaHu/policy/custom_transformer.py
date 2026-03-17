from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from src.config import PolicyConfig
from src.env.spaces import Observation
from src.policy.base import Policy
from src.policy.ratio_conditioning import RatioConditioner

_VOCAB_SIZE = 128_256
_MAX_SEQ_LEN = 1024  


class CustomTransformerPolicy(Policy):

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(_VOCAB_SIZE, config.hidden_dim)
        nn.init.normal_(self.token_embedding.weight, std=0.02) 
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
