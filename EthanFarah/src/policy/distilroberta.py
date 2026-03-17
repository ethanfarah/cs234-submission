"""DistilRoBERTa-based compression policy with Bernoulli per-token keep/drop."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel

from src.config import Config
from src.env.spaces import Observation
from src.policy.base import Policy
from src.policy.ratio_conditioning import RatioConditioner


class SelfAttentionHead(nn.Module):
    """Self-attention head: tokens negotiate keep/drop importance."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)
        nn.init.xavier_normal_(self.output.weight, gain=0.1)
        self.output.bias.data = torch.tensor([0.0, 0.2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        return self.output(x)


class ValueHead(nn.Module):
    """Learned value head: predicts scalar state value from encoder hidden states."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        nn.init.xavier_normal_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (1, seq_len, hidden_dim) -> scalar value (1,)."""
        return self.net(h).mean(dim=(1, 2))  # mean-pool over tokens


class DistilRoBERTaPolicy(Policy):
    """Frozen pretrained encoder + trainable head (PCRL-style).

    Loads pretrained distilroberta-base with native embeddings (frozen) and
    adds a trainable head (MLP or self-attention) outputting 2-class logits
    per token. Sampling uses Bernoulli (categorical over keep/drop) rather
    than top-k selection.
    """

    def __init__(self, config: Config, vocab_size: int = 50257) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained("distilroberta-base")
        for param in self.encoder.parameters():
            param.requires_grad = False
        hidden = self.encoder.config.hidden_size
        if config.head_type == "attention":
            self.head = SelfAttentionHead(hidden)
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 256),
                nn.GELU(),
                nn.Linear(256, 2),
            )
            nn.init.xavier_normal_(self.head[-1].weight, gain=0.1)
            self.head[-1].bias.data = torch.tensor([0.0, 0.2])
        self.ratio_conditioner = (
            RatioConditioner(hidden) if config.ratio_conditioned else None
        )
        self.value_head: ValueHead | None = None

    def enable_value_head(self) -> None:
        """Add a learned value head that shares the encoder hidden representation."""
        device = next(self.parameters()).device
        self.value_head = ValueHead(hidden_dim=256).to(device)

    def forward(self, obs: Observation) -> Tensor:
        """Return per-token keep/drop logits: (1, seq_len, 2)."""
        if obs.token_ids.dim() != 1:
            raise ValueError("expected unbatched obs (token_ids must be 1-D)")
        token_ids = obs.token_ids.unsqueeze(0)
        mask = obs.attention_mask.unsqueeze(0)
        pos_ids = obs.position_ids.unsqueeze(0).clamp(
            max=self.encoder.config.max_position_embeddings - 1,
        )
        x = self.encoder(
            input_ids=token_ids, attention_mask=mask, position_ids=pos_ids,
        ).last_hidden_state

        if self.ratio_conditioner is not None:
            x = self.ratio_conditioner(x, obs.target_compression_ratio)

        return self.head(x)  # (1, seq_len, 2)

    def forward_with_value(self, obs: Observation) -> tuple[Tensor, Tensor]:
        """Return (logits, value) — shared encoder, separate heads."""
        if self.value_head is None:
            raise RuntimeError("call enable_value_head() first")
        if obs.token_ids.dim() != 1:
            raise ValueError("expected unbatched obs (token_ids must be 1-D)")
        token_ids = obs.token_ids.unsqueeze(0)
        mask = obs.attention_mask.unsqueeze(0)
        pos_ids = obs.position_ids.unsqueeze(0).clamp(
            max=self.encoder.config.max_position_embeddings - 1,
        )
        x = self.encoder(
            input_ids=token_ids, attention_mask=mask, position_ids=pos_ids,
        ).last_hidden_state

        if self.ratio_conditioner is not None:
            x = self.ratio_conditioner(x, obs.target_compression_ratio)

        logits = self.head(x)
        # Value head operates on the shared intermediate representation
        # For MLP head, extract after GELU layer; for attention head, use proj output
        if isinstance(self.head, SelfAttentionHead):
            h = self.head.norm(self.head.proj(x) + self.head.attn(self.head.proj(x), self.head.proj(x), self.head.proj(x))[0])
        else:
            # Extract hidden after GELU (layers 0-2 of the Sequential: LN -> Linear -> GELU)
            h = self.head[2](self.head[1](self.head[0](x)))
        value = self.value_head(h)
        return logits, value
