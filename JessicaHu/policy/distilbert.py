from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from transformers import DistilBertModel

from src.config import PolicyConfig
from src.env.spaces import Observation
from src.policy.base import Policy
from src.policy.ratio_conditioning import RatioConditioner

_VOCAB_SIZE = 128_256  


class DistilBERTPolicy(Policy):
    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(config.pretrained_name)
        hidden = self.encoder.config.dim  # DistilBertConfig uses .dim, not .hidden_size
        self.encoder.embeddings.word_embeddings = nn.Embedding(_VOCAB_SIZE, hidden)
        nn.init.normal_(self.encoder.embeddings.word_embeddings.weight, std=0.02)
        self.encoder.config.vocab_size = _VOCAB_SIZE
        self.head_norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, 2)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)
        self.ratio_conditioner = (
            RatioConditioner(hidden) if config.ratio_conditioned else None
        )

    def forward(self, obs: Observation) -> Tensor:
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

        x = self.head_norm(x)
        return self.head(x)  
