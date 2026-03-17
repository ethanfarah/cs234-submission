"""DistilBERT-based compression policy."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from transformers import DistilBertModel

from src.config import PolicyConfig
from src.env.spaces import Observation
from src.policy.base import Policy
from src.policy.ratio_conditioning import RatioConditioner

_VOCAB_SIZE = 128_256  # Llama-3.1-8B tokenizer vocab


class DistilBERTPolicy(Policy):
    """Bidirectional within chunk, causal across chunks (block-causal).

    Loads pretrained DistilBERT and adds a keep/drop classification head.
    Replaces the pretrained word embeddings with a new embedding layer sized
    for the Llama-3.1-8B tokenizer vocabulary (128,256 tokens).
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(config.pretrained_name)
        hidden = self.encoder.config.dim  # DistilBertConfig uses .dim, not .hidden_size
        self.encoder.embeddings.word_embeddings = nn.Embedding(_VOCAB_SIZE, hidden)
        nn.init.normal_(self.encoder.embeddings.word_embeddings.weight, std=0.02)
        self.encoder.config.vocab_size = _VOCAB_SIZE
        self.head = nn.Linear(hidden, 2)
        self.ratio_conditioner = (
            RatioConditioner(hidden) if config.ratio_conditioned else None
        )

    def forward(self, obs: Observation) -> Tensor:
        if obs.token_ids.dim() != 1:
            raise ValueError("expected unbatched obs (token_ids must be 1-D)")
        token_ids = obs.token_ids.unsqueeze(0)
        mask = obs.attention_mask.unsqueeze(0)
        # Clamp to valid range; positions beyond max alias to the last embedding.
        pos_ids = obs.position_ids.unsqueeze(0).clamp(
            max=self.encoder.config.max_position_embeddings - 1,
        )
        x = self.encoder(
            input_ids=token_ids, attention_mask=mask, position_ids=pos_ids,
        ).last_hidden_state

        if self.ratio_conditioner is not None:
            x = self.ratio_conditioner(x, obs.target_compression_ratio)

        return self.head(x)  # (1, seq_len, 2)
