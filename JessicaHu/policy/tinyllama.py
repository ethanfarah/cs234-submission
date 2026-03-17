"""TinyLlama/Phi-2 causal LM compression policy."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForCausalLM

from src.config import PolicyConfig
from src.env.spaces import Observation
from src.policy.base import Policy
from src.policy.ratio_conditioning import RatioConditioner

_VOCAB_SIZE = 128_256  # Llama-3.1-8B tokenizer vocab


class TinyLlamaPolicy(Policy):
    """Fully causal compression policy using TinyLlama.

    Loads pretrained TinyLlama and adds a classification head for keep/drop.
    Replaces the pretrained embedding and LM head to match the Llama-3.1-8B
    tokenizer vocabulary (128,256 tokens) and free ~65M params from the LM head.
    """

    def __init__(self, config: PolicyConfig) -> None:
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            config.pretrained_name, torch_dtype=torch.bfloat16,
        )
        hidden = self.backbone.config.hidden_size
        embed = nn.Embedding(_VOCAB_SIZE, hidden)
        nn.init.normal_(embed.weight, std=0.02)
        self.backbone.model.embed_tokens = embed.to(torch.bfloat16)
        self.backbone.config.vocab_size = _VOCAB_SIZE
        # lm_head is not called in forward (we go through backbone.model, not backbone),
        # but replacing it drops ~65M params from backbone.parameters().
        self.backbone.lm_head = nn.Identity()
        self.head_norm = nn.LayerNorm(hidden)  # fp32 — tames extreme backbone outputs
        self.head = nn.Linear(hidden, 2)  # fp32 — receives normalized hidden states
        nn.init.normal_(self.head.weight, std=0.02)  # GPT-2 style init
        nn.init.zeros_(self.head.bias)
        self.ratio_conditioner = (
            RatioConditioner(hidden)  # fp32
            if config.ratio_conditioned else None
        )

    def forward(self, obs: Observation) -> Tensor:
        if obs.token_ids.dim() != 1:
            raise ValueError("expected unbatched obs (token_ids must be 1-D)")
        token_ids = obs.token_ids.unsqueeze(0)
        mask = obs.attention_mask.unsqueeze(0)
        # Clamp to valid range; positions beyond max alias to the last embedding.
        pos_ids = obs.position_ids.unsqueeze(0).clamp(
            max=self.backbone.config.max_position_embeddings - 1,
        )
        x = self.backbone.model(
            input_ids=token_ids, attention_mask=mask, position_ids=pos_ids,
        ).last_hidden_state.float()  # bf16 → fp32 for stable head/sampling

        if self.ratio_conditioner is not None:
            x = self.ratio_conditioner(x, obs.target_compression_ratio)

        x = self.head_norm(x)
        return self.head(x)  # (1, seq_len, 2) — fp32 logits for stable softmax/loss
