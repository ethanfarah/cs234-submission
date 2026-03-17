"""Learned reward model using DistilBERT + regression head."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from transformers import DistilBertModel, DistilBertTokenizerFast

from src.config import RewardConfig
from src.data.tokenization import DEFAULT_MODEL_NAME, get_tokenizer
from src.data.types import RewardInput
from src.reward.base import RewardFunction

_DISTILBERT_NAME = "distilbert-base-uncased"


class LearnedRewardModel(nn.Module):
    """DistilBERT encoder with linear regression head for quality prediction."""

    def __init__(self, pretrained_name: str = _DISTILBERT_NAME) -> None:
        super().__init__()
        self.encoder = DistilBertModel.from_pretrained(pretrained_name)
        hidden = self.encoder.config.hidden_size
        self.regression_head = nn.Linear(hidden * 2, 1)

    def forward(
        self,
        original_ids: Tensor,
        compressed_ids: Tensor,
        orig_mask: Tensor,
        comp_mask: Tensor,
    ) -> Tensor:
        """Predict quality score. Inputs shape (batch, seq_len). Returns (batch,)."""
        orig_cls = self.encoder(input_ids=original_ids, attention_mask=orig_mask).last_hidden_state[:, 0, :]
        comp_cls = self.encoder(input_ids=compressed_ids, attention_mask=comp_mask).last_hidden_state[:, 0, :]
        return self.regression_head(torch.cat([orig_cls, comp_cls], dim=-1)).squeeze(-1)


class LearnedReward(RewardFunction):
    """Reward function using a trained LearnedRewardModel."""

    def __init__(self, config: RewardConfig) -> None:
        if config.learned_model_path is None:
            raise ValueError("LearnedReward requires config.learned_model_path to be set")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LearnedRewardModel()
        self.model.load_state_dict(torch.load(config.learned_model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()
        self.db_tok = DistilBertTokenizerFast.from_pretrained(_DISTILBERT_NAME)
        self.llama_tok = get_tokenizer(DEFAULT_MODEL_NAME)

    def compute(self, reward_input: RewardInput) -> Tensor:
        """Return scalar reward for this compression."""
        orig_text = reward_input.original.text
        comp_text = self.llama_tok.decode(reward_input.compressed.token_ids.cpu(), skip_special_tokens=True)
        enc_orig = self.db_tok(orig_text, truncation=True, max_length=512, return_tensors="pt")
        enc_comp = self.db_tok(comp_text, truncation=True, max_length=512, return_tensors="pt")
        orig_ids = enc_orig["input_ids"].to(self.device)
        orig_mask = enc_orig["attention_mask"].to(self.device)
        comp_ids = enc_comp["input_ids"].to(self.device)
        comp_mask = enc_comp["attention_mask"].to(self.device)
        with torch.no_grad():
            return self.model(orig_ids, comp_ids, orig_mask, comp_mask).squeeze(0)

    def is_dense(self) -> bool:
        return False
