from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from src.config import KLDirection
from src.data.types import Prompt
from src.llm.frozen_llm import FrozenLLM


class KLCache:
    def __init__(self, llm: FrozenLLM) -> None:
        self.llm = llm
        self._cached_logits: Tensor | None = None

    def cache_full_prompt(self, prompt: Prompt) -> None:
        ids = prompt.token_ids.unsqueeze(0)
        self._cached_logits = self.llm.get_logits(ids).squeeze(0)

    def compute_kl(
        self,
        compressed_ids: Tensor,
        keep_mask: Tensor,
        direction: KLDirection = KLDirection.FORWARD,
    ) -> Tensor:
        comp_logits = self.llm.get_logits(compressed_ids).squeeze(0)
        full_logits = self._cached_logits[keep_mask]

        full_log_p = F.log_softmax(full_logits, dim=-1)
        comp_log_p = F.log_softmax(comp_logits, dim=-1)
        if direction == KLDirection.FORWARD:
            kl = F.kl_div(comp_log_p, full_log_p, reduction="none", log_target=True)
        elif direction == KLDirection.REVERSE:
            kl = F.kl_div(full_log_p, comp_log_p, reduction="none", log_target=True)
        return kl.sum(dim=-1)

    def clear(self) -> None:
        self._cached_logits = None
