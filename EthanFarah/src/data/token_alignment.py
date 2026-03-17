"""Token alignment between policy and LLM tokenizer spaces via char offsets."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast


@dataclass
class AlignmentResult:
    """Mapping between two tokenizer spaces for the same text."""

    policy_ids: Tensor             # (policy_len,)
    policy_attention_mask: Tensor  # (policy_len,)
    llm_ids: Tensor                # (llm_len,)
    llm_attention_mask: Tensor     # (llm_len,)
    llm_to_policy: list[list[int]]  # for each LLM token, overlapping policy indices


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _build_overlap_map(
    policy_offsets: list[tuple[int, int]],
    llm_offsets: list[tuple[int, int]],
) -> list[list[int]]:
    """For each LLM token, find indices of overlapping policy tokens."""
    result: list[list[int]] = []
    for llm_span in llm_offsets:
        overlapping: list[int] = []
        if llm_span != (0, 0):
            for pi, pol_span in enumerate(policy_offsets):
                if pol_span != (0, 0) and _spans_overlap(llm_span, pol_span):
                    overlapping.append(pi)
        result.append(overlapping)
    return result


class TokenAligner:
    """Maps keep/drop decisions between policy and LLM tokenizer spaces."""

    def __init__(
        self,
        policy_tokenizer: PreTrainedTokenizerFast,
        llm_tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        self.policy_tok = policy_tokenizer
        self.llm_tok = llm_tokenizer

    def align(
        self, text: str, max_policy_tokens: int, max_llm_tokens: int,
    ) -> AlignmentResult:
        """Tokenize with both tokenizers and compute char-level alignment."""
        policy_enc = self.policy_tok(
            text,
            max_length=max_policy_tokens,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        # GPT-2 adds no special tokens by default. For BOS-prepending LLMs
        # (LLaMA, T5), add_special_tokens=False or handle (0,0) offsets.
        llm_enc = self.llm_tok(
            text,
            max_length=max_llm_tokens,
            truncation=True,
            padding=False,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        policy_offsets = [tuple(x) for x in policy_enc["offset_mapping"][0].tolist()]
        llm_offsets = [tuple(x) for x in llm_enc["offset_mapping"][0].tolist()]
        llm_to_policy = _build_overlap_map(policy_offsets, llm_offsets)

        return AlignmentResult(
            policy_ids=policy_enc["input_ids"].squeeze(0),
            policy_attention_mask=policy_enc["attention_mask"].squeeze(0),
            llm_ids=llm_enc["input_ids"].squeeze(0),
            llm_attention_mask=llm_enc["attention_mask"].squeeze(0),
            llm_to_policy=llm_to_policy,
        )


def map_mask(policy_mask: Tensor, alignment: AlignmentResult) -> Tensor:
    """Convert policy-space keep/drop mask to LLM-space mask.

    Conservative: an LLM token is dropped if ANY overlapping policy token
    is dropped. LLM tokens with no overlapping policy tokens are kept.
    """
    llm_len = alignment.llm_ids.shape[0]
    llm_mask = torch.ones(llm_len, dtype=policy_mask.dtype, device=policy_mask.device)

    for llm_idx, policy_indices in enumerate(alignment.llm_to_policy):
        if not policy_indices:
            continue
        if not policy_mask[policy_indices].all():
            llm_mask[llm_idx] = 0

    return llm_mask
