"""Selective Context: entropy-based phrase-level token selection baseline."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.config import LlmConfig
from src.data.types import CompressedPrompt, Prompt
from src.llm.frozen_llm import FrozenLLM

_frozen_llm: FrozenLLM | None = None
_nlp = None


@dataclass
class SelectiveContextConfig:
    model_name: str = "meta-llama/Llama-3.1-8B"


def _get_frozen_llm(model_name: str) -> FrozenLLM:
    global _frozen_llm
    if _frozen_llm is None or _frozen_llm.config.model_name != model_name:
        _frozen_llm = FrozenLLM(LlmConfig(model_name=model_name, quantize=True))
    return _frozen_llm


def _get_nlp() -> "spacy.language.Language":
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _compute_self_info(token_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token self-information -log P(x_i | x_{<i})."""
    # logits[i] predicts token_ids[i+1], so shifted targets are token_ids[1:]
    log_probs = F.log_softmax(logits[:-1], dim=-1)  # (n-1, vocab)
    token_log_probs = log_probs[range(len(token_ids) - 1), token_ids[1:]]
    # First token has no context — assign self-info 0 (matches device/dtype of log_probs)
    return torch.cat([token_log_probs.new_zeros(1), -token_log_probs])


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _group_into_units(
    self_info: torch.Tensor,
    offsets: list[tuple[int, int]],
    phrase_spans: list[tuple[int, int]],
) -> list[tuple[float, list[tuple[int, int]]]]:
    """Return (avg_score, char_spans) for each noun phrase and individual token."""
    in_phrase: dict[int, int] = {}
    for phrase_idx, pspan in enumerate(phrase_spans):
        for tok_idx, tok_span in enumerate(offsets):
            if tok_span == (0, 0):
                continue
            if _spans_overlap(tok_span, pspan):
                in_phrase[tok_idx] = phrase_idx

    phrase_tokens: dict[int, list[int]] = {}
    for tok_idx, phrase_idx in in_phrase.items():
        phrase_tokens.setdefault(phrase_idx, []).append(tok_idx)

    units: list[tuple[float, list[tuple[int, int]]]] = []
    for tok_indices in phrase_tokens.values():
        avg_info = float(self_info[tok_indices].mean())
        units.append((avg_info, [offsets[i] for i in tok_indices]))

    phrase_set = set(in_phrase.keys())
    for tok_idx, tok_span in enumerate(offsets):
        if tok_idx in phrase_set or tok_span == (0, 0):
            continue
        units.append((float(self_info[tok_idx]), [tok_span]))

    return units


def _build_keep_mask(
    units_sorted: list[tuple[float, list[tuple[int, int]]]],
    llm_offsets: list[tuple[int, int]],
    target_keep: int,
) -> torch.Tensor:
    """Greedily select highest-scoring units until target Llama token count is met.

    Special tokens (BOS/EOS; offset (0,0)) are always kept and pre-counted toward
    target_keep. Greedy selection adds whole phrase units, so overshoot by up to
    one phrase unit is expected and correct for phrase-level aggregation.
    """
    n_llm = len(llm_offsets)
    kept_spans: list[tuple[int, int]] = []
    # Special tokens are always kept; pre-count them so the budget stays honest
    llm_kept = sum(1 for span in llm_offsets if span == (0, 0))

    for _score, char_spans in units_sorted:
        if llm_kept >= target_keep:
            break
        unit_count = sum(
            1
            for span in llm_offsets
            if span != (0, 0) and any(_spans_overlap(span, cs) for cs in char_spans)
        )
        kept_spans.extend(char_spans)
        llm_kept += unit_count

    mask = torch.zeros(n_llm, dtype=torch.long)
    for i, span in enumerate(llm_offsets):
        if span == (0, 0) or any(_spans_overlap(span, ks) for ks in kept_spans):
            mask[i] = 1
    return mask


def selective_context(
    prompt: Prompt,
    keep_ratio: float,
    config: SelectiveContextConfig | None = None,
) -> CompressedPrompt:
    """Keep highest self-information phrases/tokens up to keep_ratio of Llama tokens."""
    if not (0.0 <= keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in [0, 1], got {keep_ratio}")
    if config is None:
        config = SelectiveContextConfig()

    llm = _get_frozen_llm(config.model_name)
    enc = llm.tokenizer(
        prompt.text, return_tensors="pt", return_offsets_mapping=True,
        max_length=len(prompt.token_ids), truncation=True,
    )
    token_ids = enc["input_ids"][0]
    offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]

    logits = llm.get_logits(enc["input_ids"])[0]  # (seq_len, vocab)
    self_info = _compute_self_info(token_ids, logits)
    phrase_spans = [(c.start_char, c.end_char) for c in _get_nlp()(prompt.text).noun_chunks]
    units = _group_into_units(self_info, offsets, phrase_spans)

    n_llm = len(offsets)
    if n_llm != len(prompt.token_ids):
        raise RuntimeError(
            f"Re-tokenized prompt has {n_llm} tokens but original has {len(prompt.token_ids)}."
        )

    target_keep = int(round(n_llm * keep_ratio))
    units_sorted = sorted(units, key=lambda u: u[0], reverse=True)
    keep_mask = _build_keep_mask(units_sorted, offsets, target_keep)

    kept_ids = prompt.token_ids[keep_mask.bool()]
    ratio = kept_ids.shape[0] / max(n_llm, 1)
    return CompressedPrompt(token_ids=kept_ids, keep_mask=keep_mask, compression_ratio=ratio)
