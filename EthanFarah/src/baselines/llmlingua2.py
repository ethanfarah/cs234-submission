"""LLMLingua-2: token classification-based compression baseline."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, PreTrainedTokenizerFast

from src.data.tokenization import get_tokenizer
from src.data.types import CompressedPrompt, Prompt

_classifier: AutoModelForTokenClassification | None = None
_classifier_tokenizer: PreTrainedTokenizerFast | None = None


@dataclass
class LLMLingua2Config:
    model_name: str = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
    llm_model_name: str = "meta-llama/Llama-3.1-8B"


def _get_classifier(
    model_name: str,
) -> tuple[AutoModelForTokenClassification, PreTrainedTokenizerFast]:
    global _classifier, _classifier_tokenizer
    if _classifier is None or _classifier_tokenizer.name_or_path != model_name:
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForTokenClassification.from_pretrained(model_name)
        mdl.eval()
        _classifier_tokenizer, _classifier = tok, mdl  # atomic: both or neither
    return _classifier, _classifier_tokenizer


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _select_kept_spans(
    keep_probs: torch.Tensor,
    offsets: list[tuple[int, int]],
    keep_ratio: float,
) -> list[tuple[int, int]]:
    """Return char spans of the top keep_ratio fraction of non-special tokens by keep prob."""
    valid_mask = torch.tensor(
        [span != (0, 0) for span in offsets], dtype=torch.bool, device=keep_probs.device
    )
    valid_probs = keep_probs[valid_mask]
    valid_positions = valid_mask.nonzero(as_tuple=True)[0]

    n_keep = int(round(valid_probs.shape[0] * keep_ratio))
    if n_keep == 0:
        return []

    _, rel_indices = valid_probs.topk(min(n_keep, valid_probs.shape[0]))
    abs_positions = valid_positions[rel_indices]
    return [offsets[i] for i in abs_positions.tolist()]


def llmlingua2_compress(
    prompt: Prompt,
    keep_ratio: float,
    config: LLMLingua2Config | None = None,
) -> CompressedPrompt:
    """Compress using LLMLingua-2 token classification model."""
    if not (0.0 <= keep_ratio <= 1.0):
        raise ValueError(f"keep_ratio must be in [0, 1], got {keep_ratio}")
    if config is None:
        config = LLMLingua2Config()

    model, clf_tok = _get_classifier(config.model_name)
    # BERT context limit: covers only the first ~512 tokens of long documents.
    # LLMLingua-2 is therefore excluded from MeetingBank evaluation — see run_baselines.py.
    clf_enc = clf_tok(
        prompt.text, return_tensors="pt", return_offsets_mapping=True,
        max_length=model.config.max_position_embeddings, truncation=True,
    )
    clf_offsets = [tuple(x) for x in clf_enc["offset_mapping"][0].tolist()]

    inputs = {k: v for k, v in clf_enc.items() if k != "offset_mapping"}
    with torch.no_grad():
        logits = model(**inputs).logits  # (1, n_clf, 2)

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    keep_label = next(
        (idx for idx, lbl in id2label.items() if lbl.lower() in ("preserve", "keep")),
        None,
    )
    if keep_label is None:
        warnings.warn(
            f"No 'keep'/'preserve' label in id2label {id2label}; defaulting to index 1 "
            "(LABEL_1 = keep per LLMLingua-2 model card).",
            UserWarning,
            stacklevel=2,
        )
        keep_label = 1
    keep_probs = logits.softmax(dim=-1)[0, :, keep_label]
    kept_spans = _select_kept_spans(keep_probs, clf_offsets, keep_ratio)

    llm_tok = get_tokenizer(config.llm_model_name)
    llm_enc = llm_tok(
        prompt.text,
        return_tensors="pt",
        return_offsets_mapping=True,
        max_length=len(prompt.token_ids),
        truncation=True,
    )
    llm_offsets = [tuple(x) for x in llm_enc["offset_mapping"][0].tolist()]
    n_llm = len(llm_offsets)

    if n_llm != len(prompt.token_ids):
        raise RuntimeError(
            f"Re-tokenized prompt has {n_llm} tokens but original has {len(prompt.token_ids)}. "
            "Ensure prompt and baseline use the same LLM tokenizer."
        )

    keep_mask = torch.zeros(n_llm, dtype=torch.long)
    for i, span in enumerate(llm_offsets):
        if span == (0, 0) or any(_spans_overlap(span, ks) for ks in kept_spans):
            keep_mask[i] = 1  # special tokens (BOS/EOS) always kept

    kept_ids = prompt.token_ids[keep_mask.bool()]
    ratio = kept_ids.shape[0] / max(n_llm, 1)
    return CompressedPrompt(token_ids=kept_ids, keep_mask=keep_mask, compression_ratio=ratio)
