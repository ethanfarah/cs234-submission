"""Task metrics: F1, EM, ROUGE, BERTScore computation."""

from __future__ import annotations

import re
import string
from collections import Counter

def normalize_answer(text: str) -> str:
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between normalized prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """1.0 if normalized strings match, else 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


_ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]
_scorer: "RougeScorer | None" = None


def _get_rouge_scorer() -> "RougeScorer":
    global _scorer
    if _scorer is None:
        from rouge_score.rouge_scorer import RougeScorer
        _scorer = RougeScorer(_ROUGE_KEYS, use_stemmer=True)
    return _scorer


def compute_rouge(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-1, ROUGE-2, and ROUGE-L F1 scores."""
    if not prediction.strip() and not reference.strip():
        return {k: 1.0 for k in _ROUGE_KEYS}
    if not prediction.strip() or not reference.strip():
        return {k: 0.0 for k in _ROUGE_KEYS}

    scorer = _get_rouge_scorer()
    # scorer.score(target, prediction) — note the reversed arg order
    scores = scorer.score(reference, prediction)
    return {k: scores[k].fmeasure for k in _ROUGE_KEYS}


def compute_bertscore(prediction: str, reference: str) -> float:
    """BERTScore F1 between prediction and reference."""
    from bert_score import score as bert_score_fn

    _, _, f1 = bert_score_fn(
        [prediction], [reference], lang="en", verbose=False,
    )
    return f1.item()
