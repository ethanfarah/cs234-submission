"""Run all baseline compression methods and save results."""

from __future__ import annotations

import json
import os
import sys

# Ensure .env is loaded (HF_TOKEN, etc.) when run as script.
import src.config  # noqa: F401
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.baselines.llmlingua2 import llmlingua2_compress
from src.baselines.random_drop import random_drop
from src.baselines.selective_context import selective_context
from src.data.tokenization import DEFAULT_MODEL_NAME, get_tokenizer
from src.data.types import CompressedPrompt, Prompt
from src.reward.metrics import compute_f1, compute_rouge

_ALL_METHODS = ["random_drop", "selective_context", "llmlingua2"]


def _methods_for_dataset(dataset: str) -> list[str]:
    # Without HF_TOKEN, Llama (gated) cannot be loaded; only random_drop runs.
    if not os.environ.get("HF_TOKEN"):
        return ["random_drop"]
    if dataset == "meetingbank":
        # LLMLingua-2 uses BERT (512 token limit); MeetingBank transcripts
        # average ~17k chars (~4k tokens), so BERT sees <15% of each document.
        # Excluded to avoid reporting misleading scores.
        return [m for m in _ALL_METHODS if m != "llmlingua2"]
    return _ALL_METHODS


@dataclass
class SampleConfig:
    """Dataset sampling configuration."""

    max_samples: int = 200
    max_length: int = 1024


@dataclass
class EvalConfig:
    """Configuration for a single (method, ratio, dataset) evaluation run."""

    method: str
    target_ratio: float
    dataset: str


def _load_dataset(dataset: str, max_samples: int, max_length: int) -> list[Prompt]:
    if dataset == "squad":
        from src.data.squad import load_squad

        return load_squad(split="validation", max_samples=max_samples, max_length=max_length)
    if dataset == "meetingbank":
        from src.data.meetingbank import load_meetingbank

        return load_meetingbank(split="validation", max_samples=max_samples, max_length=max_length)
    raise ValueError(f"Unknown dataset: {dataset}")


def _score(decoded: str, metadata: dict[str, Any], dataset: str) -> float:
    """Score compressed text against metadata for the given dataset.

    For SQuAD: token-level F1 between compressed text and gold answers.
    For MeetingBank: proxy ROUGE-L between compressed transcript and reference summary.
      This is a proxy because the reference is abstractive — it rephrases rather than
      quotes the transcript verbatim, so scores will be low for all methods. Use for
      relative ranking only; true eval requires LLM-generated summaries.
    """
    if dataset == "squad":
        answers = metadata["answer_texts"]
        if not answers:
            return compute_f1(decoded, "")
        return max(compute_f1(decoded, ans) for ans in answers)
    if dataset == "meetingbank":
        return compute_rouge(decoded, metadata["reference_summary"])["rougeL"]
    raise ValueError(f"Unknown dataset: {dataset}")


def _metric_name(dataset: str) -> str:
    if dataset == "squad":
        return "f1"
    if dataset == "meetingbank":
        return "proxy_rougeL"
    raise ValueError(f"Unknown dataset: {dataset}")


def _run_method(method_name: str, prompt: Prompt, keep_ratio: float) -> CompressedPrompt:
    if method_name == "random_drop":
        return random_drop(prompt, keep_ratio, seed=42)
    if method_name == "selective_context":
        return selective_context(prompt, keep_ratio)
    if method_name == "llmlingua2":
        return llmlingua2_compress(prompt, keep_ratio)
    raise ValueError(f"Unknown method: {method_name}")


def _eval_single(config: EvalConfig, prompt: Prompt) -> tuple[float, float]:
    tokenizer = get_tokenizer(DEFAULT_MODEL_NAME)  # lru_cached after first call
    compressed = _run_method(config.method, prompt, keep_ratio=config.target_ratio)
    decoded = tokenizer.decode(compressed.token_ids, skip_special_tokens=True)
    score = _score(decoded, prompt.metadata, config.dataset)
    return score, compressed.compression_ratio


def _eval_method(config: EvalConfig, prompts: list[Prompt]) -> dict[str, Any]:
    if not prompts:
        raise ValueError(f"prompts list is empty for dataset={config.dataset}")
    name = _metric_name(config.dataset)
    pairs = [_eval_single(config, p) for p in prompts]
    scores = [s for s, _ in pairs]
    actual_ratios = [r for _, r in pairs]
    mean_score = sum(scores) / len(scores)
    mean_ratio = sum(actual_ratios) / len(actual_ratios)
    print(
        f"{config.method:20s} | ratio={config.target_ratio:.1f} | {name}={mean_score:.4f}",
        file=sys.stderr,
    )
    return {
        "method": config.method,
        "dataset": config.dataset,
        "target_ratio": config.target_ratio,
        "actual_ratio": mean_ratio,
        "score": mean_score,
        "score_metric": name,
    }


def run_all_baselines(
    dataset: str,
    ratios: list[float],
    config: SampleConfig | None = None,
) -> None:
    """Run all baseline methods at the given compression ratios."""
    if config is None:
        config = SampleConfig()
    prompts = _load_dataset(dataset, config.max_samples, config.max_length)
    methods = _methods_for_dataset(dataset)
    results = [
        _eval_method(EvalConfig(method, ratio, dataset), prompts)
        for ratio in ratios
        for method in methods
    ]
    out_dir = Path(__file__).parent.parent / "outputs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}", file=sys.stderr)


def main() -> None:
    """Run baselines on default datasets and ratios."""
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    run_all_baselines("squad", ratios)
    run_all_baselines("meetingbank", ratios)


if __name__ == "__main__":
    main()
