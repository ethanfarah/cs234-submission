"""Run all baseline compression methods and save results."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.baselines.llmlingua2 import llmlingua2_compress
from src.baselines.random_drop import random_drop
from src.baselines.selective_context import selective_context
from src.data.tokenization import DEFAULT_MODEL_NAME, get_tokenizer
from src.data.types import CompressedPrompt, Prompt
from src.reward.metrics import compute_f1

_ALL_METHODS = ["random_drop", "selective_context", "llmlingua2"]


@dataclass
class SampleConfig:
    """Dataset sampling configuration."""

    max_samples: int = 200
    max_length: int = 1024


@dataclass
class EvalConfig:
    """Configuration for a single (method, ratio) evaluation run."""

    method: str
    target_ratio: float


def _load_squad(max_samples: int, max_length: int) -> list[Prompt]:
    from src.data.squad import load_squad
    return load_squad(split="validation", max_samples=max_samples, max_length=max_length)


def _score(decoded: str, metadata: dict[str, Any]) -> float:
    """Token-level F1 between compressed text and gold answers."""
    answers = metadata["answer_texts"]
    if not answers:
        return compute_f1(decoded, "")
    return max(compute_f1(decoded, ans) for ans in answers)


def _run_method(method_name: str, prompt: Prompt, keep_ratio: float) -> CompressedPrompt:
    if method_name == "random_drop":
        return random_drop(prompt, keep_ratio, seed=42)
    if method_name == "selective_context":
        return selective_context(prompt, keep_ratio)
    if method_name == "llmlingua2":
        return llmlingua2_compress(prompt, keep_ratio)
    raise ValueError(f"Unknown method: {method_name}")


def _eval_single(config: EvalConfig, prompt: Prompt) -> tuple[float, float]:
    tokenizer = get_tokenizer(DEFAULT_MODEL_NAME)
    compressed = _run_method(config.method, prompt, keep_ratio=config.target_ratio)
    decoded = tokenizer.decode(compressed.token_ids, skip_special_tokens=True)
    score = _score(decoded, prompt.metadata)
    return score, compressed.compression_ratio


def _eval_method(config: EvalConfig, prompts: list[Prompt]) -> dict[str, Any]:
    if not prompts:
        raise ValueError("prompts list is empty")
    pairs = [_eval_single(config, p) for p in prompts]
    scores = [s for s, _ in pairs]
    actual_ratios = [r for _, r in pairs]
    mean_score = sum(scores) / len(scores)
    mean_ratio = sum(actual_ratios) / len(actual_ratios)
    print(
        f"{config.method:20s} | ratio={config.target_ratio:.1f} | f1={mean_score:.4f}",
        file=sys.stderr,
    )
    return {
        "method": config.method,
        "dataset": "squad",
        "target_ratio": config.target_ratio,
        "actual_ratio": mean_ratio,
        "score": mean_score,
        "score_metric": "f1",
    }


def run_all_baselines(
    ratios: list[float],
    config: SampleConfig | None = None,
) -> None:
    """Run all baseline methods at the given compression ratios on SQuAD."""
    if config is None:
        config = SampleConfig()
    prompts = _load_squad(config.max_samples, config.max_length)
    results = [
        _eval_method(EvalConfig(method, ratio), prompts)
        for ratio in ratios
        for method in _ALL_METHODS
    ]
    out_dir = Path(__file__).parent.parent / "outputs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "squad.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}", file=sys.stderr)


def main() -> None:
    """Run baselines on SQuAD at standard ratios."""
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    run_all_baselines(ratios)


if __name__ == "__main__":
    main()
