"""Sweep keep rates to find where Llama-3.2-1B's ROUGE-L falls off a cliff.

For ~20 SQuAD prompts, generates baseline output (full prompt) then compresses
at each keep rate using random masking + text bridge (decode RoBERTa -> re-tokenize
Llama). Prints a table showing mean/std/min ROUGE-L at each rate.

Usage:
    python scripts/rouge_cliff_sweep.py --device cuda
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass

os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.config import LlmConfig
from src.data.squad import load_squad
from src.llm.frozen_llm import FrozenLLM
from src.reward.metrics import compute_rouge

KEEP_RATES = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50]
NUM_PROMPTS = 20
LLM_MODEL = "meta-llama/Llama-3.2-1B"
POLICY_MODEL = "distilroberta-base"
CLIFF_THRESHOLD = 0.8


@dataclass
class SweepModels:
    """Bundle of models/tokenizers needed for the sweep."""

    policy_tok: PreTrainedTokenizerFast
    llm_tok: PreTrainedTokenizerFast
    frozen_llm: FrozenLLM


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROUGE-L cliff sweep")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-prompts", type=int, default=NUM_PROMPTS)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _make_llm_config() -> LlmConfig:
    return LlmConfig(
        model_name=LLM_MODEL,
        quantize=True,
        max_new_tokens=64,
    )


def _load_models(device: str) -> SweepModels:
    """Load policy tokenizer, LLM tokenizer, and frozen LLM."""
    print(f"Loading policy tokenizer ({POLICY_MODEL})...")
    policy_tok = AutoTokenizer.from_pretrained(POLICY_MODEL)
    print(f"Loading LLM ({LLM_MODEL}, 4-bit quantized)...")
    frozen_llm = FrozenLLM(_make_llm_config(), device=device)
    return SweepModels(
        policy_tok=policy_tok,
        llm_tok=frozen_llm.tokenizer,
        frozen_llm=frozen_llm,
    )


def _random_keep_mask(seq_len: int, keep_rate: float, rng: random.Random) -> torch.Tensor:
    """Random binary mask with given keep rate."""
    mask = torch.zeros(seq_len, dtype=torch.long)
    indices = list(range(seq_len))
    rng.shuffle(indices)
    n_keep = max(1, int(seq_len * keep_rate))
    for i in indices[:n_keep]:
        mask[i] = 1
    return mask


def _generate_baseline(frozen_llm: FrozenLLM, llm_tok: PreTrainedTokenizerFast, prompt_text: str) -> str:
    """Generate baseline output from full prompt."""
    ids = llm_tok(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return frozen_llm.generate(ids)


def _generate_compressed(models: SweepModels, policy_ids: torch.Tensor, mask: torch.Tensor) -> str:
    """Text bridge: keep masked RoBERTa tokens -> decode -> re-tokenize Llama -> generate."""
    kept_ids = policy_ids[mask.bool()]
    decoded_text = models.policy_tok.decode(kept_ids, skip_special_tokens=True)
    if not decoded_text.strip():
        decoded_text = models.policy_tok.decode(kept_ids, skip_special_tokens=False)
    retokenized = models.llm_tok(decoded_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return models.frozen_llm.generate(retokenized)


def _sweep_single_prompt(
    prompt_text: str,
    baseline_output: str,
    models: SweepModels,
    seed: int,
) -> dict[float, float]:
    """Run all keep rates on one prompt, return {rate: rouge_l}."""
    policy_ids = models.policy_tok(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    results: dict[float, float] = {}
    for rate in KEEP_RATES:
        rng = random.Random(seed)
        mask = _random_keep_mask(policy_ids.shape[0], rate, rng)
        compressed_output = _generate_compressed(models, policy_ids, mask)
        rouge = compute_rouge(compressed_output, baseline_output)
        results[rate] = rouge["rougeL"]
    return results


def _compute_stats(scores: list[float]) -> tuple[float, float, float]:
    """Return (mean, sample_std, min) for a list of scores."""
    mean = sum(scores) / len(scores)
    if len(scores) < 2:
        return mean, 0.0, min(scores)
    variance = sum((s - mean) ** 2 for s in scores) / (len(scores) - 1)
    return mean, math.sqrt(variance), min(scores)


def _print_table(all_results: list[dict[float, float]]) -> None:
    """Print summary table and identify the cliff."""
    header = f"{'keep_rate':>10} | {'mean_rouge':>10} | {'std':>8} | {'min_rouge':>10}"
    print("\n" + "=" * 50)
    print("ROUGE-L CLIFF SWEEP RESULTS")
    print("=" * 50)
    print(header)
    print("-" * 50)

    cliff_rate = None
    for rate in KEEP_RATES:
        scores = [r[rate] for r in all_results]
        mean, std, min_score = _compute_stats(scores)
        print(f"{rate:>10.2f} | {mean:>10.4f} | {std:>8.4f} | {min_score:>10.4f}")
        if cliff_rate is None and mean < CLIFF_THRESHOLD:
            cliff_rate = rate

    _print_cliff_verdict(cliff_rate)


def _print_cliff_verdict(cliff_rate: float | None) -> None:
    """Print cliff detection summary."""
    print("-" * 50)
    if cliff_rate is not None:
        print(f"CLIFF DETECTED: mean ROUGE-L drops below {CLIFF_THRESHOLD} at keep_rate={cliff_rate:.2f}")
    else:
        print(f"NO CLIFF: mean ROUGE-L stays above {CLIFF_THRESHOLD} at all tested keep rates")
    print()


def _run_sweep(models: SweepModels, prompts: list, seed: int) -> list[dict[float, float]]:
    """Run the sweep over all prompts, return per-prompt results."""
    all_results: list[dict[float, float]] = []
    for i, prompt in enumerate(prompts):
        prompt_text = f"{prompt.text}\n\nAnswer:"
        sys.stderr.write(f"\rProcessing prompt {i + 1}/{len(prompts)}...")
        sys.stderr.flush()
        baseline_output = _generate_baseline(models.frozen_llm, models.llm_tok, prompt_text)
        results = _sweep_single_prompt(prompt_text, baseline_output, models, seed)
        all_results.append(results)
    sys.stderr.write("\n")
    return all_results


def main() -> None:
    args = _parse_args()
    models = _load_models(args.device)
    print(f"Loading {args.num_prompts} SQuAD prompts...")
    prompts = load_squad("validation", max_samples=args.num_prompts)
    print("Generating baselines...")
    all_results = _run_sweep(models, prompts, args.seed)
    _print_table(all_results)


if __name__ == "__main__":
    main()
