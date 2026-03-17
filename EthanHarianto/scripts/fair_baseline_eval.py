#!/usr/bin/env python3
"""Fair baseline comparison: all methods feed compressed prompts to the same LLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import src.config  # noqa: F401 – loads .env
from src.baselines.random_drop import random_drop
from src.baselines.selective_context import selective_context
from src.baselines.llmlingua2 import llmlingua2_compress
from src.data.squad import load_squad
from src.data.types import Prompt
from src.llm.frozen_llm import FrozenLLM
from src.config import LlmConfig
from src.reward.sparse import _compute_task_score


def eval_baseline(
    method_name: str,
    prompts: list[Prompt],
    llm: FrozenLLM,
    ratios: list[float],
) -> list[dict]:
    results = []
    for ratio in ratios:
        total_score = 0.0
        total_ratio = 0.0
        for prompt in prompts:
            if method_name == "random_drop":
                compressed = random_drop(prompt, ratio, seed=42)
            elif method_name == "selective_context":
                compressed = selective_context(prompt, ratio)
            elif method_name == "llmlingua2":
                compressed = llmlingua2_compress(prompt, ratio)
            else:
                raise ValueError(f"Unknown method: {method_name}")

            if compressed.token_ids.shape[0] == 0:
                total_score += 0.0
                continue

            llm_output = llm.generate(compressed.token_ids.unsqueeze(0))
            score = _compute_task_score(llm_output, prompt.metadata)
            total_score += score
            total_ratio += compressed.compression_ratio

        n = len(prompts)
        avg_score = total_score / n
        avg_ratio = total_ratio / n
        print(f"  {method_name:20s} ratio={ratio:.1f} -> task_score={avg_score:.4f} actual_ratio={avg_ratio:.4f}", file=sys.stderr)
        results.append({
            "method": method_name,
            "target_ratio": ratio,
            "actual_ratio": avg_ratio,
            "task_score": avg_score,
        })
    return results


def main():
    print("Loading LLM...", file=sys.stderr)
    llm_config = LlmConfig(
        model_name="meta-llama/Llama-3.1-8B",
        max_new_tokens=128,
        do_sample=False,
        quantize=True,
    )
    llm = FrozenLLM(llm_config, device="cuda")

    print("Loading val data...", file=sys.stderr)
    prompts = load_squad(split="validation", max_samples=50, max_length=1024)
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7]

    all_results = []

    # Also compute "no compression" baseline
    print("\n=== No Compression ===", file=sys.stderr)
    total = 0.0
    for p in prompts:
        out = llm.generate(p.token_ids.unsqueeze(0))
        total += _compute_task_score(out, p.metadata)
    print(f"  no_compression -> task_score={total/len(prompts):.4f}", file=sys.stderr)
    all_results.append({"method": "no_compression", "target_ratio": 1.0, "task_score": total / len(prompts)})

    for method in ["random_drop", "selective_context", "llmlingua2"]:
        print(f"\n=== {method} ===", file=sys.stderr)
        all_results.extend(eval_baseline(method, prompts, llm, ratios))

    out_path = Path("outputs/baselines/squad_llm_downstream.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
