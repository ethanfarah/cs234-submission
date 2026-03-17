#!/usr/bin/env python3
"""Fair LLMLingua-2 evaluation through the same LLM pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

import src.config  # noqa: F401
from src.baselines.llmlingua2 import llmlingua2_compress
from src.data.squad import load_squad
from src.llm.frozen_llm import FrozenLLM
from src.config import LlmConfig
from src.reward.sparse import _compute_task_score


def main():
    print("Loading val data...", file=sys.stderr)
    prompts = load_squad(split="validation", max_samples=50, max_length=1024)

    print("Pre-compressing with LLMLingua-2...", file=sys.stderr)
    ratios = [0.3, 0.4, 0.5, 0.6, 0.7]
    compressed_sets: dict[float, list] = {}
    for ratio in ratios:
        compressed_sets[ratio] = []
        for i, prompt in enumerate(prompts):
            c = llmlingua2_compress(prompt, ratio)
            compressed_sets[ratio].append(c)
            if (i + 1) % 10 == 0:
                print(f"  ratio={ratio} compressed {i+1}/{len(prompts)}", file=sys.stderr)

    # Free LLMLingua-2 model memory before loading LLM
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading LLM...", file=sys.stderr)
    llm_config = LlmConfig(
        model_name="meta-llama/Llama-3.1-8B",
        max_new_tokens=128,
        do_sample=False,
        quantize=True,
    )
    llm = FrozenLLM(llm_config, device="cuda")

    results = []
    for ratio in ratios:
        total_score = 0.0
        total_ratio = 0.0
        for prompt, compressed in zip(prompts, compressed_sets[ratio]):
            if compressed.token_ids.shape[0] == 0:
                continue
            llm_output = llm.generate(compressed.token_ids.unsqueeze(0))
            score = _compute_task_score(llm_output, prompt.metadata)
            total_score += score
            total_ratio += compressed.compression_ratio

        n = len(prompts)
        avg_score = total_score / n
        avg_ratio = total_ratio / n
        print(f"  llmlingua2 ratio={ratio:.1f} -> task_score={avg_score:.4f} actual_ratio={avg_ratio:.4f}", file=sys.stderr)
        results.append({
            "method": "llmlingua2",
            "target_ratio": ratio,
            "actual_ratio": avg_ratio,
            "task_score": avg_score,
        })

    # Load existing results and merge
    out_path = Path("outputs/baselines/squad_llm_downstream.json")
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    existing = [r for r in existing if r["method"] != "llmlingua2"]
    existing.extend(results)
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
