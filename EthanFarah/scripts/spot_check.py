"""End-to-end spot check of the prompt compression pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import Config, parse_args
from src.data.tokenization import decode_compressed, get_tokenizer
from src.data.types import Episode, Prompt
from src.reward.metrics import compute_f1, compute_rouge, compute_task_score
from src.train import (
    Components,
    ScoreResult,
    _get_original_output,
    collect_episode,
    init_components,
    score_episode,
)
from src.tracking.checkpointing import CheckpointState, load_checkpoint


@dataclass
class SpotCheckArgs:
    """Script-specific CLI arguments."""

    checkpoint: str


def _parse_script_args() -> SpotCheckArgs:
    """Parse script-specific args; leave remaining for parse_args()."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return SpotCheckArgs(checkpoint=args.checkpoint)


def _load_policy(config: Config, checkpoint: str) -> Components:
    """Initialize components and load checkpoint weights."""
    c = init_components(config)
    step = load_checkpoint(
        CheckpointState(c.policy, c.algorithm, 0),
        Path(checkpoint),
    )
    print(f"Loaded checkpoint from step {step}", file=sys.stderr)
    return c


def _run_episode(c: Components, prompt: Prompt, device: str) -> tuple[Episode, ScoreResult]:
    """Collect a greedy episode and score it."""
    episode = collect_episode(c.env, c.policy, prompt, device=device, greedy=True)
    result = score_episode(episode, c.reward_fn, c.frozen_llm, c.kl_cache, sparse_only=True)
    return episode, result


def _format_answers(metadata: dict) -> str:
    """Format ground truth answers from prompt metadata."""
    answers = metadata.get("answer_texts", [])
    if not answers:
        return "(no answers)"
    return " | ".join(answers)


def _compute_faithfulness(result: ScoreResult) -> float | None:
    """Compute ROUGE-L between compressed and original LLM outputs."""
    if result.llm_output is None or result.original_llm_output is None:
        return None
    return compute_rouge(result.llm_output, result.original_llm_output)["rougeL"]


def _compute_f1_vs_ground_truth(llm_output: str | None, metadata: dict) -> float | None:
    """Compute max F1 of LLM output against all ground truth answers."""
    if llm_output is None:
        return None
    if not metadata.get("answer_texts"):
        return None
    return compute_task_score(llm_output, metadata)


def _compression_stats(episode: Episode) -> tuple[int, int]:
    """Return (tokens_kept, tokens_total) from episode actions."""
    total = episode.actions.shape[0]
    kept = int(episode.actions.sum().item())
    return kept, total


def _print_prompt_header(idx: int, prompt: Prompt) -> None:
    """Print the prompt header block."""
    print(f"\n{'=' * 70}")
    print(f"PROMPT {idx + 1}")
    print(f"{'=' * 70}")
    print(f"Original (first 300 chars):\n{prompt.text[:300]}")
    print(f"\nGround truth: {_format_answers(prompt.metadata)}")


def _print_original_output(result: ScoreResult, metadata: dict) -> None:
    """Print the original (uncompressed) LLM output and its F1."""
    original_out = result.original_llm_output or "(empty)"
    original_f1 = _compute_f1_vs_ground_truth(result.original_llm_output, metadata)
    f1_str = f"{original_f1:.4f}" if original_f1 is not None else "N/A"
    print(f"\nOriginal LLM output: {original_out}")
    print(f"Original F1 vs ground truth: {f1_str}")


def _print_compressed_output(
    episode: Episode, result: ScoreResult, metadata: dict, llm_model: str,
) -> None:
    """Print compressed text, LLM output, and compression stats."""
    tokenizer = get_tokenizer(llm_model)
    compressed_text = decode_compressed(episode.compressed, tokenizer)
    kept, total = _compression_stats(episode)

    print(f"\nCompressed text (first 300 chars):\n{compressed_text[:300]}")
    print(f"Compression ratio: {episode.compressed.compression_ratio:.4f}")
    print(f"Tokens kept/total: {kept}/{total}")

    compressed_out = result.llm_output or "(empty)"
    compressed_f1 = _compute_f1_vs_ground_truth(result.llm_output, metadata)
    f1_str = f"{compressed_f1:.4f}" if compressed_f1 is not None else "N/A"
    print(f"\nCompressed LLM output: {compressed_out}")
    print(f"Compressed F1 vs ground truth: {f1_str}")


def _print_reward_summary(episode: Episode, result: ScoreResult) -> None:
    """Print faithfulness and terminal reward."""
    faithfulness = _compute_faithfulness(result)
    faith_str = f"{faithfulness:.4f}" if faithfulness is not None else "N/A"
    print(f"\nFaithfulness ROUGE-L: {faith_str}")
    print(f"Terminal reward: {episode.terminal_reward:.4f}")


def _spot_check_one(
    idx: int, c: Components, prompt: Prompt, config: Config,
) -> None:
    """Run and print the full spot check for one prompt."""
    episode, result = _run_episode(c, prompt, config.device)

    _print_prompt_header(idx, prompt)
    _print_original_output(result, prompt.metadata)
    _print_compressed_output(episode, result, prompt.metadata, config.llm.model_name)
    _print_reward_summary(episode, result)


def main() -> None:
    """Load checkpoint, spot-check 5 validation prompts."""
    script_args = _parse_script_args()
    config = parse_args()
    torch.manual_seed(config.seed)

    c = _load_policy(config, script_args.checkpoint)
    c.policy.eval()

    prompts = c.val_prompts[:5]
    with torch.no_grad():
        for idx, prompt in enumerate(prompts):
            _spot_check_one(idx, c, prompt, config)


if __name__ == "__main__":
    main()
