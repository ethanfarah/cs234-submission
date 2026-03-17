"""AlphaZero-style supervised pre-training for the DistilRoBERTa policy head.

Generates per-token importance labels via marginal contribution analysis,
then trains the policy head with MSE loss to match those labels.

Usage:
    python scripts/pretrain_policy.py --mode generate --device cuda
    python scripts/pretrain_policy.py --mode pretrain --device cuda
    python scripts/pretrain_policy.py --mode both --device cuda
    python scripts/pretrain_policy.py --mode validate --device cuda
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ["PYTHONIOENCODING"] = "utf-8"

import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.config import Config, LlmConfig
from src.data.squad import load_squad
from src.env.spaces import Observation
from src.llm.frozen_llm import FrozenLLM
from src.policy.distilroberta import DistilRoBERTaPolicy
from src.reward.metrics import compute_rouge


def compute_k(seq_len: int, target_ratio: float) -> int:
    """Number of tokens to keep for a target compression ratio."""
    return max(1, math.ceil(seq_len * target_ratio))


def greedy_topk(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Deterministic top-k selection (for greedy baseline)."""
    _, indices = scores.topk(k, dim=-1)
    mask = torch.zeros_like(scores, dtype=torch.long)
    mask.scatter_(-1, indices, 1)
    return mask

LLM_MODEL = "meta-llama/Llama-3.2-1B"
POLICY_MODEL = "distilroberta-base"
DEFAULT_KEEP_RATE = 0.75
DEFAULT_NUM_MASKS = 20
DEFAULT_NUM_PROMPTS = 200
DEFAULT_EPOCHS = 10
DEFAULT_LR = 1e-3
DEFAULT_VAL_SAMPLES = 10
OUTPUT_DIR = Path("outputs/pretrain")


@dataclass
class SweepModels:
    """Bundle of models/tokenizers for importance data generation."""

    policy_tok: PreTrainedTokenizerFast
    llm_tok: PreTrainedTokenizerFast
    frozen_llm: FrozenLLM


@dataclass
class ImportanceSample:
    """Single prompt's importance data."""

    policy_token_ids: torch.Tensor
    attention_mask: torch.Tensor
    importance_scores: torch.Tensor


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Policy head pre-training")
    parser.add_argument("--mode", required=True, choices=["generate", "pretrain", "both", "validate"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-prompts", type=int, default=DEFAULT_NUM_PROMPTS)
    parser.add_argument("--num-masks", type=int, default=DEFAULT_NUM_MASKS)
    parser.add_argument("--keep-rate", type=float, default=DEFAULT_KEEP_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-samples", type=int, default=DEFAULT_VAL_SAMPLES)
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _make_llm_config(quantize: bool = True) -> LlmConfig:
    return LlmConfig(model_name=LLM_MODEL, quantize=quantize, max_new_tokens=64)


def _load_models(device: str, quantize: bool = True) -> SweepModels:
    """Load policy tokenizer, LLM tokenizer, and frozen LLM."""
    sys.stderr.write(f"Loading policy tokenizer ({POLICY_MODEL})...\n")
    policy_tok = AutoTokenizer.from_pretrained(POLICY_MODEL)
    sys.stderr.write(f"Loading LLM ({LLM_MODEL}, quantize={quantize})...\n")
    frozen_llm = FrozenLLM(_make_llm_config(quantize), device=device)
    return SweepModels(policy_tok=policy_tok, llm_tok=frozen_llm.tokenizer, frozen_llm=frozen_llm)


def _contiguous_block_mask(seq_len: int, keep_rate: float, block_start: int) -> torch.Tensor:
    """Create mask with a contiguous block of zeros (dropped tokens)."""
    mask = torch.ones(seq_len, dtype=torch.long)
    n_drop = seq_len - max(1, int(seq_len * keep_rate))
    clamped_start = max(0, min(block_start, seq_len - n_drop))
    mask[clamped_start:clamped_start + n_drop] = 0
    return mask


def _random_keep_mask(seq_len: int, keep_rate: float, rng: random.Random) -> torch.Tensor:
    """Random binary mask with given keep rate."""
    mask = torch.zeros(seq_len, dtype=torch.long)
    indices = list(range(seq_len))
    rng.shuffle(indices)
    n_keep = max(1, int(seq_len * keep_rate))
    for i in indices[:n_keep]:
        mask[i] = 1
    return mask


def _generate_masks(seq_len: int, keep_rate: float, n_masks: int, rng: random.Random) -> list[torch.Tensor]:
    """Generate n_masks masks: half contiguous block drops, half random."""
    masks: list[torch.Tensor] = []
    n_contiguous = n_masks // 2
    stride = max(1, seq_len // n_contiguous) if n_contiguous > 0 else 1
    for i in range(n_contiguous):
        start = (i * stride) % seq_len
        masks.append(_contiguous_block_mask(seq_len, keep_rate, start))
    for _ in range(n_masks - n_contiguous):
        masks.append(_random_keep_mask(seq_len, keep_rate, rng))
    return masks


def _generate_baseline(models: SweepModels, prompt_text: str) -> str:
    """Generate baseline output from full prompt."""
    ids = models.llm_tok(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return models.frozen_llm.generate(ids)


def _generate_compressed(models: SweepModels, policy_ids: torch.Tensor, mask: torch.Tensor) -> str | None:
    """Text bridge: keep masked tokens -> decode -> re-tokenize -> generate.

    Returns None if the mask selects only special tokens (empty decoded text).
    """
    kept_ids = policy_ids[mask.bool()]
    decoded_text = models.policy_tok.decode(kept_ids, skip_special_tokens=True)
    if not decoded_text.strip():
        return None
    retokenized = models.llm_tok(decoded_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    return models.frozen_llm.generate(retokenized)


def _compute_faithfulness(models: SweepModels, policy_ids: torch.Tensor, mask: torch.Tensor, baseline: str) -> float:
    """Compute ROUGE-L faithfulness for a single mask. Returns 0.0 on empty decode."""
    output = _generate_compressed(models, policy_ids, mask)
    if output is None:
        return 0.0
    return compute_rouge(output, baseline)["rougeL"]


def _compute_importance(masks: list[torch.Tensor], scores: list[float]) -> torch.Tensor:
    """Compute per-token importance via marginal contribution (vectorized).

    importance[i] = mean(scores where token_i kept) - mean(scores where token_i dropped)
    Normalized to [0, 1]. Tokens always kept/dropped get 0.5.
    """
    kept = torch.stack(masks).float()  # (n_masks, seq_len)
    score_vec = torch.tensor(scores, dtype=torch.float)  # (n_masks,)
    dropped = 1.0 - kept
    kept_count = kept.sum(0)
    dropped_count = dropped.sum(0)
    mean_kept = (kept * score_vec[:, None]).sum(0) / kept_count.clamp(min=1)
    mean_dropped = (dropped * score_vec[:, None]).sum(0) / dropped_count.clamp(min=1)
    importance = mean_kept - mean_dropped
    always_same = (kept_count == 0) | (dropped_count == 0)
    importance[always_same] = 0.5
    return _normalize_to_unit(importance)


def _normalize_to_unit(t: torch.Tensor) -> torch.Tensor:
    """Normalize tensor to [0, 1] range. Returns 0.5 if constant."""
    lo, hi = t.min().item(), t.max().item()
    if hi - lo < 1e-8:
        return torch.full_like(t, 0.5)
    return (t - lo) / (hi - lo)


def _process_single_prompt(
    models: SweepModels, prompt_text: str, args: argparse.Namespace, rng: random.Random,
) -> ImportanceSample:
    """Generate importance labels for a single prompt."""
    policy_ids = models.policy_tok(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    baseline = _generate_baseline(models, prompt_text)
    masks = _generate_masks(policy_ids.shape[0], args.keep_rate, args.num_masks, rng)
    scores = [_compute_faithfulness(models, policy_ids, m, baseline) for m in masks]
    importance = _compute_importance(masks, scores)
    return ImportanceSample(
        policy_token_ids=policy_ids,
        attention_mask=torch.ones_like(policy_ids),
        importance_scores=importance,
    )


def _run_generate(args: argparse.Namespace) -> None:
    """Generate importance data for all prompts, split into train/val."""
    models = _load_models(args.device, args.quantize)
    prompts = load_squad("validation", max_samples=args.num_prompts)
    rng = random.Random(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples: list[dict] = []
    for i, prompt in enumerate(prompts):
        sys.stderr.write(f"\rGenerating importance data: {i + 1}/{len(prompts)}")
        sys.stderr.flush()
        prompt_text = f"{prompt.text}\n\nAnswer:"
        sample = _process_single_prompt(models, prompt_text, args, rng)
        samples.append({
            "policy_token_ids": sample.policy_token_ids,
            "attention_mask": sample.attention_mask,
            "importance_scores": sample.importance_scores,
        })
    sys.stderr.write("\n")
    _save_train_val_split(samples, args.val_samples)


def _save_train_val_split(samples: list[dict], n_val: int) -> None:
    """Save samples split into train and val files."""
    n_val = min(n_val, len(samples))
    if n_val >= len(samples):
        raise ValueError(f"val_samples ({n_val}) >= total samples ({len(samples)}); no training data")
    train_samples = samples[:-n_val] if n_val > 0 else samples
    val_samples = samples[-n_val:] if n_val > 0 else []
    torch.save(train_samples, OUTPUT_DIR / "importance_data.pt")
    torch.save(val_samples, OUTPUT_DIR / "importance_data_val.pt")
    sys.stderr.write(f"Saved {len(train_samples)} train + {len(val_samples)} val samples\n")


def _make_obs(sample: dict, device: str, keep_rate: float = DEFAULT_KEEP_RATE) -> Observation:
    """Build an Observation from an importance sample dict."""
    token_ids = sample["policy_token_ids"]
    return Observation(
        token_ids=token_ids.to(device),
        attention_mask=sample["attention_mask"].to(device),
        position_ids=torch.arange(token_ids.shape[0], device=device),
        compression_ratio_so_far=1.0,
        target_compression_ratio=keep_rate,
        chunk_index=0,
        total_chunks=1,
    )


def _train_one_epoch(
    policy: DistilRoBERTaPolicy, optimizer: torch.optim.Optimizer, samples: list[dict], device: str,
) -> float:
    """Train head for one epoch, return mean loss."""
    total_loss = 0.0
    for sample in samples:
        obs = _make_obs(sample, device)
        scores = policy(obs).squeeze(0)
        target = sample["importance_scores"].to(device)
        loss = torch.nn.functional.mse_loss(scores, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(samples)


def _run_pretrain(args: argparse.Namespace) -> None:
    """Train policy head on importance data."""
    data_path = OUTPUT_DIR / "importance_data.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"No importance data at {data_path}. Run --mode generate first.")
    samples = torch.load(data_path, weights_only=False)
    sys.stderr.write(f"Loaded {len(samples)} training samples from {data_path}\n")
    config = Config(policy_type="distilroberta", policy_model_name=POLICY_MODEL, device=args.device)
    policy = DistilRoBERTaPolicy(config).to(args.device)
    policy.encoder.eval()
    optimizer = torch.optim.Adam(policy.head.parameters(), lr=args.lr)
    rng = random.Random(args.seed)
    for epoch in range(args.epochs):
        rng.shuffle(samples)
        mean_loss = _train_one_epoch(policy, optimizer, samples, args.device)
        sys.stderr.write(f"Epoch {epoch + 1}/{args.epochs} | MSE loss: {mean_loss:.6f}\n")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "policy_head.pt"
    torch.save({"head": policy.head.state_dict(), "metadata": {"epochs": args.epochs, "lr": args.lr}}, out_path)
    sys.stderr.write(f"Saved policy head checkpoint to {out_path}\n")


def _compute_contiguity(mask: torch.Tensor) -> float:
    """Fraction of kept tokens adjacent to another kept token."""
    kept_positions = mask.nonzero(as_tuple=True)[0]
    if len(kept_positions) <= 1:
        return 1.0
    n_adjacent_tokens = 0
    for i in range(len(kept_positions)):
        pos = kept_positions[i].item()
        has_left = (i > 0) and (kept_positions[i - 1].item() == pos - 1)
        has_right = (i < len(kept_positions) - 1) and (kept_positions[i + 1].item() == pos + 1)
        if has_left or has_right:
            n_adjacent_tokens += 1
    return n_adjacent_tokens / len(kept_positions)


def _run_validate(args: argparse.Namespace) -> None:
    """Validate pre-trained policy head on held-out prompts."""
    ckpt_path = OUTPUT_DIR / "policy_head.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. Run --mode pretrain first.")
    val_path = OUTPUT_DIR / "importance_data_val.pt"
    if not val_path.exists():
        raise FileNotFoundError(f"No held-out data at {val_path}. Run --mode generate first.")
    samples = torch.load(val_path, weights_only=False)
    config = Config(policy_type="distilroberta", policy_model_name=POLICY_MODEL, device=args.device)
    policy = DistilRoBERTaPolicy(config).to(args.device)
    ckpt = torch.load(ckpt_path, weights_only=True)
    policy.head.load_state_dict(ckpt["head"])
    policy.eval()
    _validate_on_samples(policy, samples, args.device)


def _validate_on_samples(policy: DistilRoBERTaPolicy, samples: list[dict], device: str) -> None:
    """Run validation metrics and print summary table."""
    if not samples:
        sys.stderr.write("No validation samples to evaluate.\n")
        return
    correlations: list[float] = []
    contiguities: list[float] = []
    for sample in samples:
        obs = _make_obs(sample, device)
        with torch.no_grad():
            scores = policy(obs).squeeze(0).cpu()
        k = compute_k(scores.shape[0], DEFAULT_KEEP_RATE)
        mask = greedy_topk(scores.unsqueeze(0), k).squeeze(0)
        rho, _ = spearmanr(scores.numpy(), sample["importance_scores"].numpy())
        correlations.append(rho)
        contiguities.append(_compute_contiguity(mask))
    _print_validation_table(correlations, contiguities)


def _print_validation_table(correlations: list[float], contiguities: list[float]) -> None:
    """Print validation summary to stderr."""
    mean_rho = sum(correlations) / len(correlations)
    mean_contig = sum(contiguities) / len(contiguities)
    lines = [
        "\n" + "=" * 50,
        "VALIDATION RESULTS (held-out)",
        "=" * 50,
        f"{'Metric':<25} | {'Value':>10}",
        "-" * 50,
        f"{'Spearman rho (mean)':<25} | {mean_rho:>10.4f}",
        f"{'Contiguity score (mean)':<25} | {mean_contig:>10.4f}",
        f"{'Num samples':<25} | {len(correlations):>10d}",
        "=" * 50,
        "",
    ]
    sys.stderr.write("\n".join(lines) + "\n")


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.mode == "generate":
        _run_generate(args)
    elif args.mode == "pretrain":
        _run_pretrain(args)
    elif args.mode == "both":
        _run_generate(args)
        _run_pretrain(args)
    elif args.mode == "validate":
        _run_validate(args)


if __name__ == "__main__":
    main()
