"""Train the learned reward model on SQuAD with F1 as the quality signal.

Uses FrozenLLM to evaluate compressed prompts: for each (prompt, ratio) pair,
compress via random-drop, run the LLM on the compressed tokens, then compute
F1(llm_output, ground_truth_answer). This produces labels that measure actual
downstream quality rather than text-overlap between compressed text and answers.

Run from the project root: python scripts/train_reward.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import DistilBertTokenizerFast, PreTrainedTokenizerFast

from src.baselines.random_drop import random_drop
from src.config import LlmConfig
from src.data.squad import load_squad
from src.data.tokenization import DEFAULT_MODEL_NAME, get_tokenizer
from src.data.types import CompressedPrompt, Prompt
from src.llm.frozen_llm import FrozenLLM
from src.reward.learned import LearnedRewardModel
from src.reward.metrics import compute_f1

BATCH_SIZE = 16
LR = 3e-4
NUM_EPOCHS = 10
DEFAULT_RATIOS = [0.3, 0.5, 0.7, 0.9]
MAX_SAMPLES = 500
DISTILBERT_NAME = "distilbert-base-uncased"
LOG_INTERVAL = 50
OUT_DIR = Path("outputs/reward_model")  # Must run from project root.


def _evaluate_compression(
    compressed: CompressedPrompt,
    reference: str,
    frozen_llm: FrozenLLM,
) -> tuple[float, str]:
    """Run LLM on compressed tokens and compute F1 against reference answer."""
    if compressed.token_ids.numel() == 0:
        return 0.0, ""
    input_ids = compressed.token_ids.unsqueeze(0).to(frozen_llm.device)
    llm_output = frozen_llm.generate(input_ids)
    return compute_f1(llm_output, reference), llm_output


def _process_pair(
    prompt: Prompt,
    ratio: float,
    frozen_llm: FrozenLLM,
    db_tok: DistilBertTokenizerFast,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compress a single prompt at a given ratio and return (orig_ids, comp_ids, score)."""
    ref = prompt.metadata.get("answer_texts", [""])[0]
    compressed = random_drop(prompt, keep_ratio=ratio)
    score, _ = _evaluate_compression(compressed, ref, frozen_llm)
    decoded = frozen_llm.tokenizer.decode(compressed.token_ids.cpu(), skip_special_tokens=True)
    enc_orig = db_tok(prompt.text, truncation=True, max_length=512, return_tensors="pt")
    enc_comp = db_tok(decoded, truncation=True, max_length=512, return_tensors="pt")
    return enc_orig["input_ids"].squeeze(0), enc_comp["input_ids"].squeeze(0), score


def _collect_pairs_with_llm(
    prompts: list[Prompt],
    ratios: list[float],
    frozen_llm: FrozenLLM,
    db_tok: DistilBertTokenizerFast,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float]]:
    """Build (orig_ids, comp_ids, f1_score) triples with LLM-evaluated labels."""
    orig_list: list[torch.Tensor] = []
    comp_list: list[torch.Tensor] = []
    score_list: list[float] = []
    total = len(prompts) * len(ratios)
    for i, prompt in enumerate(prompts):
        for j, ratio in enumerate(ratios):
            orig, comp, score = _process_pair(prompt, ratio, frozen_llm, db_tok)
            orig_list.append(orig)
            comp_list.append(comp)
            score_list.append(score)
            idx = i * len(ratios) + j + 1
            if idx % LOG_INTERVAL == 0:
                print(f"  data gen: {idx}/{total}", file=sys.stderr)
    return orig_list, comp_list, score_list


def _collect_pairs(
    prompts: list[Prompt],
    ratios: list[float],
    llama_tok: PreTrainedTokenizerFast,
    db_tok: DistilBertTokenizerFast,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float]]:
    """Build (orig_ids, comp_ids, f1_score) triples using text-overlap F1 (legacy)."""
    orig_list, comp_list, score_list = [], [], []
    for prompt in prompts:
        ref = prompt.metadata.get("answer_texts", [""])[0]
        for ratio in ratios:
            compressed = random_drop(prompt, keep_ratio=ratio)
            decoded = llama_tok.decode(compressed.token_ids.cpu(), skip_special_tokens=True)
            score = compute_f1(decoded, ref)
            enc_orig = db_tok(prompt.text, truncation=True, max_length=512, return_tensors="pt")
            enc_comp = db_tok(decoded, truncation=True, max_length=512, return_tensors="pt")
            orig_list.append(enc_orig["input_ids"].squeeze(0))
            comp_list.append(enc_comp["input_ids"].squeeze(0))
            score_list.append(score)
    return orig_list, comp_list, score_list


def _pad_and_package(
    orig_list: list[torch.Tensor],
    comp_list: list[torch.Tensor],
    score_list: list[float],
) -> TensorDataset:
    """Pad sequences and build a TensorDataset."""
    orig_padded = nn.utils.rnn.pad_sequence(orig_list, batch_first=True)
    comp_padded = nn.utils.rnn.pad_sequence(comp_list, batch_first=True)
    return TensorDataset(
        orig_padded, comp_padded,
        (orig_padded != 0).long(), (comp_padded != 0).long(),
        torch.tensor(score_list, dtype=torch.float32),
    )


def build_dataset(
    max_samples: int = MAX_SAMPLES,
    max_length: int = 1024,
    ratios: list[float] | None = None,
    frozen_llm: FrozenLLM | None = None,
) -> TensorDataset:
    if ratios is None:
        ratios = DEFAULT_RATIOS
    all_prompts = load_squad(split="train", max_samples=max_samples, max_length=max_length)
    prompts = [p for p in all_prompts if p.metadata.get("is_answerable")]
    db_tok = DistilBertTokenizerFast.from_pretrained(DISTILBERT_NAME)
    if frozen_llm is not None:
        orig_list, comp_list, score_list = _collect_pairs_with_llm(
            prompts, ratios, frozen_llm, db_tok,
        )
    else:
        llama_tok = get_tokenizer(DEFAULT_MODEL_NAME)
        orig_list, comp_list, score_list = _collect_pairs(
            prompts, ratios, llama_tok, db_tok,
        )
    return _pad_and_package(orig_list, comp_list, score_list)


def train_epoch(
    model: LearnedRewardModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    device = next(model.parameters()).device
    total = 0.0
    for orig, comp, orig_mask, comp_mask, targets in loader:
        orig, comp, orig_mask, comp_mask, targets = (
            t.to(device) for t in (orig, comp, orig_mask, comp_mask, targets)
        )
        optimizer.zero_grad()
        preds = model(orig, comp, orig_mask, comp_mask)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(len(loader), 1)


def val_epoch(
    model: LearnedRewardModel,
    loader: DataLoader,
    loss_fn: nn.Module,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    total = 0.0
    with torch.no_grad():
        for orig, comp, orig_mask, comp_mask, targets in loader:
            orig, comp, orig_mask, comp_mask, targets = (
                t.to(device) for t in (orig, comp, orig_mask, comp_mask, targets)
            )
            total += loss_fn(model(orig, comp, orig_mask, comp_mask), targets).item()
    return total / max(len(loader), 1)


def _prepare_loaders(dataset: TensorDataset) -> tuple[DataLoader, DataLoader]:
    val_size = max(1, len(dataset) // 10)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
    return DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True), DataLoader(val_ds, batch_size=BATCH_SIZE)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: FrozenLLM uses 4-bit quantization")
    device = torch.device("cuda")
    print("Loading FrozenLLM for data generation...", file=sys.stderr)
    frozen_llm = FrozenLLM(LlmConfig(quantize=True), device=str(device))
    dataset = build_dataset(frozen_llm=frozen_llm)
    del frozen_llm
    torch.cuda.empty_cache()
    print("FrozenLLM freed, starting reward model training...", file=sys.stderr)
    train_loader, val_loader = _prepare_loaders(dataset)
    model = LearnedRewardModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss = val_epoch(model, val_loader, loss_fn)
        print(f"epoch {epoch+1}/{NUM_EPOCHS} | train={train_loss:.4f} | val={val_loss:.4f}", file=sys.stderr)
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), OUT_DIR / "best.pt")


if __name__ == "__main__":
    main()
