"""SQuAD 2.0 dataset loader."""

from __future__ import annotations

from datasets import load_dataset

from src.data.tokenization import DEFAULT_MODEL_NAME, get_tokenizer, tokenize_prompt
from src.data.types import Prompt


def _format_squad_example(example: dict) -> tuple[str, dict]:
    """Format a single SQuAD example into text and metadata."""
    text = f"Question: {example['question']}\n\nContext: {example['context']}"
    metadata = {
        "answer_texts": example["answers"]["text"],
        "answer_starts": example["answers"]["answer_start"],
        "is_answerable": len(example["answers"]["text"]) > 0,
    }
    return text, metadata


def load_squad(
    split: str,
    max_samples: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = 1024,
    answerable_only: bool = False,
) -> list[Prompt]:
    """Load SQuAD 2.0 and convert to Prompt objects."""
    ds = load_dataset("squad_v2", split=split)
    if answerable_only:
        ds = ds.filter(lambda ex: len(ex["answers"]["text"]) > 0)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    tokenizer = get_tokenizer(model_name)
    prompts: list[Prompt] = []
    for example in ds:
        text, metadata = _format_squad_example(example)
        prompt = tokenize_prompt(text, tokenizer, max_length=max_length, metadata=metadata)
        prompts.append(prompt)
    return prompts
