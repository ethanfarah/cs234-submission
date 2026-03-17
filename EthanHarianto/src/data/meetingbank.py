"""MeetingBank dataset loader."""

from __future__ import annotations

from datasets import load_dataset

from src.data.tokenization import DEFAULT_MODEL_NAME, get_tokenizer, tokenize_prompt
from src.data.types import Prompt


def _format_meetingbank_example(example: dict) -> tuple[str, dict]:
    """Format a single MeetingBank example into text and metadata.

    Field names ``source`` and ``reference`` match lytang/MeetingBank-transcript.
    """
    text = f"Summarize the following meeting transcript:\n\n{example['source']}"
    metadata = {"reference_summary": example["reference"]}
    return text, metadata


def load_meetingbank(
    split: str,
    max_samples: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    max_length: int = 1024,
) -> list[Prompt]:
    """Load MeetingBank and convert to Prompt objects."""
    ds = load_dataset("lytang/MeetingBank-transcript", split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    tokenizer = get_tokenizer(model_name)
    prompts: list[Prompt] = []
    for example in ds:
        text, metadata = _format_meetingbank_example(example)
        prompt = tokenize_prompt(text, tokenizer, max_length=max_length, metadata=metadata)
        prompts.append(prompt)
    return prompts
