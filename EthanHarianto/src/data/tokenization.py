"""Shared tokenizer utilities."""

from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.data.types import CompressedPrompt, Prompt

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B"
# Public model used when DEFAULT_MODEL_NAME is gated (no HF_TOKEN).
FALLBACK_MODEL_NAME = "gpt2"


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    """Return a cached tokenizer for the given model.

    Callers must pass model_name explicitly — a default arg creates
    two lru_cache keys (no-arg call vs explicit-default call).
    When model_name is DEFAULT_MODEL_NAME and the repo is gated (no HF_TOKEN),
    falls back to FALLBACK_MODEL_NAME so baselines and data loaders can run.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError as e:
        if model_name == DEFAULT_MODEL_NAME:
            tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME)
        else:
            raise OSError(
                f"Failed to load tokenizer '{model_name}'. "
                "If this is a gated model, set HF_TOKEN in your .env file."
            ) from e
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_prompt(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 1024,
    metadata: dict | None = None,
) -> Prompt:
    """Tokenize raw text into a Prompt dataclass."""
    encoding = tokenizer(
        text,
        max_length=max_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    return Prompt(
        token_ids=encoding["input_ids"].squeeze(0),
        attention_mask=encoding["attention_mask"].squeeze(0),
        text=text,
        metadata=metadata or {},
    )


def decode_compressed(
    compressed: CompressedPrompt,
    tokenizer: PreTrainedTokenizerFast,
) -> str:
    """Decode compressed token IDs back to text.

    No need to apply keep_mask here — CompressedPrompt.token_ids already
    contains only the kept tokens (shape ``(compressed_len,)``).
    keep_mask is metadata recording which *original* positions survived.
    """
    return tokenizer.decode(compressed.token_ids, skip_special_tokens=True)
