"""Shared tokenizer utilities."""

from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerFast

from src.data.token_alignment import TokenAligner
from src.data.types import CompressedPrompt, Prompt

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.1-8B"


@lru_cache(maxsize=4)
def get_tokenizer(model_name: str) -> PreTrainedTokenizerFast:
    """Return a cached tokenizer for the given model.

    Callers must pass model_name explicitly — a default arg creates
    two lru_cache keys (no-arg call vs explicit-default call).
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError as e:
        raise OSError(
            f"Failed to load tokenizer '{model_name}'. "
            "If this is a gated model, set HF_TOKEN in your .env file."
        ) from e
    if tokenizer.pad_token is None:
        if tokenizer.eos_token_id != tokenizer.bos_token_id and tokenizer.bos_token_id is not None:
            tokenizer.pad_token = tokenizer.bos_token
        else:
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


def tokenize_prompt_dual(
    text: str,
    aligner: TokenAligner,
    max_length: int = 1024,
    metadata: dict | None = None,
) -> Prompt:
    """Tokenize with both policy and LLM tokenizers, returning aligned Prompt."""
    alignment = aligner.align(text, max_policy_tokens=max_length, max_llm_tokens=max_length)
    return Prompt(
        token_ids=alignment.policy_ids,
        attention_mask=alignment.policy_attention_mask,
        text=text,
        metadata=metadata or {},
        llm_token_ids=alignment.llm_ids,
        llm_attention_mask=alignment.llm_attention_mask,
        alignment=alignment,
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
