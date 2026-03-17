"""Frozen LLM wrapper for generation and logit extraction."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor

from transformers import AutoModelForCausalLM

from src.config import LlmConfig
from src.data.tokenization import get_tokenizer


class FrozenLLM:
    """Frozen LLM wrapper with optional 4-bit quantization.

    Provides generation and logit extraction without gradient tracking.
    Quantization requires CUDA; CPU mode uses full-precision weights.
    """

    def __init__(self, config: LlmConfig, device: str = "cuda") -> None:
        if config.quantize and not torch.cuda.is_available():
            raise RuntimeError(
                "4-bit quantization requires CUDA, but no GPU is available. "
                "Set quantize=False for CPU inference."
            )
        self.config = config
        self.device = device
        self.tokenizer = get_tokenizer(config.model_name)
        self.model = self._load_model()
        self.model.eval()
        self.model.requires_grad_(False)

    def _load_model(self) -> AutoModelForCausalLM:
        if self.config.quantize:
            return self._load_quantized()
        warnings.warn(
            f"Loading {self.config.model_name} in FP32 without quantization. "
            "Large models (7B+) require ~32GB RAM. Set quantize=True for production.",
            UserWarning,
            stacklevel=3,
        )
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
        ).to(self.device)

    def _load_quantized(self) -> AutoModelForCausalLM:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int | None = None) -> str:
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be 2-D with batch size 1, got shape {input_ids.shape}"
            )
        if input_ids.shape[1] == 0:
            raise ValueError(
                "Cannot generate from empty prompt (0 tokens). "
                "Policy dropped all tokens during compression."
            )
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        prompt_len = input_ids.shape[1]
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        gen_kwargs = self._build_gen_kwargs(max_new_tokens)
        gen_kwargs["attention_mask"] = attention_mask
        gen_kwargs["pad_token_id"] = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )
        output_ids = self.model.generate(input_ids, **gen_kwargs)
        new_tokens = output_ids[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _build_gen_kwargs(self, max_new_tokens: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        kwargs["do_sample"] = self.config.do_sample
        if self.config.do_sample:
            kwargs["temperature"] = self.config.temperature
            kwargs["top_p"] = self.config.top_p
        return kwargs

    @torch.no_grad()
    def get_logits(self, input_ids: Tensor) -> Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-D, got shape {input_ids.shape}"
            )
        if input_ids.shape[1] == 0:
            raise ValueError(
                "Cannot compute logits for empty prompt (0 tokens). "
                "Policy dropped all tokens during compression."
            )
        input_ids = input_ids.to(self.model.device)
        logits = self.model(input_ids).logits
        if logits is None:
            raise RuntimeError(
                f"Model {self.config.model_name} returned None logits. "
                "Ensure the model has a language modeling head."
            )
        return logits.detach().cpu()
