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
    def generate(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate text from input token IDs.

        attention_mask: optional (1, seq_len) mask. 1=attend, 0=masked.
        When None, defaults to all-ones (attend to everything).
        """
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be 2-D with batch size 1, got shape {input_ids.shape}"
            )
        if input_ids.shape[1] == 0:
            raise ValueError(
                "Cannot generate from empty prompt (0 tokens). "
                "Policy dropped all tokens during compression."
            )
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} != "
                f"input_ids shape {input_ids.shape}"
            )
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        prompt_len = input_ids.shape[1]
        input_ids = input_ids.to(self.model.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(self.model.device)
        gen_kwargs = self._build_gen_kwargs(max_new_tokens)
        output_ids = self.model.generate(
            input_ids, attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id, **gen_kwargs,
        )
        new_tokens = output_ids[0, prompt_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _build_gen_kwargs(self, max_new_tokens: int) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
        if self.config.do_sample:
            kwargs["do_sample"] = True
            kwargs["temperature"] = self.config.temperature
            kwargs["top_p"] = self.config.top_p
        else:
            kwargs["do_sample"] = False
        return kwargs

    @torch.no_grad()
    def teacher_forced_ce(
        self,
        prompt_ids: Tensor,
        target_ids: Tensor,
    ) -> float:
        """Cross-entropy of target_ids under the model conditioned on prompt_ids.

        Concatenates prompt + target, runs a single forward pass, and computes
        CE loss only on the target positions (teacher-forced). Returns a scalar.
        """
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if target_ids.dim() == 1:
            target_ids = target_ids.unsqueeze(0)
        if prompt_ids.shape[-1] == 0:
            raise ValueError("prompt_ids must contain at least one token")
        if target_ids.shape[-1] == 0:
            raise ValueError("target_ids must contain at least one token")
        input_ids = torch.cat([prompt_ids, target_ids], dim=1).to(self.model.device)
        logits = self.model(input_ids).logits  # (1, prompt+target, vocab)
        prompt_len = prompt_ids.shape[1]
        # Logits at position i predict token at position i+1
        target_logits = logits[0, prompt_len - 1 : -1]  # (target_len, vocab)
        targets = target_ids[0].to(self.model.device)    # (target_len,)
        ce = torch.nn.functional.cross_entropy(target_logits, targets)
        return ce.item()

    @torch.no_grad()
    def get_logits(
        self, input_ids: Tensor, attention_mask: Tensor | None = None,
    ) -> Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2-D, got shape {input_ids.shape}"
            )
        if input_ids.shape[1] == 0:
            raise ValueError(
                "Cannot compute logits for empty prompt (0 tokens). "
                "Policy dropped all tokens during compression."
            )
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape {attention_mask.shape} != "
                f"input_ids shape {input_ids.shape}"
            )
        input_ids = input_ids.to(self.model.device)
        kwargs: dict = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask.to(self.model.device)
        logits = self.model(input_ids, **kwargs).logits
        if logits is None:
            raise RuntimeError(
                f"Model {self.config.model_name} returned None logits. "
                "Ensure the model has a language modeling head."
            )
        return logits.detach().cpu()
