from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor

from transformers import AutoModelForCausalLM
from transformers import BitsAndBytesConfig


from src.config import LlmConfig
from src.data.tokenization import get_tokenizer


class FrozenLLM:
    def __init__(self, config: LlmConfig, device: str = "cuda") -> None:
        self.config = config
        self.device = device
        self.tokenizer = get_tokenizer(config.model_name)
        self.model = self._load_model()
        self.model.eval()
        self.model.requires_grad_(False)

    def _load_model(self) -> AutoModelForCausalLM:
        if self.config.quantize:
            return self._load_quantized()
        return AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float32,
        ).to(self.device)

    def _load_quantized(self) -> AutoModelForCausalLM:
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
        if self.config.do_sample:
            kwargs["do_sample"] = True
            kwargs["temperature"] = self.config.temperature
            kwargs["top_p"] = self.config.top_p
        return kwargs

    @torch.no_grad()
    def get_logits(self, input_ids: Tensor) -> Tensor:
        input_ids = input_ids.to(self.model.device)
        logits = self.model(input_ids).logits
        return logits.detach().cpu()
