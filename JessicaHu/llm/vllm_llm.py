from __future__ import annotations

from torch import Tensor

from src.config import LlmConfig
from src.data.tokenization import get_tokenizer


class VLLMFrozenLLM:
    def __init__(self, config: LlmConfig, device: str = "cuda") -> None:
        from vllm import LLM, SamplingParams

        self.config = config
        self.tokenizer = get_tokenizer(config.model_name)

        self.llm = LLM(
            model=config.model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.85,
        )
        temperature = config.temperature if config.do_sample else 0.0
        top_p = config.top_p if config.do_sample else 1.0
        self._sampling_params = SamplingParams(
            max_tokens=config.max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def generate(self, input_ids: Tensor, max_new_tokens: int | None = None) -> str:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        params = self._sampling_params
        if max_new_tokens is not None:
            params = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=self._sampling_params.temperature,
                top_p=self._sampling_params.top_p,
            )

        prompt = TokensPrompt(prompt_token_ids=input_ids[0].tolist())
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text
