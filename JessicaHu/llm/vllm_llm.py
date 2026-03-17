"""vLLM-based frozen LLM for fast batched inference.

Drop-in replacement for FrozenLLM.generate(). vLLM provides 2-5x faster
throughput than HuggingFace generate() via PagedAttention and continuous batching.

Limitation: get_logits() is not supported — vLLM does not expose full vocabulary
logits. Use FrozenLLM (HuggingFace) for KL-dense or hybrid reward.
"""

from __future__ import annotations

from torch import Tensor

from src.config import LlmConfig
from src.data.tokenization import get_tokenizer


class VLLMFrozenLLM:
    """vLLM wrapper matching FrozenLLM.generate() interface.

    Use with sparse reward only. Raises NotImplementedError on get_logits().
    Requires CUDA — vLLM does not support CPU inference.
    """

    def __init__(self, config: LlmConfig, device: str = "cuda") -> None:
        from vllm import LLM, SamplingParams

        if device == "cpu":
            raise RuntimeError("VLLMFrozenLLM requires CUDA; vLLM does not support CPU.")

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
        """Generate text from compressed token IDs via vLLM.

        Args:
            input_ids: (1, seq_len) token IDs — batch size must be 1.
            max_new_tokens: override config.max_new_tokens if set.

        Returns:
            Decoded generated text (new tokens only, no prompt).
        """
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"input_ids must be 2-D with batch size 1, got shape {input_ids.shape}"
            )
        if input_ids.shape[1] == 0:
            raise ValueError(
                "Cannot generate from empty prompt (0 tokens). "
                "Policy dropped all tokens during compression."
            )

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

    def get_logits(self, input_ids: Tensor) -> Tensor:
        raise NotImplementedError(
            "VLLMFrozenLLM does not support get_logits(). "
            "vLLM does not expose full vocabulary logits. "
            "Use FrozenLLM (HuggingFace) for KL-dense or hybrid reward."
        )
