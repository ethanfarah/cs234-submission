"""Gym-style compression environment for token-level keep/drop decisions."""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerFast

from src.data.types import CompressedPrompt, Episode, Prompt
from src.env.chunking import ChunkConfig, compute_chunks, merge_chunk_actions
from src.env.spaces import Observation


class CompressionEnv:
    """Episode-based environment that streams chunks to the policy.

    Each episode processes one prompt: reset() returns the first chunk
    observation, then step() is called repeatedly until all chunks are
    processed. After done=True, call get_episode() for the full Episode.
    """

    def __init__(
        self,
        chunk_config: ChunkConfig,
        target_ratio: float = 0.5,
        policy_tokenizer: PreTrainedTokenizerFast | None = None,
        llm_tokenizer: PreTrainedTokenizerFast | None = None,
    ) -> None:
        self._chunk_config = chunk_config
        self._target_ratio = target_ratio
        self._policy_tokenizer = policy_tokenizer
        self._llm_tokenizer = llm_tokenizer or policy_tokenizer
        self._prompt: Prompt | None = None
        self._chunks: list[Observation] = []
        self._chunk_actions: list[Tensor] = []
        self._chunk_index: int = 0
        self._done: bool = False
        self._compressed: CompressedPrompt | None = None
        self._merged_actions: Tensor | None = None
        self._kept_count: int = 0
        self._seen_count: int = 0

    @property
    def chunks(self) -> list[Observation]:
        """Precomputed chunk observations from the last reset() call."""
        return self._chunks

    @property
    def chunk_config(self) -> ChunkConfig:
        return self._chunk_config

    @property
    def target_ratio(self) -> float:
        return self._target_ratio

    @target_ratio.setter
    def target_ratio(self, value: float) -> None:
        self._target_ratio = value

    def reset(self, prompt: Prompt) -> Observation:
        """Start a new episode for the given prompt."""
        self._prompt = prompt
        self._chunks = compute_chunks(
            prompt, self._chunk_config, self._target_ratio,
        )
        if not self._chunks:
            raise ValueError("Prompt has no tokens")
        self._chunk_actions = []
        self._chunk_index = 0
        self._done = False
        self._compressed = None
        self._merged_actions = None
        self._kept_count = 0
        self._seen_count = 0
        return self._chunks[0]

    def step(self, actions: Tensor) -> tuple[Observation, float, bool, dict]:
        """Process one chunk's keep/drop actions.

        Returns (next_observation, reward, done, info). Reward is always 0.0;
        terminal reward is computed externally by the caller via _score_episode.
        On the terminal step, next_observation is a dummy (empty tensors).
        """
        if self._prompt is None:
            raise RuntimeError("Must call reset() before step()")
        if self._done:
            raise RuntimeError("Episode is already done")

        expected = self._chunks[self._chunk_index].token_ids.shape[0]
        if actions.ndim != 1 or actions.shape[0] != expected:
            raise ValueError(
                f"Expected 1-D actions of length {expected}, "
                f"got shape {actions.shape}"
            )

        self._chunk_actions.append(actions)
        self._chunk_index += 1

        if self._chunk_index < len(self._chunks):
            return self._advance_to_next_chunk()
        return self._finish_episode()

    def get_episode(self) -> Episode:
        """Return the completed Episode after done=True.

        log_probs, rewards, and terminal_reward are placeholder zeros —
        caller must populate them before computing policy gradients.
        """
        if self._prompt is None:
            raise RuntimeError("Must call reset() before get_episode()")
        if not self._done:
            raise RuntimeError("Episode is not done yet")

        if self._merged_actions is None:
            raise RuntimeError("No merged actions after done=True (internal error)")
        merged = self._merged_actions
        return Episode(
            prompt=self._prompt,
            actions=merged,
            log_probs=torch.zeros_like(merged, dtype=torch.float),
            rewards=torch.zeros_like(merged, dtype=torch.float),
            compressed=self._compressed,
            terminal_reward=0.0,
            chunk_boundaries=[int(c.position_ids[0].item()) for c in self._chunks],
            chunk_size=self._chunk_config.chunk_size,
            overlap=self._chunk_config.overlap,
            target_compression_ratio=self._target_ratio,
        )

    def _advance_to_next_chunk(self) -> tuple[Observation, float, bool, dict]:
        prev_actions = self._chunk_actions[-1]
        skip = self._chunk_config.overlap if len(self._chunk_actions) > 1 else 0
        new_unique = prev_actions[skip:]
        self._kept_count += int(new_unique.sum().item())
        self._seen_count += new_unique.shape[0]
        ratio_so_far = self._kept_count / self._seen_count

        next_obs = self._chunks[self._chunk_index]
        next_obs.compression_ratio_so_far = ratio_so_far
        return next_obs, 0.0, False, {}

    def _finish_episode(self) -> tuple[Observation, float, bool, dict]:
        merged = merge_chunk_actions(self._chunk_actions, self._chunk_config)
        self._merged_actions = merged

        if self._policy_tokenizer is not None:
            kept_ids, ratio = self._text_bridge(merged)
        else:
            kept_ids = self._prompt.token_ids[merged.bool()]
            ratio = kept_ids.shape[0] / self._prompt.token_ids.shape[0]

        self._compressed = CompressedPrompt(
            token_ids=kept_ids, keep_mask=merged, compression_ratio=ratio,
        )
        self._done = True
        dummy = self._make_dummy_obs(ratio)
        return dummy, 0.0, True, {"compression_ratio": ratio}

    def _text_bridge(self, merged: Tensor) -> tuple[Tensor, float]:
        """Decode kept policy tokens to text, re-tokenize with LLM tokenizer."""
        kept_policy_ids = self._prompt.token_ids[merged.bool()]
        decoded_text = self._policy_tokenizer.decode(
            kept_policy_ids, skip_special_tokens=True,
        )
        retokenized = self._llm_tokenizer(
            decoded_text, return_tensors="pt", add_special_tokens=False,
        )
        kept_ids = retokenized["input_ids"].squeeze(0)
        kept_ids = kept_ids.to(self._prompt.token_ids.device)
        original_len = self._original_llm_len()
        if original_len == 0:
            return kept_ids, 0.0
        ratio = min(kept_ids.shape[0] / original_len, 1.0)
        return kept_ids, ratio

    def _original_llm_len(self) -> int:
        """Return LLM-space token count for the original prompt."""
        if self._prompt.llm_token_ids is not None:
            return self._prompt.llm_token_ids.shape[0]
        original_llm = self._llm_tokenizer(
            self._prompt.text, return_tensors="pt", add_special_tokens=False,
        )
        return original_llm["input_ids"].shape[1]

    def _make_dummy_obs(self, ratio: float) -> Observation:
        return Observation(
            token_ids=torch.tensor([], dtype=torch.long),
            attention_mask=torch.tensor([], dtype=torch.long),
            position_ids=torch.tensor([], dtype=torch.long),
            compression_ratio_so_far=ratio,
            target_compression_ratio=self._target_ratio,
            chunk_index=self._chunk_index,
            total_chunks=len(self._chunks),
        )
