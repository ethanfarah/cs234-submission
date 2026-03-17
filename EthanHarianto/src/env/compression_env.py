"""Gym-style compression environment for token-level keep/drop decisions."""

from __future__ import annotations

import math
import torch
from torch import Tensor

from src.data.types import CompressedPrompt, Episode, Prompt
from src.config import MinRatioEnforcementMode, MinRatioSelectionStrategy
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
        min_ratio: float = 0.2,
        min_ratio_mode: MinRatioEnforcementMode = MinRatioEnforcementMode.SOFT,
        min_ratio_soft_fraction: float = 0.25,
        min_ratio_selection_strategy: MinRatioSelectionStrategy = MinRatioSelectionStrategy.PREFIX,
    ) -> None:
        self._chunk_config = chunk_config
        self._target_ratio = target_ratio
        self._min_ratio = min_ratio
        self._min_ratio_mode = min_ratio_mode
        self._min_ratio_soft_fraction = min_ratio_soft_fraction
        self._min_ratio_selection_strategy = min_ratio_selection_strategy
        self._prompt: Prompt | None = None
        self._chunks: list[Observation] = []
        self._chunk_actions: list[Tensor] = []
        self._chunk_index: int = 0
        self._done: bool = False
        self._compressed: CompressedPrompt | None = None
        self._merged_actions: Tensor | None = None
        self._raw_merged_actions: Tensor | None = None
        self._forced_keep_mask: Tensor | None = None
        self._raw_merged_actions: Tensor | None = None
        self._forced_keep_mask: Tensor | None = None
        self._kept_count: int = 0
        self._seen_count: int = 0
        self._episode_info: dict[str, float] = {}

    @property
    def chunk_config(self) -> ChunkConfig:
        """Public access to the chunk configuration."""
        return self._chunk_config

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
        self._raw_merged_actions = None
        self._forced_keep_mask = None
        self._raw_merged_actions = None
        self._forced_keep_mask = None
        self._kept_count = 0
        self._seen_count = 0
        self._episode_info = {}
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

        self._chunk_actions.append(actions.cpu())
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
        sampled = self._raw_merged_actions if self._raw_merged_actions is not None else merged
        return Episode(
            prompt=self._prompt,
            actions=merged,
            sampled_actions=sampled,
            log_probs=torch.zeros_like(merged, dtype=torch.float),
            rewards=torch.zeros_like(merged, dtype=torch.float),
            values=None,
            compressed=self._compressed,
            terminal_reward=0.0,
            chunk_boundaries=[int(c.position_ids[0].item()) for c in self._chunks],
            chunk_size=self._chunk_config.chunk_size,
            overlap=self._chunk_config.overlap,
            target_compression_ratio=self._target_ratio,
            info=dict(self._episode_info),
            forced_keep_mask=self._forced_keep_mask,
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
        raw_merged = merged.clone()
        raw_merged = merged.clone()
        n_total = self._prompt.token_ids.shape[0]
        n_kept = int(merged.sum().item())
        min_kept = math.ceil(n_total * self._min_ratio)
        forced_keeps = 0
        forced_keep_mask = torch.zeros_like(merged, dtype=torch.bool)
        forced_keep_mask = torch.zeros_like(merged, dtype=torch.bool)
        if n_kept < min_kept and self._min_ratio_mode != MinRatioEnforcementMode.OFF:
            drop_idx = (merged == 0).nonzero(as_tuple=True)[0]
            deficit = min_kept - n_kept
            if self._min_ratio_mode == MinRatioEnforcementMode.HARD:
                n_to_flip = deficit
            else:
                n_to_flip = max(1, math.ceil(deficit * self._min_ratio_soft_fraction))
                n_to_flip = min(n_to_flip, deficit)
            if self._min_ratio_selection_strategy == MinRatioSelectionStrategy.RANDOM:
                flip_idx = drop_idx[
                    torch.randperm(drop_idx.shape[0], device=drop_idx.device)[:n_to_flip]
                ]
            else:
                # Deterministic prefix keeps reduce seed sensitivity under hard constraints.
                flip_idx = drop_idx[:n_to_flip]
            merged = merged.clone()
            merged[flip_idx] = 1
            forced_keep_mask[flip_idx] = True
            forced_keeps = int(n_to_flip)
        self._merged_actions = merged
        self._raw_merged_actions = raw_merged
        self._forced_keep_mask = forced_keep_mask
        self._raw_merged_actions = raw_merged
        self._forced_keep_mask = forced_keep_mask
        kept_ids = self._prompt.token_ids[merged.bool()]
        ratio = kept_ids.shape[0] / n_total
        self._episode_info = {
            "compression_ratio": ratio,
            "forced_keep_count": float(forced_keeps),
            "forced_keep_fraction": float(forced_keeps / n_total),
            "min_kept_target": float(min_kept),
        }

        self._compressed = CompressedPrompt(
            token_ids=kept_ids, keep_mask=merged, compression_ratio=ratio,
        )

        self._done = True
        dummy = self._make_dummy_obs(ratio)
        return dummy, 0.0, True, dict(self._episode_info)

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
