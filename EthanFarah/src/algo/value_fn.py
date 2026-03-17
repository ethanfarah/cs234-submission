"""Value functions for MCTS leaf evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from src.algo.mcts_types import MCTSState

if TYPE_CHECKING:
    from src.env.spaces import Observation
    from src.policy.distilroberta import DistilRoBERTaPolicy

_INITIAL_FAITHFULNESS_EMA = 0.3  # conservative prior: assume 30% faithfulness
_ASSUMED_UNSEEN_KEEP_RATE = 0.5  # assume 50% of unseen tokens will be kept


class HeuristicValue:
    """Estimates terminal reward from partial compression state using faithfulness EMA."""

    def __init__(self, ema_decay: float = 0.95) -> None:
        self._decay = ema_decay
        self._ema = _INITIAL_FAITHFULNESS_EMA

    def estimate(self, state: MCTSState) -> float:
        """Estimate reward from partial state.

        Terminal: faithfulness_ema * (1 - actual_ratio).
        Non-terminal: faithfulness_ema * (1 - estimated_ratio), where unseen
        tokens are assumed to be kept at _ASSUMED_UNSEEN_KEEP_RATE.
        """
        if state.is_terminal:
            ratio = state.kept_count() / max(state.total_tokens, 1)
            return self._ema * (1.0 - ratio)
        kept = state.kept_count()
        remaining = state.total_tokens - state.seen_count()
        estimated_kept = kept + remaining * _ASSUMED_UNSEEN_KEEP_RATE
        estimated_ratio = min(estimated_kept / max(state.total_tokens, 1), 1.0)
        return self._ema * (1.0 - estimated_ratio)

    def update(self, faithfulness: float) -> None:
        """Update EMA with observed faithfulness. Requires faithfulness in [0, 1]."""
        if not 0.0 <= faithfulness <= 1.0:
            raise ValueError(f"faithfulness must be in [0, 1], got {faithfulness}")
        self._ema = self._decay * self._ema + (1.0 - self._decay) * faithfulness

    @property
    def ema(self) -> float:
        return self._ema

    def state_dict(self) -> dict:
        return {"ema": self._ema}

    def load_state_dict(self, state: dict) -> None:
        self._ema = state["ema"]


class LearnedValue(HeuristicValue):
    """Learned value head blended with heuristic EMA for MCTS leaf evaluation.

    Uses the policy's value head to estimate leaf values from encoder features,
    blended with the heuristic EMA estimate for stability.
    """

    def __init__(
        self,
        policy: DistilRoBERTaPolicy,
        chunks: list[Observation],
        device: str = "cpu",
        ema_decay: float = 0.95,
        blend: float = 0.7,
        warmup_episodes: int = 50,
    ) -> None:
        super().__init__(ema_decay)
        self._policy = policy
        self._chunks = chunks
        self._device = device
        self._blend = blend
        self._warmup = warmup_episodes
        self._episode = 0

    def set_episode(self, episode: int) -> None:
        self._episode = episode

    def estimate(self, state: MCTSState) -> float:
        """Blend learned value with heuristic, annealing from 0 to blend over warmup."""
        heuristic = super().estimate(state)

        if state.is_terminal or state.chunk_index >= len(self._chunks):
            return heuristic

        effective_blend = min(
            self._blend, self._blend * (self._episode / max(self._warmup, 1))
        )
        if effective_blend < 0.01:
            return heuristic

        obs = self._chunks[state.chunk_index]
        obs_device = _obs_to_device(obs, self._device)
        with torch.no_grad():
            _, value = self._policy.forward_with_value(obs_device)
        learned = value.item()

        return effective_blend * learned + (1.0 - effective_blend) * heuristic

    def store_value_targets(self, terminal_reward: float) -> None:
        """Called after an episode to record (state, reward) pairs for value training.

        Value targets are stored on the LearnedValue instance and consumed
        by MCTSAlgorithm.update() for the value loss.
        """
        if not hasattr(self, "_targets"):
            self._targets: list[float] = []
        self._targets.append(terminal_reward)

    def pop_targets(self) -> list[float]:
        targets = getattr(self, "_targets", [])
        self._targets = []
        return targets


def _obs_to_device(obs: Observation, device: str) -> Observation:
    from src.env.spaces import Observation as Obs
    return Obs(
        token_ids=obs.token_ids.to(device),
        attention_mask=obs.attention_mask.to(device),
        position_ids=obs.position_ids.to(device),
        compression_ratio_so_far=obs.compression_ratio_so_far,
        target_compression_ratio=obs.target_compression_ratio,
        chunk_index=obs.chunk_index,
        total_chunks=obs.total_chunks,
    )
