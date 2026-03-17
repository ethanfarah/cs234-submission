"""Abstract policy interface for prompt compression."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.env.spaces import Observation


class Policy(ABC, nn.Module):
    """Base class for all compression policies.

    Outputs 3D (batch, seq_len, 2) categorical keep/drop logits.
    """

    head: nn.Module

    @abstractmethod
    def forward(self, obs: Observation) -> torch.Tensor:
        """Compute per-token logits."""
        ...

    def act(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return (actions, log_probs)."""
        output = self.forward(obs)
        return self._act_categorical(output)

    def act_greedy(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        """Take greedy (argmax) actions and return (actions, log_probs)."""
        output = self.forward(obs)
        return self._act_categorical_greedy(output)

    def _act_categorical(
        self, logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist.log_prob(actions)

    def _act_categorical_greedy(
        self, logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(dim=-1)
        return actions, dist.log_prob(actions)

    def evaluate_actions(
        self, obs: Observation, actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log_probs, entropy, and score_std for given actions."""
        output = self.forward(obs)
        return self._eval_categorical(output, actions)

    def _eval_categorical(
        self, logits: torch.Tensor, actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=logits)
        keep_probs = F.softmax(logits, dim=-1)[:, :, 1]  # (batch, seq_len)
        score_std = keep_probs.std(dim=-1)  # (batch,) — scalar per batch element
        return dist.log_prob(actions), dist.entropy(), score_std
