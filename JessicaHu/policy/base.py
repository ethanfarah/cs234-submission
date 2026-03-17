"""Abstract policy interface for prompt compression."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.env.spaces import Observation


class Policy(ABC, nn.Module):
    """Base class for all compression policies.

    A policy maps an observation (chunk of tokens + context) to
    per-token keep/drop logits.

    Subclasses must define ``self.head: nn.Linear`` — the final classification
    layer. PPO's value head hooks into this for feature capture.
    """

    head: nn.Linear

    @abstractmethod
    def forward(self, obs: Observation) -> torch.Tensor:
        """Compute keep/drop logits for each token.

        Args:
            obs: Current observation (chunk of tokens + metadata).

        Returns:
            Tensor of shape (batch, seq_len, 2) — logits for [drop, keep].
        """
        ...

    def act(self, obs: Observation) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample actions and return (actions, log_probs).

        Args:
            obs: Current observation.

        Returns:
            actions: (batch, seq_len) binary tensor (0=drop, 1=keep).
            log_probs: (batch, seq_len) log probability of each sampled action.
        """
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    def evaluate_actions(
        self, obs: Observation, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log_probs and entropy for given actions (used in PPO updates).

        Args:
            obs: Observation at the time actions were taken.
            actions: (batch, seq_len) the actions that were taken.

        Returns:
            log_probs: (batch, seq_len) log prob of each action under current policy.
            entropy: (batch, seq_len) entropy of the action distribution.
        """
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy
