"""Returns, GAE, and baseline computation utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from src.config import AlgoConfig


def compute_returns(rewards: Tensor, gamma: float) -> Tensor:
    """Compute discounted returns for each timestep.

    Args:
        rewards: (seq_len,) per-step rewards. Caller must inject terminal
            reward into rewards[-1] before calling.
        gamma: Discount factor in [0, 1].

    Returns:
        (seq_len,) discounted returns.
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got shape {rewards.shape}")
    returns = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t].item() + gamma * running
        returns[t] = running
    return returns


def compute_gae(rewards: Tensor, values: Tensor, config: AlgoConfig) -> Tensor:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: (seq_len,) per-step rewards.
        values: (seq_len,) value estimates from critic.
        config: Algorithm config providing gamma and gae_lambda.

    Returns:
        (seq_len,) advantage estimates.

    Note: assumes terminal episodes. V(s_{T+1}) is bootstrapped as 0.
    For truncated rollouts, this underestimates advantages at the last step.
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got shape {rewards.shape}")
    if rewards.shape != values.shape:
        raise ValueError(
            f"rewards and values must have the same shape, got {rewards.shape} vs {values.shape}"
        )
    gamma, lam = config.gamma, config.gae_lambda
    advantages = torch.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1].item() if t + 1 < len(values) else 0.0
        delta = rewards[t].item() + gamma * next_value - values[t].item()
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages


class MovingAverageBaseline:
    """Exponential moving average baseline for variance reduction."""

    def __init__(self, decay: float = 0.99) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        self._decay = decay
        self._value: float | None = None

    def update(self, returns: Tensor) -> None:
        """Update the running average with new episode returns."""
        if returns.numel() == 0:
            raise ValueError("returns tensor must not be empty")
        batch_mean = returns.mean().item()
        if self._value is None:
            self._value = batch_mean
        else:
            self._value = self._decay * self._value + (1 - self._decay) * batch_mean

    def get(self) -> float:
        """Return current baseline value."""
        if self._value is None:
            raise RuntimeError("Baseline not initialized — call update() first.")
        return self._value
