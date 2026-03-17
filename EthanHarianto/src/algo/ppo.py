"""PPO with clipped surrogate objective and value head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from src.algo.advantage import compute_gae
from src.algo.base import Algorithm
from src.config import AlgoConfig, TerminalRewardDistribution
from src.data.types import Episode
from src.env.spaces import Observation
from src.policy.base import Policy


@dataclass
class _PPOBatch:
    """Bundled tensors for a single PPO update step."""

    old_log_probs: Tensor
    advantages: Tensor
    returns: Tensor


class ValueHead(nn.Module):
    """Simple MLP for state value estimation."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Compute scalar value per state.

        Args:
            features: (batch, seq_len, hidden) pre-head hidden states.

        Returns:
            (batch, seq_len) value estimates.
        """
        return self.mlp(features).squeeze(-1)


def _build_chunk_obs(
    episode: Episode, start: int, end: int, chunk_idx: int, total_chunks: int,
) -> Observation:
    """Construct Observation for a single chunk of an episode."""
    seen_end = min(start + episode.overlap, end) if start > 0 else 0
    ratio_so_far = episode.actions[:seen_end].float().mean().item() if seen_end > 0 else 0.0
    return Observation(
        token_ids=episode.prompt.token_ids[start:end],
        attention_mask=episode.prompt.attention_mask[start:end],
        position_ids=torch.arange(start, end, device=episode.prompt.token_ids.device),
        compression_ratio_so_far=ratio_so_far,
        target_compression_ratio=episode.target_compression_ratio,
        chunk_index=chunk_idx,
        total_chunks=total_chunks,
    )


class PPO(Algorithm):
    """PPO with clipped surrogate objective.

    Requires policy to expose a ``head: nn.Linear`` attribute for feature capture.
    """

    def __init__(self, policy: Policy, config: AlgoConfig) -> None:
        self.policy = policy
        self.config = config
        self.value_head = ValueHead(policy.head.in_features).to(
            next(policy.parameters()).device
        )
        self.clip_eps = config.clip_eps
        self._captured_features: Tensor | None = None
        policy.head.register_forward_pre_hook(self._capture_hook)
        self.optimizer = Adam(
            list(policy.parameters()) + list(self.value_head.parameters()),
            lr=config.lr,
        )

    def _capture_hook(self, module: nn.Module, input: tuple[Tensor, ...]) -> None:
        self._captured_features = input[0]

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        """PPO clipped surrogate update with multiple epochs.

        Returns:
            Dict with keys: policy_loss, value_loss, clip_fraction, entropy, mean_return.
        """
        if not episodes:
            raise ValueError("episodes must not be empty")

        totals: dict[str, float] = {
            "policy_loss": 0.0, "value_loss": 0.0,
            "clip_fraction": 0.0, "entropy": 0.0, "mean_return": 0.0,
        }
        for episode in episodes:
            metrics = self._update_episode(episode)
            for k, v in metrics.items():
                totals[k] += v

        n = len(episodes)
        return {k: v / n for k, v in totals.items()}

    def _update_episode(self, episode: Episode) -> dict[str, float]:
        device = next(self.policy.parameters()).device
        actions_for_grad = episode.sampled_actions if episode.sampled_actions is not None else episode.actions
        rewards = episode.rewards.clone().to(device)
        if episode.terminal_reward != 0.0:
            self._apply_terminal_reward(rewards, episode, actions_for_grad)

        try:
            self.policy.eval()
            with torch.no_grad():
                old_log_probs, old_values, _ = self._evaluate_episode(episode)
        finally:
            self.policy.train()

        advantages = compute_gae(rewards, old_values, self.config)
        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        if episode.forced_keep_mask is not None:
            train_mask = (~episode.forced_keep_mask.to(device)).float()
        else:
            train_mask = torch.ones_like(advantages)
        batch = _PPOBatch(old_log_probs, advantages * train_mask, returns * train_mask)

        totals = {"policy_loss": 0.0, "value_loss": 0.0, "clip_fraction": 0.0, "entropy": 0.0}
        for _ in range(self.config.num_epochs):
            step = self._ppo_step(episode, batch)
            for k, v in step.items():
                totals[k] += v

        n = self.config.num_epochs
        return {k: v / n for k, v in totals.items()} | {"mean_return": returns.mean().item()}

    def _apply_terminal_reward(self, rewards: Tensor, episode: Episode, actions_for_grad: Tensor) -> None:
        """Apply scalar terminal reward according to configured distribution."""
        kept = actions_for_grad.bool().nonzero(as_tuple=True)[0]
        mode = self.config.terminal_reward_distribution
        if kept.numel() == 0:
            rewards[-1] += episode.terminal_reward
            return
        if mode == TerminalRewardDistribution.LAST_KEPT:
            rewards[kept[-1]] += episode.terminal_reward
            return
        if mode == TerminalRewardDistribution.ALL_KEPT_MEAN:
            rewards[kept] += episode.terminal_reward / kept.numel()
            return
        raise ValueError(f"Unknown terminal_reward_distribution: {mode}")

    def _ppo_step(self, episode: Episode, batch: _PPOBatch) -> dict[str, float]:
        """Single PPO epoch: forward pass, clipped surrogate loss, gradient step."""
        new_log_probs, new_values, entropy = self._evaluate_episode(episode)
        ratio = torch.exp(new_log_probs - batch.old_log_probs)

        surr1 = ratio * batch.advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * batch.advantages
        if batch.advantages.numel() > 0:
            active = (batch.advantages != 0).float()
            denom = active.sum().clamp_min(1.0)
            policy_loss = -(torch.min(surr1, surr2) * active).sum() / denom
            value_loss = ((new_values - batch.returns) ** 2 * active).sum() / denom
            entropy_mean = (entropy * active).sum() / denom
        else:
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(new_values, batch.returns)
            entropy_mean = entropy.mean()

        loss = policy_loss + self.config.value_coeff * value_loss - self.config.entropy_coeff * entropy_mean
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
        return {
            "policy_loss": policy_loss.item(), "value_loss": value_loss.item(),
            "clip_fraction": clip_fraction, "entropy": entropy_mean.item(),
        }

    def _get_values(self) -> Tensor:
        """Extract values from captured features via the value head."""
        if self._captured_features is None:
            raise RuntimeError(
                "ValueHead hook did not capture features; "
                "ensure policy.head is called in forward()"
            )
        return self.value_head(self._captured_features)

    def _evaluate_episode(self, episode: Episode) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate log_probs, values, and entropy for all tokens in episode."""
        actions_for_grad = episode.sampled_actions if episode.sampled_actions is not None else episode.actions
        boundaries = episode.chunk_boundaries or [0]
        seq_len = actions_for_grad.shape[0]
        device = next(self.policy.parameters()).device
        all_lp: list[Tensor] = []
        all_val: list[Tensor] = []
        all_ent: list[Tensor] = []

        for i, start in enumerate(boundaries):
            obs_end = min(start + episode.chunk_size, seq_len) if episode.chunk_size > 0 else (
                boundaries[i + 1] if i + 1 < len(boundaries) else seq_len
            )
            if start >= obs_end:
                continue
            obs = _build_chunk_obs(episode, start, obs_end, i, len(boundaries))
            obs = obs.to(device)
            lp, ent = self.policy.evaluate_actions(obs, actions_for_grad[start:obs_end].unsqueeze(0).to(device))
            values = self._get_values()
            skip = episode.overlap if i > 0 and episode.overlap > 0 else 0
            all_lp.append(lp[0][skip:])
            all_val.append(values[0][skip:])
            all_ent.append(ent[0][skip:])

        if not all_lp:
            raise ValueError("No valid chunks processed; check chunk_boundaries and seq_len")
        return torch.cat(all_lp), torch.cat(all_val), torch.cat(all_ent)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "value_head": self.value_head.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.value_head.load_state_dict(state["value_head"])
