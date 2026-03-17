from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from src.algo.advantage import compute_gae
from src.algo.base import Algorithm
from src.config import AlgoConfig
from src.data.types import Episode
from src.env.spaces import Observation
from src.policy.base import Policy


def whiten(values: Tensor) -> Tensor:
    return (values - values.mean()) * torch.rsqrt(values.var() + 1e-8)


class ValueHead(nn.Module):

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.mlp(features).squeeze(-1)


def _build_chunk_obs(
    episode: Episode, start: int, end: int, chunk_idx: int, total_chunks: int,
) -> Observation:
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


class A2C(Algorithm):
    def __init__(self, policy: Policy, config: AlgoConfig) -> None:
        self.policy = policy
        self.config = config
        self.value_head = ValueHead(policy.head.in_features).to(
            next(policy.parameters()).device
        )
        self._captured_features: Tensor | None = None
        policy.head.register_forward_pre_hook(self._capture_hook)
        self.optimizer = Adam(
            list(policy.parameters()) + list(self.value_head.parameters()),
            lr=config.lr,
        )

    def _capture_hook(self, module: nn.Module, input: tuple[Tensor, ...]) -> None:
        self._captured_features = input[0]

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        if not episodes:
            raise ValueError("episodes must not be empty")

        totals: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "mean_return": 0.0,
            "returns_std": 0.0,
            "value_mean": 0.0,
        }
        for episode in episodes:
            metrics = self._update_episode(episode)
            for k, v in metrics.items():
                totals[k] += v

        n = len(episodes)
        return {k: v / n for k, v in totals.items()}

    def _update_episode(self, episode: Episode) -> dict[str, float]:
        device = next(self.policy.parameters()).device
        rewards = episode.rewards.clone().to(device)
        if episode.terminal_reward != 0.0:
            kept = episode.actions.bool().nonzero(as_tuple=True)[0]
            if kept.numel() > 0:
                rewards[kept[-1]] += episode.terminal_reward
            else:
                rewards[-1] += episode.terminal_reward

        log_probs, values, entropy = self._evaluate_episode(episode)
        advantages = compute_gae(rewards, values.detach(), self.config)
        returns = advantages + values.detach()
        advantages = whiten(advantages)

        policy_loss = -(log_probs * advantages).mean()
        value_loss = (values - returns).pow(2).mean()
        entropy_mean = entropy.mean()

        loss = (
            policy_loss
            + self.config.value_coeff * value_loss
            - self.config.entropy_coeff * entropy_mean
        )
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(
            list(self.policy.parameters()) + list(self.value_head.parameters()),
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_mean.item(),
            "mean_return": returns.mean().item(),
            "returns_std": returns.std().item(),
            "value_mean": values.mean().item(),
        }

    def _get_values(self) -> Tensor:
        return self.value_head(self._captured_features)

    def _evaluate_episode(self, episode: Episode) -> tuple[Tensor, Tensor, Tensor]:
        boundaries = episode.chunk_boundaries or [0]
        seq_len = episode.actions.shape[0]
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
            obs = _build_chunk_obs(episode, start, obs_end, i, len(boundaries)).to(device)
            lp, ent = self.policy.evaluate_actions(
                obs, episode.actions[start:obs_end].unsqueeze(0).to(device),
            )
            values = self._get_values()
            skip = episode.overlap if i > 0 and episode.overlap > 0 else 0
            all_lp.append(lp[0][skip:])
            all_val.append(values[0][skip:])
            all_ent.append(ent[0][skip:])

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
