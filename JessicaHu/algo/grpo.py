from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from src.algo.base import Algorithm
from src.config import AlgoConfig
from src.data.types import Episode, Prompt
from src.env.compression_env import CompressionEnv
from src.env.spaces import Observation
from src.env.chunking import merge_chunk_actions
from src.policy.base import Policy


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


def _group_advantages(episodes: list[Episode]) -> list[float]:
    rewards = [e.rewards.sum().item() + e.terminal_reward for e in episodes]
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = var_r ** 0.5 + 1e-8
    return [(r - mean_r) / std_r for r in rewards]


class GRPO(Algorithm):
    def __init__(self, policy: Policy, config: AlgoConfig) -> None:
        self.policy = policy
        self.config = config
        self.clip_eps = config.clip_eps
        self._buffer: list[Episode] = []
        self._last_metrics: dict[str, float] | None = None
        self._active_group_prompt: Prompt | None = None
        self._active_group_count = 0
        self.optimizer = Adam(policy.parameters(), lr=config.lr)

    def collect_episode(
        self, env: CompressionEnv, prompt: Prompt,
    ) -> Episode:
        group_prompt = self._active_group_prompt
        if group_prompt is None or self._active_group_count == 0:
            group_prompt = prompt
            self._active_group_prompt = prompt
            self._active_group_count = 0

        episode = self._collect_single_episode(env, group_prompt)
        self._active_group_count += 1
        return episode

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        self._buffer.extend(episodes)
        buffer_fill = len(self._buffer) / self.config.grpo_group_size
        if len(self._buffer) < self.config.grpo_group_size:
            base = self._last_metrics if self._last_metrics else {
                "policy_loss": 0.0, "entropy": 0.0, "mean_return": 0.0,
                "clip_fraction": 0.0, "group_reward_std": 0.0,
                "group_reward_min": 0.0, "group_reward_max": 0.0,
                "grad_norm": 0.0,
            }
            return base | {"buffer_fill": buffer_fill}
        group = self._buffer[:self.config.grpo_group_size]
        self._buffer = self._buffer[self.config.grpo_group_size:]
        metrics = self._update_group(group)
        if not self._buffer:
            self._active_group_prompt = None
            self._active_group_count = 0
        self._last_metrics = metrics
        return metrics | {"buffer_fill": 1.0}

    def _collect_single_episode(
        self, env: CompressionEnv, prompt: Prompt,
    ) -> Episode:
        device = next(self.policy.parameters()).device
        obs = env.reset(prompt).to(device)
        chunk_log_probs: list[Tensor] = []
        done = False

        while not done:
            with torch.no_grad():
                actions, log_probs = self.policy.act(obs)
            chunk_log_probs.append(log_probs[0])
            obs, _reward, done, _info = env.step(actions[0])
            obs = obs.to(device)

        episode = env.get_episode()
        episode.log_probs = merge_chunk_actions(chunk_log_probs, env.chunk_config)
        return episode

    def _update_group(self, group: list[Episode]) -> dict[str, float]:
        advantages = _group_advantages(group)
        rewards = [e.rewards.sum().item() + e.terminal_reward for e in group]
        mean_return = sum(rewards) / len(rewards)
        var_r = sum((r - mean_return) ** 2 for r in rewards) / len(rewards)
        totals: dict[str, float] = {
            "policy_loss": 0.0, "entropy": 0.0, "clip_fraction": 0.0,
            "grad_norm": 0.0,
        }
        for episode, adv in zip(group, advantages):
            for _ in range(self.config.num_epochs):
                step = self._grpo_step(episode, adv)
                for k, v in step.items():
                    totals[k] += v
        n = len(group) * self.config.num_epochs
        return {k: v / n for k, v in totals.items()} | {
            "mean_return": mean_return,
            "group_reward_std": var_r ** 0.5,
            "group_reward_min": min(rewards),
            "group_reward_max": max(rewards),
        }

    def _grpo_step(self, episode: Episode, advantage: float) -> dict[str, float]:
        device = next(self.policy.parameters()).device
        new_log_probs, entropy = self._evaluate_episode(episode)
        ratio = torch.exp(new_log_probs - episode.log_probs.to(device))

        adv_t = torch.tensor(advantage, device=device)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_mean = entropy.mean()

        loss = policy_loss - self.config.entropy_coeff * entropy_mean
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = float(
            clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        )
        self.optimizer.step()

        clip_fraction = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()
        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy_mean.item(),
            "clip_fraction": clip_fraction,
            "grad_norm": grad_norm,
        }

    def _evaluate_episode(self, episode: Episode) -> tuple[Tensor, Tensor]:
        boundaries = episode.chunk_boundaries or [0]
        seq_len = episode.actions.shape[0]
        device = next(self.policy.parameters()).device
        all_lp: list[Tensor] = []
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
            skip = episode.overlap if i > 0 and episode.overlap > 0 else 0
            all_lp.append(lp[0][skip:])
            all_ent.append(ent[0][skip:])

        if not all_lp:
            raise ValueError("No valid chunks processed; check chunk_boundaries and seq_len")
        return torch.cat(all_lp), torch.cat(all_ent)

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "buffer": self._buffer,
            "last_metrics": self._last_metrics,
            "active_group_prompt": self._active_group_prompt,
            "active_group_count": self._active_group_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._buffer = state.get("buffer", [])
        self._last_metrics = state.get("last_metrics")
        self._active_group_prompt = state.get("active_group_prompt")
        self._active_group_count = state.get("active_group_count", 0)
