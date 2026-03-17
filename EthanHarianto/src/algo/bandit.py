"""Contextual bandit: independent per-token decisions, no temporal credit assignment."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from src.algo.base import Algorithm
from src.config import AlgoConfig
from src.data.types import Episode, Prompt
from src.env.chunking import merge_chunk_actions
from src.env.compression_env import CompressionEnv
from src.env.spaces import Observation
from src.policy.base import Policy


class ContextualBandit(Algorithm):
    """Contextual bandit treating each token independently.

    No discounting, no advantage estimation, no baseline.
    Works with any reward type; zero rewards produce valid entropy-only gradient updates.
    """

    def __init__(self, policy: Policy, config: AlgoConfig) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = Adam(policy.parameters(), lr=config.lr)
        self._step = 0
        self._collection_epsilon: float = config.epsilon_start

    def _get_epsilon(self) -> float:
        """Linear decay from epsilon_start to epsilon_end over epsilon_decay steps."""
        frac = min(1.0, self._step / max(1, self.config.epsilon_decay))
        return self.config.epsilon_start + frac * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def _select_action(
        self, obs: Observation, epsilon: float,
    ) -> tuple[Tensor, Tensor]:
        """Epsilon-greedy: random with prob epsilon, else softmax sample."""
        seq_len = obs.token_ids.shape[0]
        if torch.rand(1).item() < epsilon:
            # log_probs unused; update() recomputes via _recompute_log_probs.
            # Off-policy IS correction omitted: REINFORCE gradient on random
            # actions is a known approximation; exploration benefit outweighs bias.
            return (
                torch.randint(0, 2, (seq_len,), device=obs.token_ids.device),
                torch.zeros(seq_len, device=obs.token_ids.device),
            )
        with torch.no_grad():
            actions, log_probs = self.policy.act(obs)
        return actions.squeeze(0), log_probs.squeeze(0).detach()

    def collect_episode(
        self, env: CompressionEnv, prompt: Prompt,
    ) -> Episode:
        """Epsilon-greedy episode collection with chunk merging."""
        self._collection_epsilon = self._get_epsilon()
        episode = self._collect_chunks(env, prompt, self._collection_epsilon)
        self._step += 1
        return episode

    def _collect_chunks(
        self, env: CompressionEnv, prompt: Prompt, epsilon: float,
    ) -> Episode:
        obs = env.reset(prompt)
        device = next(self.policy.parameters()).device
        obs = obs.to(device)
        done = False
        chunk_log_probs: list[Tensor] = []
        while not done:
            actions, lp = self._select_action(obs, epsilon)
            chunk_log_probs.append(lp)
            obs, _reward, done, _info = env.step(actions)
            obs = obs.to(device)
        episode = env.get_episode()
        episode.log_probs = merge_chunk_actions(chunk_log_probs, env.chunk_config)
        return episode

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        """Per-token policy gradient with immediate reward only.

        Returns:
            Metrics: policy_loss, mean_reward, entropy, epsilon.
        """
        if not episodes:
            raise ValueError("episodes must not be empty")
        total_policy_loss = total_reward = total_entropy = 0.0
        for episode in episodes:
            pl, rw, ent = self._compute_episode_loss(episode)
            total_policy_loss += pl
            total_reward += rw
            total_entropy += ent
        n = len(episodes)
        return {
            "policy_loss": total_policy_loss / n,
            "mean_reward": total_reward / n,
            "entropy": total_entropy / n,
            "epsilon": self._collection_epsilon,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._step = state.get("step", 0)

    def _compute_episode_loss(self, episode: Episode) -> tuple[float, float, float]:
        device = next(self.policy.parameters()).device
        actions_for_grad = episode.sampled_actions if episode.sampled_actions is not None else episode.actions
        rewards = episode.rewards.clone().to(device)
        if episode.terminal_reward != 0.0:
            n_kept = actions_for_grad.bool().sum().item()
            if n_kept > 0:
                keep_mask = actions_for_grad.bool().to(device)
                rewards[keep_mask] += episode.terminal_reward / n_kept
            else:
                rewards += episode.terminal_reward / rewards.shape[0]
        log_probs, entropy = self._recompute_log_probs(episode)
        if episode.forced_keep_mask is not None:
            train_mask = (~episode.forced_keep_mask.to(device)).float()
            denom = train_mask.sum().clamp_min(1.0)
            policy_loss = -((log_probs * rewards) * train_mask).sum() / denom
            entropy_term = (entropy * train_mask).sum() / denom
        else:
            policy_loss = -(log_probs * rewards).mean()
            entropy_term = entropy.mean()
        loss = policy_loss - self.config.entropy_coeff * entropy_term
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return policy_loss.item(), rewards.mean().item(), entropy_term.item()

    def _recompute_log_probs(self, episode: Episode) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute per-token log_probs matching collection-time context windows."""
        actions_for_grad = episode.sampled_actions if episode.sampled_actions is not None else episode.actions
        boundaries = episode.chunk_boundaries or [0]
        seq_len = actions_for_grad.shape[0]
        device = next(self.policy.parameters()).device
        chunk_log_probs: list[torch.Tensor] = []
        chunk_entropies: list[torch.Tensor] = []
        for i, start in enumerate(boundaries):
            obs_end = min(start + episode.chunk_size, seq_len) if episode.chunk_size > 0 else (
                boundaries[i + 1] if i + 1 < len(boundaries) else seq_len
            )
            if start >= obs_end:
                continue
            seen_end = min(start + episode.overlap, seq_len) if start > 0 else 0
            ratio_so_far = actions_for_grad[:seen_end].float().mean().item() if seen_end > 0 else 0.0
            obs = Observation(
                token_ids=episode.prompt.token_ids[start:obs_end],
                attention_mask=episode.prompt.attention_mask[start:obs_end],
                position_ids=torch.arange(start, obs_end, device=device),
                compression_ratio_so_far=ratio_so_far,
                target_compression_ratio=episode.target_compression_ratio,
                chunk_index=i,
                total_chunks=len(boundaries),
            )
            obs = obs.to(device)
            chunk_actions = actions_for_grad[start:obs_end].unsqueeze(0).to(device)
            lp, ent = self.policy.evaluate_actions(obs, chunk_actions)
            skip = episode.overlap if i > 0 and episode.overlap > 0 else 0
            chunk_log_probs.append(lp[0][skip:])
            chunk_entropies.append(ent[0][skip:])
        if not chunk_log_probs:
            raise ValueError("No valid chunks processed; check chunk_boundaries and seq_len")
        return torch.cat(chunk_log_probs), torch.cat(chunk_entropies)
