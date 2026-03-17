"""Standard REINFORCE with SCST baseline for token-level binary decisions."""

from __future__ import annotations

import sys
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from src.algo.base import Algorithm
from src.config import Config
from src.data.types import Episode
from src.env.spaces import Observation
from src.policy.base import Policy


def _compute_grad_norm(policy: Policy) -> float:
    total = 0.0
    for p in policy.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


class SimpleREINFORCE(Algorithm):
    """REINFORCE with SCST baseline, faithful to the PCRL paper.

    Key differences from ContextualBandit:
    - Scalar advantage (R - R_greedy), not per-token reward distribution
    - sum(log_probs) * advantage (standard REINFORCE gradient estimator)
    - No per-token reward distribution — advantage is episode-level
    """

    def __init__(self, policy: Policy, config: Config) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = Adam(policy.parameters(), lr=config.lr)
        self.baseline_ema: float = 0.0

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        if not episodes:
            raise ValueError("episodes must not be empty")

        # First pass: compute advantages and cache log_probs/entropy/score_std
        cached: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        advantages: list[float] = []
        sample_rewards: list[float] = []
        for episode in episodes:
            lp, ent, sstd = self._recompute_log_probs(episode)
            cached.append((lp, ent, sstd))
            hybrid_baseline = 0.5 * episode.baseline_reward + 0.5 * self.baseline_ema
            advantages.append(episode.terminal_reward - hybrid_baseline)
            sample_rewards.append(episode.terminal_reward)

        # Normalize advantages if enough samples
        if len(advantages) >= 8:
            adv_t = torch.tensor(advantages)
            advantages = ((adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)).tolist()

        # Second pass: compute losses with normalized advantages
        self.optimizer.zero_grad()
        total_policy_loss = total_reward = total_entropy = total_score_std = 0.0
        n = len(episodes)
        for i, episode in enumerate(episodes):
            log_probs, entropy, score_std = cached[i]
            policy_loss = -(log_probs.sum() * advantages[i])
            loss = (policy_loss - self.config.entropy_coeff * entropy.mean()) / n
            loss.backward()
            total_policy_loss += policy_loss.item()
            total_reward += episode.terminal_reward
            total_entropy += entropy.mean().item()
            total_score_std += score_std

        for name, p in self.policy.named_parameters():
            if p.grad is not None and "head" in name:
                print(f"  grad/{name}: {p.grad.norm().item():.6f}", file=sys.stderr)

        grad_norm_pre = _compute_grad_norm(self.policy)
        grad_norm_post = clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        # Update EMA baseline
        mean_reward = sum(sample_rewards) / len(sample_rewards)
        self.baseline_ema = 0.99 * self.baseline_ema + 0.01 * mean_reward

        clip_ratio = grad_norm_pre / self.config.max_grad_norm if self.config.max_grad_norm > 0 else 0.0
        return {
            "policy_loss": total_policy_loss / n,
            "mean_reward": total_reward / n,
            "entropy": total_entropy / n,
            "grad_norm_pre": grad_norm_pre,
            "grad_norm_post": float(grad_norm_post),
            "clip_ratio": clip_ratio,
            "advantage_mean": sum(advantages) / n,
            "advantage_max": max(advantages),
            "advantage_min": min(advantages),
            "baseline_reward": episodes[0].baseline_reward,
            "num_nonzero_adv": sum(1 for a in advantages if abs(a) > 1e-8),
            "sample_rewards": sample_rewards,
            "score_std": total_score_std / n,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "baseline_ema": self.baseline_ema,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"], strict=False)
        try:
            self.optimizer.load_state_dict(state["optimizer"])
        except (ValueError, KeyError):
            pass  # optimizer shape mismatch from architecture changes; safe to skip for eval
        self.baseline_ema = state.get("baseline_ema", 0.0)

    def _recompute_log_probs(self, episode: Episode) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Recompute per-token log_probs matching collection-time context windows."""
        self.policy.eval()
        boundaries = episode.chunk_boundaries or [0]
        seq_len = episode.actions.shape[0]
        policy_device = next(self.policy.parameters()).device
        chunk_log_probs: list[torch.Tensor] = []
        chunk_entropies: list[torch.Tensor] = []
        chunk_score_stds: list[float] = []
        for i, start in enumerate(boundaries):
            obs_end = min(start + episode.chunk_size, seq_len) if episode.chunk_size > 0 else (
                boundaries[i + 1] if i + 1 < len(boundaries) else seq_len
            )
            if start >= obs_end:
                continue
            seen_end = min(start + episode.overlap, seq_len) if start > 0 else 0
            ratio_so_far = episode.actions[:seen_end].float().mean().item() if seen_end > 0 else 0.0
            obs = Observation(
                token_ids=episode.prompt.token_ids[start:obs_end].to(policy_device),
                attention_mask=episode.prompt.attention_mask[start:obs_end].to(policy_device),
                position_ids=torch.arange(start, obs_end, device=policy_device),
                compression_ratio_so_far=ratio_so_far,
                target_compression_ratio=episode.target_compression_ratio,
                chunk_index=i,
                total_chunks=len(boundaries),
            )
            chunk_actions = episode.actions[start:obs_end].unsqueeze(0).to(policy_device)
            lp, ent, sstd = self.policy.evaluate_actions(obs, chunk_actions)
            skip = episode.overlap if i > 0 and episode.overlap > 0 else 0
            chunk_log_probs.append(lp[0][skip:])
            chunk_entropies.append(ent[0][skip:])
            chunk_score_stds.append(sstd[0].item())
        self.policy.train()
        if not chunk_log_probs:
            raise ValueError("No valid chunks processed; check chunk_boundaries and seq_len")
        mean_score_std = sum(chunk_score_stds) / len(chunk_score_stds)
        return torch.cat(chunk_log_probs), torch.cat(chunk_entropies), mean_score_std
