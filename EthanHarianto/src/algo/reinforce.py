"""REINFORCE with variance-reduction baselines."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from src.algo.advantage import MovingAverageBaseline, compute_returns
from src.algo.base import Algorithm
from src.config import AlgoConfig, BaselineType
from src.data.types import Episode
from src.env.spaces import Observation
from src.policy.base import Policy

_MAX_SEQ_LEN = 1024  # matches DataConfig.max_prompt_tokens; used to normalize seq_len feature


class LearnedBaseline(nn.Module):
    """Small MLP baseline: [compression_ratio, normalized_seq_len] -> scalar."""

    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 1))
        self._optimizer = Adam(self.parameters(), lr=lr)

    def _features(self, compression_ratio: float, seq_len: int) -> Tensor:
        return torch.tensor(
            [compression_ratio, seq_len / _MAX_SEQ_LEN],
            dtype=torch.float32,
            device=next(self.parameters()).device,
        )

    def predict(self, compression_ratio: float, seq_len: int) -> float:
        with torch.no_grad():
            return self.net(self._features(compression_ratio, seq_len)).item()

    def update(self, returns: Tensor, compression_ratio: float, seq_len: int) -> None:
        target = returns.mean().detach()
        pred = self.net(self._features(compression_ratio, seq_len)).squeeze()
        loss = nn.functional.mse_loss(pred, target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()


class REINFORCE(Algorithm):
    """REINFORCE policy gradient with optional baseline."""

    def __init__(self, policy: Policy, config: AlgoConfig) -> None:
        self.policy = policy
        self.config = config
        self.optimizer = Adam(policy.parameters(), lr=config.lr)
        if config.baseline_type == BaselineType.MOVING_AVERAGE:
            self.baseline: MovingAverageBaseline | LearnedBaseline | None = MovingAverageBaseline()
        elif config.baseline_type == BaselineType.LEARNED:
            device = next(policy.parameters()).device
            self.baseline = LearnedBaseline(lr=config.lr).to(device)
        else:
            self.baseline = None

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        """REINFORCE policy gradient update. Returns: policy_loss, mean_return, entropy."""
        if not episodes:
            raise ValueError("episodes must not be empty")
        total_pl = total_ret = total_ent = 0.0
        for episode in episodes:
            pl, ret, ent = self._compute_episode_loss(episode)
            total_pl += pl
            total_ret += ret
            total_ent += ent
        n = len(episodes)
        return {"policy_loss": total_pl / n, "mean_return": total_ret / n, "entropy": total_ent / n}

    def state_dict(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        if isinstance(self.baseline, MovingAverageBaseline):
            state["baseline"] = {"value": self.baseline._value}
        elif isinstance(self.baseline, LearnedBaseline):
            state["baseline"] = {
                "net": self.baseline.net.state_dict(),
                "optimizer": self.baseline._optimizer.state_dict(),
            }
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy.load_state_dict(state["policy"])
        self.optimizer.load_state_dict(state["optimizer"])
        if "baseline" not in state:
            return
        if isinstance(self.baseline, MovingAverageBaseline):
            self.baseline._value = state["baseline"]["value"]
        elif isinstance(self.baseline, LearnedBaseline):
            self.baseline.net.load_state_dict(state["baseline"]["net"])
            self.baseline._optimizer.load_state_dict(state["baseline"]["optimizer"])

    def _compute_episode_loss(self, episode: Episode) -> tuple[float, float, float]:
        device = next(self.policy.parameters()).device
        actions_for_grad = episode.sampled_actions if episode.sampled_actions is not None else episode.actions
        rewards = episode.rewards.clone().to(device)
        if episode.terminal_reward != 0.0:
            kept = actions_for_grad.bool().nonzero(as_tuple=True)[0]
            if kept.numel() > 0:
                rewards[kept[-1]] += episode.terminal_reward
            else:
                rewards[-1] += episode.terminal_reward
        returns = compute_returns(rewards, self.config.gamma)
        b = self._get_baseline(episode, returns)
        self._update_baseline(episode, returns)
        advantages = returns - b
        if advantages.numel() > 1 and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        log_probs, entropy = self._recompute_log_probs(episode)
        if episode.forced_keep_mask is not None:
            train_mask = (~episode.forced_keep_mask.to(device)).float()
            denom = train_mask.sum().clamp_min(1.0)
            policy_loss = -((log_probs * advantages) * train_mask).sum() / denom
            entropy_term = (entropy * train_mask).sum() / denom
        else:
            policy_loss = -(log_probs * advantages).mean()
            entropy_term = entropy.mean()
        loss = policy_loss - self.config.entropy_coeff * entropy_term
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        return policy_loss.item(), returns.mean().item(), entropy_term.item()

    def _get_baseline(self, episode: Episode, returns: Tensor) -> float:
        if self.baseline is None:
            return 0.0
        if isinstance(self.baseline, MovingAverageBaseline):
            if self.baseline._value is None:
                return 0.0
            return self.baseline.get()
        return self.baseline.predict(
            episode.compressed.compression_ratio,
            actions_for_grad.shape[0],
        )

    def _update_baseline(self, episode: Episode, returns: Tensor) -> None:
        if self.baseline is None:
            return
        if isinstance(self.baseline, MovingAverageBaseline):
            self.baseline.update(returns)
        else:
            self.baseline.update(
                returns,
                episode.compressed.compression_ratio,
                actions_for_grad.shape[0],
            )

    def _recompute_log_probs(self, episode: Episode) -> tuple[Tensor, Tensor]:
        """Recompute per-token log_probs matching collection-time context windows."""
        boundaries = episode.chunk_boundaries or [0]
        actions_for_grad = episode.sampled_actions if episode.sampled_actions is not None else episode.actions
        seq_len = actions_for_grad.shape[0]
        device = next(self.policy.parameters()).device
        chunk_log_probs: list[Tensor] = []
        chunk_entropies: list[Tensor] = []
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
