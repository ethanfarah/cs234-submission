"""MCTS episode collection with REINFORCE policy updates + learned value head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from src.algo.mcts_search import run_mcts
from src.algo.mcts_types import MCTSConfig
from src.algo.reinforce_simple import SimpleREINFORCE
from src.algo.value_fn import HeuristicValue, LearnedValue
from src.config import Config
from src.data.types import Episode, Prompt
from src.env.compression_env import CompressionEnv
from src.env.spaces import Observation
from src.policy.base import Policy
from src.policy.distilroberta import DistilRoBERTaPolicy


class MCTSAlgorithm(SimpleREINFORCE):
    """MCTS for episode collection, REINFORCE for policy updates.

    When use_learned_value=True, the policy's value head is used for leaf
    evaluation (blended with the heuristic EMA), and a value loss is added
    to the update step.
    """

    def __init__(self, policy: Policy, config: Config) -> None:
        if config.ratio_conditioned:
            raise ValueError(
                "MCTS is incompatible with ratio_conditioned=True; "
                "ratio context is not propagated through the search tree"
            )
        super().__init__(policy, config)
        self.mcts_config = MCTSConfig(
            num_simulations=config.num_simulations,
            c_puct=config.c_puct,
            num_action_samples=config.num_action_samples,
            temperature=config.mcts_temperature,
            chunk_overlap=config.chunk_overlap,
        )
        self.use_learned_value = (
            isinstance(policy, DistilRoBERTaPolicy)
            and policy.value_head is not None
        )
        self.value_fn: HeuristicValue | LearnedValue = HeuristicValue(
            ema_decay=config.faithfulness_ema_decay
        )
        self._value_coeff = 0.5
        self._value_warmup = config.value_warmup_episodes
        self._episode_count = 0
        if self.use_learned_value:
            assert isinstance(policy, DistilRoBERTaPolicy)
            from torch.optim import Adam
            value_params = list(policy.value_head.parameters())
            policy_params = [p for p in policy.parameters()
                             if not any(p is vp for vp in value_params)]
            self.optimizer = Adam(policy_params, lr=config.lr)
            self._value_optimizer = Adam(value_params, lr=config.value_lr)
            self._value_grad_clip = config.value_grad_clip

    def collect_episode(
        self, env: CompressionEnv, prompt: Prompt,
    ) -> Episode | None:
        """Run MCTS to select actions, replay through env."""
        env.reset(prompt)
        chunks = list(env.chunks)

        # Create a per-episode LearnedValue if value head is available
        if self.use_learned_value:
            assert isinstance(self.policy, DistilRoBERTaPolicy)
            value_fn = LearnedValue(
                policy=self.policy,
                chunks=chunks,
                device=self.config.device,
                ema_decay=self.value_fn._decay,
                warmup_episodes=self._value_warmup,
            )
            value_fn._ema = self.value_fn._ema
            value_fn.set_episode(self._episode_count)
        else:
            value_fn = self.value_fn

        action_sequence = run_mcts(
            chunks, prompt, self.policy, value_fn,
            self.mcts_config, device=self.config.device,
        )

        # Replay MCTS actions through a fresh env reset
        obs = env.reset(prompt)
        done = False
        for chunk_actions in action_sequence:
            obs, _, done, _ = env.step(chunk_actions)

        # Fallback: stochastic policy rollout for unexplored chunks
        while not done:
            obs, _, done, _ = self._fallback_step(env, obs)

        return env.get_episode()

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        """REINFORCE update with separate value head optimizer."""
        if not self.use_learned_value:
            return super().update(episodes)

        if not episodes:
            raise ValueError("episodes must not be empty")

        self._episode_count += len(episodes)

        cached: list[tuple[torch.Tensor, torch.Tensor, float]] = []
        advantages: list[float] = []
        sample_rewards: list[float] = []
        for episode in episodes:
            lp, ent, sstd = self._recompute_log_probs(episode)
            cached.append((lp, ent, sstd))
            hybrid_baseline = 0.5 * episode.baseline_reward + 0.5 * self.baseline_ema
            advantages.append(episode.terminal_reward - hybrid_baseline)
            sample_rewards.append(episode.terminal_reward)

        if len(advantages) >= 8:
            adv_t = torch.tensor(advantages)
            advantages = ((adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)).tolist()

        from torch.nn.utils import clip_grad_norm_
        from src.algo.reinforce_simple import _compute_grad_norm

        n = len(episodes)
        total_policy_loss = total_value_loss = 0.0
        total_reward = total_entropy = total_score_std = 0.0

        assert isinstance(self.policy, DistilRoBERTaPolicy)

        # --- Policy backward pass ---
        self.optimizer.zero_grad()
        for i, episode in enumerate(episodes):
            log_probs, entropy, score_std = cached[i]
            policy_loss = -(log_probs.sum() * advantages[i])
            loss = (policy_loss - self.config.entropy_coeff * entropy.mean()) / n
            loss.backward()
            total_policy_loss += policy_loss.item()
            total_reward += episode.terminal_reward
            total_entropy += entropy.mean().item()
            total_score_std += score_std

        policy_grad_norm = _compute_grad_norm(self.policy)
        clip_grad_norm_(
            [p for p in self.policy.parameters()
             if p not in set(self.policy.value_head.parameters())],
            self.config.max_grad_norm,
        )
        self.optimizer.step()

        # --- Value backward pass (separate) ---
        self._value_optimizer.zero_grad()
        for i, episode in enumerate(episodes):
            value_pred = self._predict_value(episode)
            value_target = torch.tensor(
                [episode.terminal_reward], device=value_pred.device
            )
            value_loss = F.smooth_l1_loss(value_pred, value_target, beta=0.1)
            scaled_vloss = (self._value_coeff * value_loss) / n
            scaled_vloss.backward()
            total_value_loss += value_loss.item()

        value_grad_norm = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.policy.value_head.parameters() if p.grad is not None
        ) ** 0.5
        clip_grad_norm_(
            self.policy.value_head.parameters(), self._value_grad_clip,
        )
        self._value_optimizer.step()

        mean_reward = sum(sample_rewards) / len(sample_rewards)
        self.baseline_ema = 0.99 * self.baseline_ema + 0.01 * mean_reward

        clip_ratio = (
            policy_grad_norm / self.config.max_grad_norm
            if self.config.max_grad_norm > 0 else 0.0
        )
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "mean_reward": total_reward / n,
            "entropy": total_entropy / n,
            "grad_norm_pre": policy_grad_norm,
            "grad_norm_post": value_grad_norm,
            "clip_ratio": clip_ratio,
            "advantage_mean": sum(advantages) / n,
            "advantage_max": max(advantages),
            "advantage_min": min(advantages),
            "baseline_reward": episodes[0].baseline_reward,
            "num_nonzero_adv": sum(1 for a in advantages if abs(a) > 1e-8),
            "sample_rewards": sample_rewards,
            "score_std": total_score_std / n,
        }

    def _predict_value(self, episode: Episode) -> torch.Tensor:
        """Get value prediction from the first chunk of the episode."""
        assert isinstance(self.policy, DistilRoBERTaPolicy)
        self.policy.eval()
        boundaries = episode.chunk_boundaries or [0]
        seq_len = episode.actions.shape[0]
        device = next(self.policy.parameters()).device

        start = boundaries[0]
        obs_end = min(start + episode.chunk_size, seq_len) if episode.chunk_size > 0 else (
            boundaries[1] if len(boundaries) > 1 else seq_len
        )
        obs = Observation(
            token_ids=episode.prompt.token_ids[start:obs_end].to(device),
            attention_mask=episode.prompt.attention_mask[start:obs_end].to(device),
            position_ids=torch.arange(start, obs_end, device=device),
            compression_ratio_so_far=0.0,
            target_compression_ratio=episode.target_compression_ratio,
            chunk_index=0,
            total_chunks=len(boundaries),
        )
        _, value = self.policy.forward_with_value(obs)
        self.policy.train()
        return value

    def _fallback_step(
        self, env: CompressionEnv, obs: Observation,
    ) -> tuple[Observation, float, bool, dict]:
        """Step one chunk with stochastic policy (no obs mutation)."""
        obs_device = Observation(
            token_ids=obs.token_ids.to(self.config.device),
            attention_mask=obs.attention_mask.to(self.config.device),
            position_ids=obs.position_ids.to(self.config.device),
            compression_ratio_so_far=obs.compression_ratio_so_far,
            target_compression_ratio=obs.target_compression_ratio,
            chunk_index=obs.chunk_index,
            total_chunks=obs.total_chunks,
        )
        with torch.no_grad():
            actions, _ = self.policy.act(obs_device)
        if actions.shape[0] != 1:
            raise ValueError(f"_fallback_step requires batch=1, got {actions.shape[0]}")
        return env.step(actions[0].cpu())

    def update_value_fn(self, faithfulness: float) -> None:
        self.value_fn.update(faithfulness)

    def state_dict(self) -> dict[str, Any]:
        base = super().state_dict()
        base["value_fn"] = self.value_fn.state_dict()
        base["episode_count"] = self._episode_count
        if self.use_learned_value:
            base["value_optimizer"] = self._value_optimizer.state_dict()
        return base

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        if "value_fn" in state:
            self.value_fn.load_state_dict(state["value_fn"])
        self._episode_count = state.get("episode_count", 0)
        if self.use_learned_value and "value_optimizer" in state:
            self._value_optimizer.load_state_dict(state["value_optimizer"])
