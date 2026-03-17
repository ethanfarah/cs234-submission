"""DQN with dueling architecture for token-level decisions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from src.algo.base import Algorithm
from src.algo.replay_buffer import ReplayBuffer, Transition
from src.config import AlgoConfig
from src.data.types import Episode, Prompt
from src.env.compression_env import CompressionEnv
from src.env.spaces import Observation
from src.policy.base import Policy


NUM_ACTIONS = 2  # keep / drop


class DuelingHead(nn.Module):
    """Dueling network head: separate V(s) and A(s,a) streams."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        mid = input_dim // 2
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, mid), nn.ReLU(), nn.Linear(mid, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, mid), nn.ReLU(), nn.Linear(mid, NUM_ACTIONS),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Q = V + (A - A.mean over actions)."""
        v = self.value_stream(features)       # (..., 1)
        a = self.advantage_stream(features)   # (..., 2)
        return v + (a - a.mean(dim=-1, keepdim=True))


class DQN(Algorithm):
    """DQN with experience replay and target network."""

    def __init__(self, policy: Policy, config: AlgoConfig) -> None:
        self.policy = policy
        self.config = config
        if not hasattr(policy.head, 'in_features'):
            raise TypeError(
                f"DQN requires policy.head with in_features attribute; "
                f"got {type(policy.head).__name__}"
            )
        self._device = next(policy.parameters()).device
        policy.head = DuelingHead(policy.head.in_features).to(self._device)
        self.target_policy = deepcopy(policy)
        self.buffer = ReplayBuffer(config.buffer_size)
        self.optimizer = Adam(policy.parameters(), lr=config.lr)
        self._target_ratio = 0.5
        self._chunk_size = 128
        self._overlap = 16
        self._step = 0

    def _get_epsilon(self) -> float:
        """Linear decay from 1.0 to eps_end over eps_decay_steps."""
        frac = min(1.0, self._step / max(1, self.config.epsilon_decay))
        return self.config.epsilon_start + frac * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_action(self, obs: Observation, epsilon: float) -> Tensor:
        """Epsilon-greedy: random with prob epsilon, else argmax Q.

        Returns:
            (seq_len,) 1-D binary actions tensor.
        """
        seq_len = obs.token_ids.shape[0]
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, NUM_ACTIONS, (seq_len,), device=obs.token_ids.device)
        with torch.no_grad():
            q_values = self.policy(obs)  # (1, seq_len, 2)
        return q_values[0].argmax(dim=-1)  # (seq_len,)

    def collect_episode(
        self, env: CompressionEnv, prompt: Prompt,
    ) -> Episode:
        """Epsilon-greedy episode collection."""
        epsilon = self._get_epsilon()
        obs = env.reset(prompt)
        obs = obs.to(self._device)
        self._target_ratio = obs.target_compression_ratio
        self._chunk_size = env.chunk_config.chunk_size
        self._overlap = env.chunk_config.overlap
        done = False

        while not done:
            actions = self.select_action(obs, epsilon)
            obs, _reward, done, _info = env.step(actions)
            obs = obs.to(self._device)

        return env.get_episode()

    def _store_transitions(self, episodes: list[Episode]) -> None:
        """Decompose episodes into transitions and push to replay buffer."""
        for episode in episodes:
            for t in self._decompose_episode(episode):
                self.buffer.push(t)

    def update(self, episodes: list[Episode]) -> dict[str, float]:
        """Decompose episodes, push to buffer, sample and train."""
        self._store_transitions(episodes)
        if len(self.buffer) < self.config.batch_size:
            return {"q_loss": 0.0, "mean_q": 0.0, "epsilon": self._get_epsilon()}

        loss, mean_q = self._train_step()
        return {"q_loss": loss, "mean_q": mean_q, "epsilon": self._get_epsilon()}

    def _train_step(self) -> tuple[float, float]:
        """Sample from buffer, compute loss, and update weights."""
        batch = self.buffer.sample(self.config.batch_size)
        loss, mean_q = self._compute_td_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        self._step += 1

        if self._step % self.config.target_update_freq == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())
        return loss.item(), mean_q

    def _chunk_reward(
        self, episode: Episode, start: int, end: int, is_last: bool,
    ) -> Tensor:
        """Extract per-token reward for a chunk, adding terminal if sparse."""
        reward = episode.rewards[start:end]
        if not is_last:
            return reward
        dense_sum = episode.rewards.sum().item()
        has_independent_terminal = (
            abs(dense_sum) < 1e-8
            or abs(episode.terminal_reward - dense_sum) > 1e-6
        )
        if has_independent_terminal and episode.terminal_reward != 0.0:
            chunk_actions = episode.actions[start:end]
            kept = chunk_actions.bool().nonzero(as_tuple=True)[0]
            reward = reward.clone()
            if kept.numel() > 0:
                reward[kept[-1]] += episode.terminal_reward
            else:
                reward[-1] += episode.terminal_reward
        return reward

    def _next_obs(
        self, episode: Episode, boundaries: list[int],
        idx: int, current_obs: Observation,
    ) -> Observation:
        """Build next-chunk observation, or dummy if terminal."""
        if idx == len(boundaries) - 1:
            return _dummy_obs(current_obs)
        return self._build_chunk_obs(
            episode, boundaries[idx + 1], idx + 1, len(boundaries),
        )

    def _decompose_episode(self, episode: Episode) -> list[Transition]:
        """Convert episode into chunk-level Transitions.

        Each chunk OWNS tokens from boundaries[i] to boundaries[i+1]
        (earlier-chunk-wins dedup), but SEES the full observation window
        from start to start+chunk_size. Actions/rewards are sliced to
        the owned range, not the full observation range.
        """
        boundaries = episode.chunk_boundaries or [0]
        actions_for_grad = episode.sampled_actions if episode.sampled_actions is not None else episode.actions
        seq_len = actions_for_grad.shape[0]
        transitions: list[Transition] = []

        for i in range(len(boundaries)):
            start = boundaries[i]
            obs_end = min(start + self._chunk_size, seq_len)
            # Deduplicate: only own tokens up to next boundary
            own_end = boundaries[i + 1] if i + 1 < len(boundaries) else seq_len
            action_end = min(own_end, obs_end)
            if start >= action_end:
                continue
            is_last = (i == len(boundaries) - 1)
            obs = self._build_chunk_obs(episode, start, i, len(boundaries))
            transitions.append(Transition(
                observation=obs,
                action=actions_for_grad[start:action_end].to(self._device),
                reward=self._chunk_reward(episode, start, action_end, is_last).to(self._device),
                next_observation=self._next_obs(episode, boundaries, i, obs),
                done=is_last,
            ))
        return transitions

    def _build_chunk_obs(
        self, episode: Episode, start: int,
        chunk_idx: int, total_chunks: int,
    ) -> Observation:
        """Reconstruct full overlapping Observation for a chunk.

        Uses chunk_size to restore the original context window, not the
        non-overlapping boundary slice. Ratio matches env's counting logic.
        """
        seq_len = episode.prompt.token_ids.shape[0]
        obs_end = min(start + self._chunk_size, seq_len)
        ratio_so_far = self._ratio_before_chunk(episode, start, seq_len)
        obs = Observation(
            token_ids=episode.prompt.token_ids[start:obs_end],
            attention_mask=episode.prompt.attention_mask[start:obs_end],
            position_ids=torch.arange(start, obs_end, device=self._device),
            compression_ratio_so_far=ratio_so_far,
            target_compression_ratio=self._target_ratio,
            chunk_index=chunk_idx,
            total_chunks=total_chunks,
        )
        return obs.to(self._device)

    def _ratio_before_chunk(
        self, episode: Episode, start: int, seq_len: int,
    ) -> float:
        """Compute compression_ratio_so_far matching env's counting logic."""
        if start == 0:
            return 0.0
        seen_end = min(start + self._overlap, seq_len)
        actions_for_grad = episode.sampled_actions if episode.sampled_actions is not None else episode.actions
        return actions_for_grad[:seen_end].float().mean().item()

    def _compute_td_loss(
        self, batch: list[Transition],
    ) -> tuple[Tensor, float]:
        """Compute Huber loss over a batch of transitions.

        NOTE: unbatched — one forward pass per transition. Batching requires
        padding variable-length chunks; optimize if this becomes a bottleneck.
        """
        total_loss = torch.tensor(0.0, device=self._device)
        total_q = 0.0

        for t in batch:
            q_values = self.policy(t.observation)
            action_len = t.action.shape[0]
            q_taken = q_values[0, :action_len].gather(
                1, t.action.unsqueeze(-1),
            ).squeeze(-1)
            td_target = self._compute_td_target(t)
            total_loss = total_loss + nn.functional.huber_loss(
                q_taken, td_target.detach(), reduction="mean",
            )
            total_q += q_taken.mean().item()

        return total_loss / len(batch), total_q / len(batch)

    def _compute_td_target(self, transition: Transition) -> Tensor:
        """Per-token reward + gamma * chunk-mean bootstrap from next state.

        Bootstrap is the mean of max-Q across next-chunk tokens because chunks
        have variable length, so per-token alignment is not possible. Each token
        gets its own reward but a shared future-value estimate.
        """
        if transition.done:
            return transition.reward

        with torch.no_grad():
            next_q = self.target_policy(transition.next_observation)
        max_next_q = next_q[0].max(dim=-1).values.mean()
        return transition.reward + self.config.gamma * max_next_q

    def state_dict(self) -> dict[str, Any]:
        return {
            "q_network": self.policy.state_dict(),
            "target_network": self.target_policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self._step,
            "buffer": self.buffer.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.policy.load_state_dict(state["q_network"])
        self.target_policy.load_state_dict(state["target_network"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._step = state["step"]
        if "buffer" in state:
            self.buffer.load_state_dict(state["buffer"])


def _dummy_obs(obs: Observation) -> Observation:
    """Create an empty terminal observation on the same device as obs."""
    device = obs.token_ids.device
    return Observation(
        token_ids=torch.tensor([], dtype=torch.long, device=device),
        attention_mask=torch.tensor([], dtype=torch.long, device=device),
        position_ids=torch.tensor([], dtype=torch.long, device=device),
        compression_ratio_so_far=obs.compression_ratio_so_far,
        target_compression_ratio=obs.target_compression_ratio,
        chunk_index=obs.chunk_index + 1,
        total_chunks=obs.total_chunks,
    )
