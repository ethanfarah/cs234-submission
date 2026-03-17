"""Core MCTS search: select, expand, evaluate, backup."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from src.algo.mcts_types import MCTSConfig, MCTSNode, MCTSState
from src.algo.value_fn import HeuristicValue, LearnedValue
from src.data.types import Prompt
from src.env.spaces import Observation
from src.policy.base import Policy


def run_mcts(
    chunks: list[Observation],
    prompt: Prompt,
    policy: Policy,
    value_fn: HeuristicValue | LearnedValue,
    config: MCTSConfig,
    device: str = "cpu",
) -> list[Tensor]:
    """Run MCTS search, return one action tensor per chunk."""
    root_state = MCTSState(
        prompt=prompt,
        chunk_index=0,
        actions_so_far=[],
        total_chunks=len(chunks),
        # Use LLM token count when available — reward ratio is in LLM space
        total_tokens=(prompt.llm_token_ids.shape[0] if prompt.llm_token_ids is not None
                      else prompt.token_ids.shape[0]),
        chunk_overlap=config.chunk_overlap,
    )
    root = MCTSNode(state=root_state, parent=None, action_index=-1)

    for _ in range(config.num_simulations):
        node = _select(root, config.c_puct)
        if not node.state.is_terminal and not node.is_expanded:
            _expand(node, chunks, policy, config, device)
            # Evaluate highest-prior child (standard AlphaZero post-expansion)
            assert node.children, "BUG: _expand created no children"
            node = max(node.children.values(), key=lambda c: c.prior)
        value = value_fn.estimate(node.state)
        _backup(node, value)

    return _extract_actions(root)


def _select(node: MCTSNode, c_puct: float) -> MCTSNode:
    """Walk tree via PUCT to an unexpanded or terminal node."""
    while node.is_expanded and not node.state.is_terminal:
        node = max(
            node.children.values(),
            key=lambda c: _puct_score(c, node.visit_count, c_puct),
        )
    return node


def _puct_score(child: MCTSNode, parent_visits: int, c_puct: float) -> float:
    exploitation = child.q_value
    exploration = c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
    return exploitation + exploration


def _expand(
    node: MCTSNode,
    chunks: list[Observation],
    policy: Policy,
    config: MCTSConfig,
    device: str,
) -> None:
    """Sample B Bernoulli per-token keep/drop actions, create child nodes."""
    obs = chunks[node.state.chunk_index]
    obs_device = Observation(
        token_ids=obs.token_ids.to(device),
        attention_mask=obs.attention_mask.to(device),
        position_ids=obs.position_ids.to(device),
        compression_ratio_so_far=obs.compression_ratio_so_far,
        target_compression_ratio=obs.target_compression_ratio,
        chunk_index=obs.chunk_index,
        total_chunks=obs.total_chunks,
    )

    with torch.no_grad():
        logits = policy.forward(obs_device)  # (1, seq_len, 2)

    tempered = logits / config.temperature
    dist = torch.distributions.Categorical(logits=tempered)
    log_prob_sums: list[float] = []
    sampled_actions: list[Tensor] = []
    for _ in range(config.num_action_samples):
        actions = dist.sample()  # (1, seq_len) — 0=drop, 1=keep
        log_probs = dist.log_prob(actions)  # (1, seq_len)
        sampled_actions.append(actions[0].cpu())
        log_prob_sums.append(log_probs.sum().item())

    priors = _normalize_log_probs(log_prob_sums)

    for i, (acts, prior) in enumerate(zip(sampled_actions, priors)):
        child_state = MCTSState(
            prompt=node.state.prompt,
            chunk_index=node.state.chunk_index + 1,
            actions_so_far=node.state.actions_so_far + [acts],
            total_chunks=node.state.total_chunks,
            total_tokens=node.state.total_tokens,
            chunk_overlap=node.state.chunk_overlap,
        )
        node.children[i] = MCTSNode(
            state=child_state, parent=node, action_index=i, prior=prior,
        )


def _normalize_log_probs(log_probs: list[float]) -> list[float]:
    """Softmax over log-prob sums to get normalized priors."""
    max_lp = max(log_probs)
    exp_lps = [math.exp(lp - max_lp) for lp in log_probs]
    total = sum(exp_lps)
    return [e / total for e in exp_lps]


def _backup(node: MCTSNode, value: float) -> None:
    """Propagate value up to root."""
    while node is not None:
        node.visit_count += 1
        node.total_value += value
        node = node.parent


def _extract_actions(root: MCTSNode) -> list[Tensor]:
    """Extract best action at each depth by visit count."""
    actions: list[Tensor] = []
    node = root
    while not node.state.is_terminal and node.children:
        best_child = max(node.children.values(), key=lambda c: c.visit_count)
        actions.append(best_child.state.actions_so_far[-1])
        node = best_child
    return actions
