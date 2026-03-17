"""Dataclasses for the MCTS tree structure."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

from src.data.types import Prompt


@dataclass
class MCTSConfig:
    """Configuration for MCTS search."""

    num_simulations: int = 32
    c_puct: float = 1.5
    num_action_samples: int = 8
    temperature: float = 1.2
    chunk_overlap: int = 0

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"MCTS temperature must be > 0, got {self.temperature}")
        if self.num_simulations <= 0:
            raise ValueError(f"num_simulations must be > 0, got {self.num_simulations}")
        if self.num_action_samples <= 0:
            raise ValueError(f"num_action_samples must be > 0, got {self.num_action_samples}")
        if self.c_puct < 0:
            raise ValueError(f"c_puct must be >= 0, got {self.c_puct}")


@dataclass
class MCTSState:
    """State at a node in the MCTS tree."""

    prompt: Prompt
    chunk_index: int
    actions_so_far: list[Tensor]
    total_chunks: int
    total_tokens: int
    chunk_overlap: int = 0

    @property
    def is_terminal(self) -> bool:
        return self.chunk_index >= self.total_chunks

    def kept_count(self) -> int:
        """Count kept tokens, skipping overlap from all chunks after the first."""
        total = 0
        for i, a in enumerate(self.actions_so_far):
            skip = self.chunk_overlap if i > 0 else 0
            total += int(a[skip:].sum().item())
        return total

    def seen_count(self) -> int:
        """Count unique tokens seen, skipping overlap from all chunks after the first."""
        total = 0
        for i, a in enumerate(self.actions_so_far):
            skip = self.chunk_overlap if i > 0 else 0
            total += a.numel() - skip
        return total


@dataclass
class MCTSNode:
    """Node in the MCTS search tree."""

    state: MCTSState
    parent: MCTSNode | None
    action_index: int
    children: dict[int, MCTSNode] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    prior: float = 1.0  # normalized across siblings; unused for root node

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    @property
    def is_expanded(self) -> bool:
        return len(self.children) > 0 or self.state.is_terminal
