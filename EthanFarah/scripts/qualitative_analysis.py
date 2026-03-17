"""Visualize token-level keep/drop decisions from a trained compression policy."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

from src.config import parse_args
from src.data.tokenization import get_tokenizer
from src.data.types import Episode
from src.tracking.checkpointing import CheckpointState, load_checkpoint
from src.train import Components, collect_episode, init_components

ANSI_GREEN = "\033[92m"
ANSI_DIM = "\033[2m\033[91m"
ANSI_RESET = "\033[0m"


@dataclass
class AnalysisArgs:
    """Script-specific CLI arguments."""

    checkpoint: str
    num_examples: int
    pos_analysis: bool


def _parse_script_args() -> AnalysisArgs:
    """Parse script-specific args; leave remaining for parse_args()."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-examples", type=int, default=10)
    parser.add_argument("--pos-analysis", action="store_true")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    return AnalysisArgs(
        checkpoint=args.checkpoint,
        num_examples=args.num_examples,
        pos_analysis=args.pos_analysis,
    )


def _decode_tokens(token_ids: torch.Tensor, model_name: str) -> list[str]:
    """Decode each token ID to its string representation."""
    tokenizer = get_tokenizer(model_name)
    return [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]


def _colorize_tokens(tokens: list[str], actions: torch.Tensor) -> str:
    """Build ANSI-colored string: green=kept, dim red=dropped."""
    parts: list[str] = []
    for token, action in zip(tokens, actions):
        if action.item() == 1:
            parts.append(f"{ANSI_GREEN}{token}{ANSI_RESET}")
        else:
            parts.append(f"{ANSI_DIM}{token}{ANSI_RESET}")
    return "".join(parts)


def _compression_stats(actions: torch.Tensor) -> tuple[int, int, float]:
    """Return (kept, total, ratio) from binary action tensor."""
    total = actions.shape[0]
    kept = int(actions.sum().item())
    ratio = kept / total if total > 0 else 0.0
    return kept, total, ratio


def _print_example(
    idx: int, total: int, episode: Episode, model_name: str,
) -> None:
    """Print one colored example with compression stats."""
    tokens = _decode_tokens(episode.prompt.token_ids, model_name)
    kept, num_tokens, ratio = _compression_stats(episode.actions)
    print(f"\n--- Example {idx + 1}/{total} (ratio={ratio:.2f}, {kept}/{num_tokens} tokens kept) ---")
    print(_colorize_tokens(tokens, episode.actions))

    kept_tokens = [t for t, a in zip(tokens, episode.actions) if a.item() == 1]
    dropped_tokens = [t for t, a in zip(tokens, episode.actions) if a.item() == 0]
    print(f"\nOriginal:   {''.join(tokens)}")
    print(f"Compressed: {''.join(kept_tokens)}")
    print(f"Dropped:    {''.join(dropped_tokens)}")


def _collect_episodes(
    components: Components, device: str, num_examples: int,
) -> list[Episode]:
    """Run policy on validation prompts and collect episodes."""
    components.policy.eval()
    episodes: list[Episode] = []
    prompts = components.val_prompts[:num_examples]
    with torch.no_grad():
        for prompt in prompts:
            episode = collect_episode(
                components.env, components.policy, prompt, device=device,
            )
            episodes.append(episode)
    return episodes


@dataclass
class POSStats:
    """Aggregated keep/drop counts for a POS tag."""

    total: int = 0
    kept: int = 0

    @property
    def drop_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.total - self.kept) / self.total


def _build_special_token_set(model_name: str) -> set[str]:
    """Precompute the set of decoded special token strings."""
    tokenizer = get_tokenizer(model_name)
    return {tokenizer.decode([tid]) for tid in tokenizer.all_special_ids}


def _build_pos_table(
    episodes: list[Episode], model_name: str,
) -> dict[str, POSStats]:
    """Map each token to a POS tag via spaCy, aggregate keep/drop stats.

    Filters out special tokens (e.g. <|begin_of_text|>) before alignment
    to prevent character offset misalignment with spaCy.
    """
    try:
        import spacy
    except ImportError:
        raise ImportError("POS analysis requires spacy: pip install spacy && python -m spacy download en_core_web_sm")

    nlp = _load_spacy_model()
    special_tokens = _build_special_token_set(model_name)
    stats: dict[str, POSStats] = defaultdict(POSStats)

    for episode in episodes:
        tokens = _decode_tokens(episode.prompt.token_ids, model_name)
        actions = episode.actions
        non_special = [
            (tok, act) for tok, act in zip(tokens, actions)
            if tok not in special_tokens
        ]
        if not non_special:
            continue
        filtered_tokens, filtered_actions = zip(*non_special)
        pos_tags = _align_pos_tags(nlp, list(filtered_tokens))
        for tag, action in zip(pos_tags, filtered_actions):
            stats[tag].total += 1
            if action.item() == 1:
                stats[tag].kept += 1

    return dict(stats)


def _load_spacy_model() -> "spacy.Language":
    """Load the small English spaCy model."""
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        raise OSError("spaCy model not found: python -m spacy download en_core_web_sm")


def _align_pos_tags(nlp: "spacy.Language", tokens: list[str]) -> list[str]:
    """Assign a POS tag to each subword token by running spaCy on joined text.

    Uses character offset alignment: each subword token maps to the spaCy
    token that covers its starting character position.
    """
    text = "".join(tokens)
    doc = nlp(text)
    spacy_spans = [(token.idx, token.idx + len(token.text), token.pos_) for token in doc]
    return [_find_pos_at_offset(spacy_spans, offset) for offset in _token_offsets(tokens)]


def _token_offsets(tokens: list[str]) -> list[int]:
    """Compute the starting character offset for each token."""
    offsets: list[int] = []
    pos = 0
    for token in tokens:
        offsets.append(pos)
        pos += len(token)
    return offsets


def _find_pos_at_offset(
    spans: list[tuple[int, int, str]], char_offset: int,
) -> str:
    """Find the POS tag for the spaCy token covering a character offset."""
    for start, end, pos in spans:
        if start <= char_offset < end:
            return pos
    return "X"


def _print_pos_table(stats: dict[str, POSStats]) -> None:
    """Print POS tag drop-rate table sorted by drop rate descending."""
    sorted_tags = sorted(stats.items(), key=lambda kv: kv[1].drop_rate, reverse=True)
    print("\n=== POS Tag Analysis ===")
    print(f"{'POS Tag':10s} | {'Total':>5s} | {'Kept':>5s} | {'Drop Rate':>9s}")
    print(f"{'-' * 10}-|-{'-' * 5}-|-{'-' * 5}-|-{'-' * 9}")
    for tag, s in sorted_tags:
        print(f"{tag:10s} | {s.total:5d} | {s.kept:5d} | {s.drop_rate:8.1%}")


def main() -> None:
    """Load checkpoint, run policy on validation prompts, display results."""
    script_args = _parse_script_args()
    config = parse_args()
    torch.manual_seed(config.seed)

    components = init_components(config)
    step = load_checkpoint(
        CheckpointState(components.policy, components.algorithm, 0),
        Path(script_args.checkpoint),
    )
    print(f"Loaded checkpoint from step {step}", file=sys.stderr)

    episodes = _collect_episodes(components, config.device, script_args.num_examples)

    for idx, episode in enumerate(episodes):
        _print_example(idx, len(episodes), episode, config.llm.model_name)

    if script_args.pos_analysis:
        pos_stats = _build_pos_table(episodes, config.llm.model_name)
        _print_pos_table(pos_stats)


if __name__ == "__main__":
    main()
