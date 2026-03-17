"""Weights & Biases logger for training metrics and episode visualization."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import ExperimentConfig
    from src.data.types import Episode


class WandbLogger:
    """Logs training metrics and per-token episode tables to W&B."""

    def __init__(self, config: ExperimentConfig) -> None:
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb is required for logging. Install with: pip install wandb"
            ) from e
        wandb.init(
            project=config.train.wandb_project,
            entity=config.train.wandb_entity,
            name=config.experiment_name,
            config=dataclasses.asdict(config),
        )
        self._wandb = wandb
        self._llm_model_name = config.llm.model_name

    def log_metrics(
        self, metrics: dict[str, float], step: int, commit: bool = True
    ) -> None:
        self._wandb.log(metrics, step=step, commit=commit)

    def log_episode(self, episode: Episode, step: int, commit: bool = True) -> None:
        from src.data.tokenization import get_tokenizer

        tokenizer = get_tokenizer(self._llm_model_name)
        tokens = tokenizer.convert_ids_to_tokens(episode.prompt.token_ids.tolist())
        actions = episode.actions.tolist()
        table = self._wandb.Table(columns=["token", "action"])
        for tok, act in zip(tokens, actions):
            table.add_data(tok, int(act))
        self._wandb.log(
            {
                "episode/token_table": table,
                "episode/compression_ratio": episode.compressed.compression_ratio,
                "episode/terminal_reward": episode.terminal_reward,
            },
            step=step,
            commit=commit,
        )

    def finish(self) -> None:
        self._wandb.finish()
