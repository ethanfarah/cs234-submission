"""Typed configuration dataclasses with YAML loading via OmegaConf."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from omegaconf import OmegaConf


class PolicyArch(str, Enum):
    DISTILBERT = "distilbert"
    TINYLLAMA = "tinyllama"
    CUSTOM_TRANSFORMER = "custom_transformer"


class RewardType(str, Enum):
    SPARSE = "sparse"
    KL_DENSE = "kl_dense"
    HYBRID = "hybrid"
    LEARNED = "learned"


class AlgoType(str, Enum):
    REINFORCE = "reinforce"
    A2C = "a2c"
    PPO = "ppo"
    DQN = "dqn"
    BANDIT = "bandit"
    GRPO = "grpo"


class HybridMode(str, Enum):
    WEIGHTED = "weighted"
    THRESHOLD = "threshold"


class KLDirection(str, Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


class SparseRewardMode(str, Enum):
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    THRESHOLD = "threshold"


class BaselineType(str, Enum):
    NONE = "none"
    MOVING_AVERAGE = "moving_average"
    LEARNED = "learned"


@dataclass
class DataConfig:
    dataset: str = "squad"
    max_prompt_tokens: int = 1024
    chunk_size: int = 128
    chunk_overlap: int = 16
    seed: int = 42
    train_split: str = "train"
    val_split: str = "validation"
    max_train_samples: int | None = None
    max_val_samples: int | None = None


@dataclass
class PolicyConfig:
    arch: PolicyArch = PolicyArch.DISTILBERT
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 4
    pretrained_name: str = "distilbert-base-uncased"
    ratio_conditioned: bool = False
    dropout: float = 0.1
    causal: bool = False


@dataclass
class RewardConfig:
    reward_type: RewardType = RewardType.SPARSE
    kl_coeff: float = 0.01
    compression_bonus: float = 0.0
    compression_penalty: float = 2.0
    learned_model_path: str | None = None
    target_compression_ratio: float = 0.5
    hybrid_mode: HybridMode = HybridMode.WEIGHTED
    threshold_tau: float = 0.1  # per-token KL typically in [0.01, 3.0] nats; 0.1 ≈ 10th-20th percentile
    threshold_penalty: float = 0.01  # penalty applied when KL exceeds tau
    kl_direction: KLDirection = KLDirection.FORWARD
    sparse_reward_mode: SparseRewardMode = SparseRewardMode.MULTIPLICATIVE
    quality_threshold: float = 0.7
    failure_penalty: float = 0.1


@dataclass
class AlgoConfig:
    algo_type: AlgoType = AlgoType.PPO
    lr: float = 3e-4
    gamma: float = 1.0
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.05
    max_grad_norm: float = 1.0
    buffer_size: int = 10000
    batch_size: int = 32
    num_epochs: int = 4
    baseline_type: BaselineType = BaselineType.MOVING_AVERAGE
    target_update_freq: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 10000
    grpo_group_size: int = 8


@dataclass
class LlmConfig:
    """Configuration for the frozen LLM used for generation and logit extraction.

    temperature and top_p are only used when do_sample=True.
    Production configs should set quantize=True via YAML.
    """

    model_name: str = "meta-llama/Llama-3.1-8B"
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    quantize: bool = False


@dataclass
class TrainConfig:
    num_episodes: int = 10000
    eval_every: int = 500
    checkpoint_every: int = 1000
    log_every: int = 50
    wandb_project: str | None = "prompt-compression"
    wandb_entity: str | None = None
    device: str = "cuda"
    seed: int = 42
    output_dir: str = "outputs"
    resume_from: str | None = None
    early_stop_window: int = 0
    early_stop_entropy_min: float = 0.01
    early_stop_ratio_max: float = 0.95


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    experiment_name: str = "default"


def load_config(yaml_paths: list[str]) -> ExperimentConfig:
    """Load and merge multiple YAML config files into an ExperimentConfig."""
    base = OmegaConf.structured(ExperimentConfig)
    for path in yaml_paths:
        override = OmegaConf.load(path)
        base = OmegaConf.merge(base, override)
    schema = OmegaConf.to_object(base)
    return schema  # type: ignore[return-value]


def parse_args() -> ExperimentConfig:
    """Parse CLI arguments: --config file1.yaml file2.yaml ... + any dot-notation overrides."""
    parser = argparse.ArgumentParser(description="RL Prompt Compression")
    parser.add_argument(
        "--config",
        nargs="+",
        default=["configs/base.yaml"],
        help="YAML config files to merge (later files override earlier)",
    )
    args, overrides = parser.parse_known_args()
    overrides = [x for x in overrides if x != "--"]

    cfg = load_config(args.config)

    if overrides:
        cli_conf = OmegaConf.from_dotlist(overrides)
        merged = OmegaConf.merge(OmegaConf.structured(cfg), cli_conf)
        cfg = OmegaConf.to_object(merged)  # type: ignore[assignment]

    return cfg  # type: ignore[return-value]
