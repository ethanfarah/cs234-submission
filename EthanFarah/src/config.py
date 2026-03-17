"""Flat configuration dataclass with argparse CLI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field


@dataclass
class LlmConfig:
    """Configuration for the frozen LLM used for generation and logit extraction."""

    model_name: str = "meta-llama/Llama-3.1-8B"
    max_new_tokens: int = 16
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    quantize: bool = True


@dataclass
class Config:
    """Single flat config for the REINFORCE/MCTS + DistilRoBERTa pipeline."""

    # Data
    max_prompt_tokens: int = 1024
    chunk_size: int = 128
    chunk_overlap: int = 16
    train_split: str = "train"
    val_split: str = "validation"
    max_train_samples: int | None = None
    max_val_samples: int | None = None

    # Policy
    ratio_conditioned: bool = False

    # Reward (Hybrid = KL-dense + sparse)
    kl_coeff: float = 0.01
    target_compression_ratio: float = 0.5
    sparse_reward_mode: str = "continuous"
    compression_penalty: float = 2.0
    quality_threshold: float = 0.9
    failure_penalty: float = 0.01
    faithfulness_metric: str = "rougeL"
    dense_ce_coeff: float = 0.0

    # Algorithm
    lr: float = 3e-5
    entropy_coeff: float = 0.001
    max_grad_norm: float = 5.0
    algorithm_type: str = "reinforce"
    k_samples: int = 8
    accumulation_steps: int = 16

    # MCTS
    num_simulations: int = 64
    c_puct: float = 2.0
    num_action_samples: int = 8
    mcts_temperature: float = 1.5
    faithfulness_ema_decay: float = 0.95
    value_lr: float = 1e-3
    value_grad_clip: float = 1.0
    value_warmup_episodes: int = 50

    # Policy selection
    policy_type: str = "distilroberta"
    policy_model_name: str = "distilroberta-base"
    head_type: str = "mlp"

    # Training
    num_episodes: int = 10000
    eval_every: int = 500
    checkpoint_every: int = 1000
    log_every: int = 50
    wandb_project: str | None = None
    wandb_entity: str | None = None
    device: str = "cuda"
    seed: int = 42
    shuffle: bool = False
    output_dir: str = "outputs"
    resume_from: str | None = None
    policy_checkpoint: str | None = None
    experiment_name: str = "default"
    debug_log: str | None = None

    # LLM (nested — genuinely different concern)
    llm: LlmConfig = field(default_factory=LlmConfig)


def parse_args() -> Config:
    """Parse CLI arguments into a Config."""
    parser = argparse.ArgumentParser(description="RL Prompt Compression (REINFORCE/MCTS)")

    # Data
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--chunk-overlap", type=int, default=16)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)

    # Policy
    parser.add_argument("--ratio-conditioned", action="store_true")

    # Reward
    parser.add_argument("--kl-coeff", type=float, default=0.01)
    parser.add_argument("--target-compression-ratio", type=float, default=0.5)
    parser.add_argument("--sparse-reward-mode", default="continuous",
                        choices=["continuous", "multiplicative", "soft_gated", "harmonic"])
    parser.add_argument("--compression-penalty", type=float, default=2.0)
    parser.add_argument("--quality-threshold", type=float, default=0.9)
    parser.add_argument("--failure-penalty", type=float, default=0.01)
    parser.add_argument("--faithfulness-metric", default="rougeL",
                        choices=["rougeL", "f1"])
    parser.add_argument("--dense-ce-coeff", type=float, default=0.0,
                        help="Coefficient for teacher-forced cross-entropy dense reward")

    # Algorithm
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--entropy-coeff", type=float, default=0.001)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--algorithm-type", default="reinforce",
                        choices=["reinforce", "mcts"])
    parser.add_argument("--k-samples", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--num-simulations", type=int, default=64)
    parser.add_argument("--c-puct", type=float, default=2.0)
    parser.add_argument("--num-action-samples", type=int, default=8)
    parser.add_argument("--mcts-temperature", type=float, default=1.5)
    parser.add_argument("--faithfulness-ema-decay", type=float, default=0.95)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--value-grad-clip", type=float, default=1.0)
    parser.add_argument("--value-warmup-episodes", type=int, default=50)
    parser.add_argument("--policy-type", default="distilroberta",
                        choices=["distilroberta"])
    parser.add_argument("--policy-model-name", default="distilroberta-base")
    parser.add_argument("--head-type", type=str, default="mlp", choices=["mlp", "attention"])

    # Training
    parser.add_argument("--num-episodes", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--policy-checkpoint", default=None,
                        help="Path to pre-trained policy head checkpoint (.pt)")
    parser.add_argument("--experiment-name", default="default")
    parser.add_argument("--debug-log", default=None,
                        help="Path to debug log file for detailed episode logging")

    # LLM
    parser.add_argument("--model-name", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--quantize", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()

    llm = LlmConfig(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        quantize=args.quantize,
    )

    return Config(
        max_prompt_tokens=args.max_prompt_tokens,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        train_split=args.train_split,
        val_split=args.val_split,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        ratio_conditioned=args.ratio_conditioned,
        kl_coeff=args.kl_coeff,
        target_compression_ratio=args.target_compression_ratio,
        sparse_reward_mode=args.sparse_reward_mode,
        compression_penalty=args.compression_penalty,
        quality_threshold=args.quality_threshold,
        failure_penalty=args.failure_penalty,
        faithfulness_metric=args.faithfulness_metric,
        dense_ce_coeff=args.dense_ce_coeff,
        lr=args.lr,
        entropy_coeff=args.entropy_coeff,
        max_grad_norm=args.max_grad_norm,
        algorithm_type=args.algorithm_type,
        k_samples=args.k_samples,
        accumulation_steps=args.accumulation_steps,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        num_action_samples=args.num_action_samples,
        mcts_temperature=args.mcts_temperature,
        faithfulness_ema_decay=args.faithfulness_ema_decay,
        value_lr=args.value_lr,
        value_grad_clip=args.value_grad_clip,
        value_warmup_episodes=args.value_warmup_episodes,
        policy_type=args.policy_type,
        policy_model_name=args.policy_model_name,
        head_type=args.head_type,
        num_episodes=args.num_episodes,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        log_every=args.log_every,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        device=args.device,
        seed=args.seed,
        shuffle=args.shuffle,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
        policy_checkpoint=args.policy_checkpoint,
        experiment_name=args.experiment_name,
        debug_log=args.debug_log,
        llm=llm,
    )
