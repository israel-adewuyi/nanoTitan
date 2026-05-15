from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_workers: PositiveInt
    train_tokens_path: str
    val_tokens_path: str


class OptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str
    lr: float


class TokenEmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PositionalEmbeddingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    # Literal[single, ddp_reference, ddp, pp_naive, pp_gpipe]
    reducer: str = "v0"
    # if ddp, options are v0 or v1
    bucket_size: int = 25

    # PP specific args
    num_microbatches: int = 1


class ProfilerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    wait_steps: NonNegativeInt = 2
    warmup_steps: NonNegativeInt = 2
    active_steps: NonNegativeInt = 6
    record_shapes: bool = True
    profile_memory: bool = True
    with_stack: bool = False
    with_flops: bool = False

    @model_validator(mode="after")
    def validate_active_steps(self) -> ProfilerConfig:
        if self.enabled and self.active_steps == 0:
            raise ValueError("profiler.active_steps must be positive when profiler.enabled=true")
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocab_size: PositiveInt
    d_model: PositiveInt
    d_head: PositiveInt
    n_heads: PositiveInt
    n_layers: PositiveInt
    max_seq_len: PositiveInt
    ffn_in: PositiveInt
    token_embedding: TokenEmbeddingConfig = Field(default_factory=TokenEmbeddingConfig)
    positional_embedding: PositionalEmbeddingConfig = Field(
        default_factory=PositionalEmbeddingConfig
    )

    @model_validator(mode="after")
    def validate_embedding_compatibility(self) -> ModelConfig:
        if self.d_model % 2 != 0:
            raise ValueError(
                "model.d_model must be even when using sinusoidal positional embeddings"
            )
        return self


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: NonNegativeInt
    eval_every_step: int
    per_device_batch_size: PositiveInt
    seed: PositiveInt


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_name: str
    model: ModelConfig
    trainer: TrainerConfig
    data: DataConfig
    optim: OptimizerConfig
    runtime: RuntimeConfig
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)

    track_backward_time: bool = False


def resolve_config_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    configs_candidate = Path("configs") / candidate
    if configs_candidate.exists():
        return configs_candidate

    raise FileNotFoundError(f"Could not find config file at '{path}' or '{configs_candidate}'")


def load_config(path: str | Path) -> AppConfig:
    config_path = resolve_config_path(path)
    with config_path.open("rb") as file:
        raw_config = tomllib.load(file)
    return AppConfig.model_validate(raw_config)
