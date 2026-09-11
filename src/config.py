from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    model_validator,
)

DTypeName = Literal["float16", "float32", "bfloat16"]


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    dataset_path: Path | None = None


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
    # dp specific args
    bucket_size: int = 25

    # PP specific args
    num_microbatches: int = 1
    pipeline_schedule: Literal["gpipe", "1f1b"] = "gpipe"
    activation_checkpointing: bool = False

    dp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    num_expert: PositiveInt = 1

    @model_validator(mode="after")
    def validate_microbatches_for_pipeline_size(self) -> RuntimeConfig:
        if self.pp_size == 1:
            if self.num_microbatches != 1:
                raise ValueError("runtime.num_microbatches must be 1 when runtime.pp_size is 1")
            if self.activation_checkpointing:
                raise ValueError(
                    "runtime.activation_checkpointing must be false when runtime.pp_size is 1"
                )
        return self


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


class HardwareConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    peak_flops_tflops_per_gpu: float = 0.0


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vocab_size: PositiveInt
    d_model: PositiveInt
    d_head: PositiveInt
    n_heads: PositiveInt
    n_layers: PositiveInt
    max_seq_len: PositiveInt
    ffn_in: PositiveInt
    num_experts: PositiveInt
    top_k: PositiveInt
    capacity_factor: float = Field(default=1.25, ge=1.0)
    token_embedding: TokenEmbeddingConfig = Field(default_factory=TokenEmbeddingConfig)
    positional_embedding: PositionalEmbeddingConfig = Field(
        default_factory=PositionalEmbeddingConfig
    )
    moe_backend: str = "torch"
    dtype: DTypeName = "float32"
    moe_router_dtype: DTypeName = "float32"
    router_alpha: float
    moe_bias_coef: float

    @model_validator(mode="after")
    def validate_embedding_compatibility(self) -> ModelConfig:
        if self.d_model % 2 != 0:
            raise ValueError(
                "model.d_model must be even when using sinusoidal positional embeddings"
            )
        return self


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    val_interval: int
    num_val_batches: PositiveInt
    per_device_batch_size: PositiveInt
    seed: PositiveInt
    grad_norm: float

    @model_validator(mode="after")
    def validate_val_interval(self) -> TrainerConfig:
        if self.val_interval == 0 or self.val_interval < -1:
            raise ValueError("trainer.val_interval must be -1 (disabled) or a positive integer")
        return self


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_name: str
    max_steps: PositiveInt
    model: ModelConfig
    trainer: TrainerConfig
    data: DataConfig
    optim: OptimizerConfig
    runtime: RuntimeConfig
    profiler: ProfilerConfig = Field(default_factory=ProfilerConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    track_backward_time: bool = False

    @model_validator(mode="before")
    @classmethod
    def extend_runtime_args(cls, config: object) -> object:
        if not isinstance(config, dict):
            return config

        model_config = config.get("model")
        runtime_config = config.get("runtime")
        if not isinstance(model_config, dict) or not isinstance(runtime_config, dict):
            return config

        extended_config = config.copy()
        extended_runtime_config = runtime_config.copy()
        extended_runtime_config["num_expert"] = model_config.get("num_experts")
        extended_config["runtime"] = extended_runtime_config
        return extended_config


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
