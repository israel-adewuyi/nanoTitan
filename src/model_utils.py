from dataclasses import dataclass

from src.config import AppConfig
from src.parallel_dims import ParallelDims


@dataclass(frozen=True)
class ModelShardSpec:
    layer_start: int
    layer_end: int
    has_token_embed: bool
    has_pos_embed: bool
    has_unembed_head: bool


def get_layer_bounds(cfg: AppConfig, pp_rank: int):
    """
    Each PP stage maps to a subset of layers.
    We compute the layer bounds for each PP stage.
    The format is [start_layer, end_layer)
    """
    per_rank_layers = cfg.model.n_layers // cfg.runtime.pp_size
    start_idx = pp_rank * per_rank_layers
    end_idx = (pp_rank + 1) * per_rank_layers
    return (start_idx, end_idx)


def get_model_shard_specs(dim: ParallelDims, cfg: AppConfig):
    has_token_embed = dim.is_pp_first_stage
    has_pos_embed = dim.is_pp_first_stage
    has_unembed_head = dim.is_pp_last_stage
    layer_start, layer_end = get_layer_bounds(cfg, dim.pp_rank)

    spec = ModelShardSpec(
        has_token_embed=has_token_embed,
        has_pos_embed=has_pos_embed,
        has_unembed_head=has_unembed_head,
        layer_start=layer_start,
        layer_end=layer_end,
    )

    return spec
