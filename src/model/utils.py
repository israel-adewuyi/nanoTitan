from dataclasses import dataclass

import torch

from src.config import AppConfig, ModelConfig
from src.parallel_dims import ParallelDims


@dataclass(frozen=True)
class ModelShardSpec:
    layer_start: int
    layer_end: int
    has_token_embed: bool
    has_pos_embed: bool
    has_unembed_head: bool
    per_rank_expert: int
    start_expert_id: int
    end_expert_id: int
    ep_size: int


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


def get_logical_expert_bounds(ep_rank: int, num_per_rank_experts: int):
    """
    Each EP group maps to a subset of experts.
    Each EP rank holds a subset of experts and we return the indices of the experts.
    """
    start_expert_id = ep_rank * num_per_rank_experts
    end_expert_id = start_expert_id + num_per_rank_experts
    return (start_expert_id, end_expert_id)


def get_model_shard_specs(dim: ParallelDims, cfg: AppConfig):
    has_token_embed = dim.is_pp_first_stage
    has_pos_embed = dim.is_pp_first_stage
    has_unembed_head = dim.is_pp_last_stage
    layer_start, layer_end = get_layer_bounds(cfg, dim.pp_rank)
    num_per_rank_experts = dim.num_experts // dim.ep_size
    start_expert_id, end_expert_id = get_logical_expert_bounds(dim.ep_rank, num_per_rank_experts)

    spec = ModelShardSpec(
        has_token_embed=has_token_embed,
        has_pos_embed=has_pos_embed,
        has_unembed_head=has_unembed_head,
        layer_start=layer_start,
        layer_end=layer_end,
        per_rank_expert=num_per_rank_experts,
        start_expert_id=start_expert_id,
        end_expert_id=end_expert_id,
        ep_size=dim.ep_size,
    )

    return spec


@dataclass
class MoELayerStats:
    tokens_per_expert: torch.Tensor
    probs_per_expert: torch.Tensor
    cfg: ModelConfig
    aux_loss: torch.Tensor | None = None

    def __post_init__(
        self,
    ):
        self.num_tokens = self.probs_per_expert.shape[0]
        self.total_assignments = self.tokens_per_expert.sum()
        self.ass_frac_per_expert = self.tokens_per_expert.float() / self.total_assignments

        self.probs_per_expert = self.probs_per_expert.mean(dim=0)

        assert self.ass_frac_per_expert.shape == self.probs_per_expert.shape
        assert self.ass_frac_per_expert.ndim == 1

        self.aux_loss = (
            self.cfg.router_alpha
            * self.cfg.num_experts
            * torch.sum(self.ass_frac_per_expert * self.probs_per_expert)
        )
