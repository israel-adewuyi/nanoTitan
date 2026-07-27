from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from torch.nn.utils import clip_grads_with_norm_, get_total_norm

from src.config import AppConfig, ModelConfig, TrainerConfig
from src.model.model import NanoTitanModel
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
    ep_size: int = 1
    ep_group: ProcessGroup | None = None


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
        ep_group=dim.ep_group,
    )

    return spec


def compute_grad_norm(model: NanoTitanModel, dims: ParallelDims) -> torch.Tensor:
    expert_params, nonexpert_params = [], []
    expert_param_name_slug = ["W_gate", "W_val", "W_out"]

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        found = any(pname in name for pname in expert_param_name_slug)
        if found:
            expert_params.append(param.grad)
        else:
            nonexpert_params.append(param.grad)

    ep_squared_norm = get_total_norm(expert_params) ** 2
    nonep_squared_norm = get_total_norm(nonexpert_params) ** 2

    dist.all_reduce(ep_squared_norm, op=dist.ReduceOp.SUM, group=dims.ep_group)
    dist.all_reduce(ep_squared_norm, op=dist.ReduceOp.SUM, group=dims.pp_group)
    dist.all_reduce(nonep_squared_norm, op=dist.ReduceOp.SUM, group=dims.pp_group)

    global_norm = torch.sqrt(ep_squared_norm + nonep_squared_norm)
    assert isinstance(global_norm, torch.Tensor)
    return global_norm


def clip_gradients(model: NanoTitanModel, grad_norm: torch.Tensor, cfg: TrainerConfig):
    clip_grads_with_norm_(model.parameters(), max_norm=cfg.grad_norm, total_norm=grad_norm)


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
