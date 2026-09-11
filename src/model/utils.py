import math
from dataclasses import dataclass

import torch
from torch.distributed import ProcessGroup

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
    ep_size: int = 1
    ep_group: ProcessGroup | None = None
    non_expert_dp_group: ProcessGroup | None = None


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
        non_expert_dp_group=dim.shared_dp_group,
    )

    return spec


@torch.no_grad()
def max_violation(tokens_per_expert: torch.Tensor) -> torch.Tensor:
    """
    Metric to measure the degree of load imbalance in a MoE layer
    https://arxiv.org/pdf/2408.15664
    """
    counts = tokens_per_expert.float()
    optimal_load = counts.mean(dim=-1)
    return (counts.amax(dim=-1) - optimal_load) / optimal_load


@dataclass
class MoELayerStats:
    tokens_per_expert: torch.Tensor
    cfg: ModelConfig
    aux_loss: torch.Tensor | None = None

    def __post_init__(
        self,
    ):
        self.total_assignments = self.tokens_per_expert.sum()
        self.ass_frac_per_expert = self.tokens_per_expert.float() / self.total_assignments

        assert self.ass_frac_per_expert.ndim == 1

        self.max_vio = max_violation(self.tokens_per_expert)


def topk_with_capacity(
    expert_logits: torch.Tensor,
    top_k: int,
    capacity_factor: float = 1.25,
):
    """
    expert_logits: [N, E], attached to autograd.

    Returns:
        weights:       [N, K], attached
        final_idx:     [N, K], discrete
        final_count:   [E]
        raw_count:     [E]
    """
    N, E = expert_logits.shape

    capacity = math.ceil(capacity_factor * N * top_k / E)
    with torch.no_grad():
        raw_idx = torch.topk(
            expert_logits.detach(),
            k=top_k,
            dim=-1,
        ).indices  # [N, K]

        raw_count = torch.bincount(
            raw_idx.flatten(),
            minlength=E,
        )

        # Rank all experts for each token.
        ranked_idx = torch.argsort(
            expert_logits.detach(),
            dim=-1,
            descending=True,
        )  # [N, E]

        ranked_logits = expert_logits.detach().gather(1, ranked_idx)

        final_idx = torch.full(
            (N, top_k),
            -1,
            dtype=torch.long,
            device=expert_logits.device,
        )

        # How many assignments each token already has.
        token_slots = torch.zeros(
            N,
            dtype=torch.long,
            device=expert_logits.device,
        )

        # Current capacity usage.
        final_count = torch.zeros(
            E,
            dtype=torch.long,
            device=expert_logits.device,
        )

        # Try 1st choice, then 2nd, then 3rd, ...
        for candidate_rank in range(E):
            candidate_expert = ranked_idx[:, candidate_rank]
            candidate_score = ranked_logits[:, candidate_rank]

            for expert_id in range(E):
                remaining = capacity - int(final_count[expert_id].item())

                if remaining <= 0:
                    continue

                eligible = (token_slots < top_k) & (candidate_expert == expert_id)

                tokens = eligible.nonzero(as_tuple=False).flatten()

                if tokens.numel() == 0:
                    continue

                # Expert is oversubscribed:
                # keep tokens that wanted it most strongly.
                if tokens.numel() > remaining:
                    scores = candidate_score[tokens]

                    keep = torch.topk(
                        scores,
                        k=remaining,
                        sorted=False,
                    ).indices

                    tokens = tokens[keep]

                slots = token_slots[tokens]

                final_idx[tokens, slots] = expert_id
                token_slots[tokens] += 1
                final_count[expert_id] += tokens.numel()

        if not torch.all(token_slots == top_k):
            raise RuntimeError(
                f"Some tokens failed to obtain {top_k} experts. "
                f"min slots={token_slots.min().item()}"
            )

    selected_logits = expert_logits.gather(
        dim=1,
        index=final_idx,
    )

    weights = selected_logits.softmax(dim=-1)

    return (
        weights,
        final_idx.to(torch.int32),
        final_count.to(torch.int32),
        raw_count.to(torch.int32),
    )
