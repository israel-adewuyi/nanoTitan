import torch
import torch.nn as nn
from torch.profiler import record_function

import random_ext
from src.config import ModelConfig
from src.model.moe_ops import combine_tokens_fn, pack_tokens_fn
from src.model.utils import MoELayerStats


class CUDAMoEBackend:
    def __init__(self, cfg: ModelConfig, experts: nn.ModuleList, router: nn.Linear):
        self.cfg = cfg
        self.router = router
        self.experts = experts

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, MoELayerStats]:
        batch, seq_len, d_model = x.shape
        num_tokens = batch * seq_len
        # flatten residual stream tokens into a 2D tensor of shape (num_tokens, d_model)
        flat_tokens = x.reshape(-1, d_model)

        # get the expert logits
        router_dtype = self.router.weight.dtype
        with record_function("moe/router"):
            expert_logits = self.router(
                flat_tokens.to(router_dtype)
            )  # cast to fp32 (or whatever dtype router is)

        with record_function("moe/topK"):
            expert_probs = expert_logits.softmax(
                dim=-1
            )  # also in fp32 (or whatever dtype router is)
            topk_weights, topk_expert_idx = torch.topk(expert_probs, dim=-1, k=self.cfg.top_k)
            expert_weights = topk_weights / topk_weights.sum(
                dim=-1, keepdim=True
            )  # also in fp32 (or whatever dtype router is)

        # TODO: Will experimentally validate later. But this seems like the intuitive solution
        moe_aux_logits = self.router(flat_tokens.detach().to(router_dtype))
        moe_aux_probs = moe_aux_logits.softmax(dim=-1)

        assert expert_weights.dtype == torch.float32, "Expert topk weights should be in fp32"

        with record_function("moe/count_expert"):
            mask = torch.ones(num_tokens, device=x.device, dtype=torch.int32)
            expert_count = random_ext.count_expert_kernel(
                topk_expert_idx.to(torch.int32), mask, self.cfg.num_experts, self.cfg.top_k
            )

            expert_offsets = torch.empty(
                self.cfg.num_experts + 1, device=x.device, dtype=torch.int32
            )
            expert_offsets[0] = 0
            expert_offsets[1:] = torch.cumsum(expert_count, dim=0)
            expert_offsets_cpy = expert_offsets.clone()

        with record_function("moe/pack_tokens"):
            packed_X, packed_tokenId, _, packed_topk_weights = pack_tokens_fn(
                flat_tokens,
                expert_weights,
                topk_expert_idx.to(torch.int32),
                expert_offsets_cpy,
            )

        with record_function("moe/expert_compute"):
            packed_expert_outputs = self.experts(packed_X, expert_offsets)

        with record_function("moe/combine_tokens"):
            pool = combine_tokens_fn(
                packed_expert_outputs, packed_tokenId, packed_topk_weights, num_tokens, d_model
            ).to(dtype=packed_expert_outputs.dtype)

        moe_stats = MoELayerStats(
            tokens_per_expert=expert_count.detach(), probs_per_expert=moe_aux_probs, cfg=self.cfg
        )

        return (pool.reshape(batch, seq_len, d_model), moe_stats)
