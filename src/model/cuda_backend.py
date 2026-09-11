import torch
import torch.distributed as dist
import torch.nn as nn
from torch.profiler import record_function

from src.config import ModelConfig
from src.model.cuda_extension import get_cuda_extension
from src.model.moe_ops import (
    combine_tokens_fn,
    pack_tokens_fn,
    permute_expert_assignment_fn,
    torch_backend_all_to_all,
)
from src.model.utils import ModelShardSpec, MoELayerStats


class CUDAMoEBackend:
    def __init__(
        self, cfg: ModelConfig, experts: nn.ModuleList, router: nn.Linear, spec: ModelShardSpec
    ):
        self.cfg = cfg
        self.spec = spec
        self.router = router
        self.experts = experts

    def forward(
        self, x: torch.Tensor, expert_bias: torch.Tensor
    ) -> tuple[torch.Tensor, MoELayerStats, torch.Tensor]:
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

        # This implementation only works for 1x DP i.e assumption that per_rank_expert == num_experts.
        with record_function("moe/topK"):
            affinity_scores = expert_logits.sigmoid()
            topk_expert_idx = (
                (affinity_scores + expert_bias).topk(self.cfg.top_k, dim=-1).indices
            )  # [N, K]
            weights = affinity_scores.gather(dim=-1, index=topk_expert_idx)  # [N, K]
            expert_weights = weights / weights.sum(dim=-1, keepdim=True)  # [N, K]

        assert expert_weights.dtype == torch.float32, "Expert topk weights should be in fp32"

        with record_function("moe/count_expert"):
            mask = torch.ones(num_tokens, device=x.device, dtype=torch.int32)

            # Lazily import cuda extension
            nanotitan_cuda = get_cuda_extension()
            expert_count = nanotitan_cuda.count_expert_kernel(
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

        if self.spec.ep_size == 1:
            expert_inputs = packed_X
            local_offsets = expert_offsets
        else:
            send_matrix = expert_count.view(self.spec.ep_size, self.spec.per_rank_expert)
            recv_matrix = torch.empty_like(send_matrix)

            dist.all_to_all_single(
                recv_matrix,
                send_matrix,
                group=self.spec.ep_group,
            )

            send_splits = send_matrix.sum(dim=1)
            recv_splits = recv_matrix.sum(dim=1)
            send_split_sizes = send_splits.tolist()
            recv_split_sizes = recv_splits.tolist()

            local_counts = recv_matrix.sum(dim=0)
            local_offsets = torch.empty(
                self.spec.per_rank_expert + 1, dtype=torch.int32, device=x.device
            )
            local_offsets[0] = 0
            local_offsets[1:] = torch.cumsum(local_counts, dim=0)

            receiver_X = torch_backend_all_to_all(
                packed_X,
                input_splits=send_split_sizes,
                output_splits=recv_split_sizes,
                group=self.spec.ep_group,
            )

            expert_inputs = permute_expert_assignment_fn(receiver_X, recv_matrix)

        with record_function("moe/expert_compute"):
            packed_expert_outputs = self.experts(expert_inputs, local_offsets)

        if self.spec.ep_size == 1:
            returned_outputs = packed_expert_outputs
        else:
            source_outputs = permute_expert_assignment_fn(
                packed_expert_outputs, recv_matrix.T.contiguous()
            )
            returned_outputs = torch_backend_all_to_all(
                source_outputs,
                input_splits=recv_split_sizes,
                output_splits=send_split_sizes,
                group=self.spec.ep_group,
            )

        with record_function("moe/combine_tokens"):
            pool = combine_tokens_fn(
                returned_outputs, packed_tokenId, packed_topk_weights, num_tokens, d_model
            ).to(dtype=returned_outputs.dtype)

        moe_stats = MoELayerStats(
            tokens_per_expert=expert_count.detach(),
            cfg=self.cfg,
        )

        # Adjust the bias in case some experts were overloaded/underloaded
        with torch.no_grad():
            dist.all_reduce(expert_count, op=dist.ReduceOp.SUM, group=self.spec.non_expert_dp_group)
            counts = expert_count.float()
            optimal_load = counts.mean()
            expert_bias = expert_bias + self.cfg.moe_bias_coef * (optimal_load - counts).sign()

        return (pool.reshape(batch, seq_len, d_model), moe_stats, expert_bias)
