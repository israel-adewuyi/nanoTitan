import torch
import random_ext
import torch.nn as nn
from torch.profiler import record_function

from src.config import ModelConfig


class CUDAMoEBackend:
    def __init__(self, cfg: ModelConfig, experts: nn.ModuleList, router: nn.Linear):
        self.cfg = cfg
        self.router = router
        self.experts = experts

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        num_tokens = batch * seq_len
        # flatten residual stream tokens into a 2D tensor of shape (num_tokens, d_model)
        flat_tokens = x.reshape(-1, d_model)

        # get the expert logits
        with record_function("moe/router"):
            expert_logits = self.router(flat_tokens)

        with record_function("moe/topK"):
            expert_probs = expert_logits.softmax(dim=-1)
            topk_weights, topk_expert_idx = torch.topk(expert_probs, dim=-1, k=self.cfg.top_k)
        expert_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        total_assignments = num_tokens * self.cfg.top_k

        packed_X = torch.empty(
            total_assignments,
            d_model,
            device=x.device,
            dtype=x.dtype,
        )
        packed_expert = torch.empty(
            total_assignments,
            device=x.device,
            dtype=torch.int32,
        )
        packed_tokenId = torch.empty(
            total_assignments,
            device=x.device,
            dtype=torch.int32,
        )
        packed_topk_weights = torch.empty(
            total_assignments,
            device=x.device,
            dtype=x.dtype,
        )

        mask = torch.ones(num_tokens, device=x.device, dtype=torch.int32)
        expert_count = random_ext.count_expert_kernel(
            topk_expert_idx.to(torch.int32), mask, self.cfg.num_experts, self.cfg.top_k
        )

        expert_offsets = torch.empty(self.cfg.num_experts + 1, device=x.device, dtype=torch.int32)
        expert_offsets[0] = 0
        expert_offsets[1:] = torch.cumsum(expert_count, dim=0)
        expert_offsets_cpy = expert_offsets.clone()

        random_ext.pack_tokens_kernel(
            flat_tokens,
            expert_weights,
            topk_expert_idx.to(torch.int32),
            self.cfg.top_k,
            total_assignments,
            expert_offsets_cpy,
            packed_X,
            packed_tokenId,
            packed_expert,
            packed_topk_weights,
            d_model,
        )

        # Temporary Python expert compute. Replace this with grouped GEMM later.
        packed_expert_outputs = torch.empty_like(packed_X)
        for expert in range(self.cfg.num_experts):
            start = int(expert_offsets[expert].item())
            end = int(expert_offsets[expert + 1].item())
            if start == end:
                continue
            packed_expert_outputs[start:end] = self.experts[expert](packed_X[start:end])

        pool = random_ext.combine_tokens_kernel(
            packed_expert_outputs, packed_tokenId, packed_topk_weights, num_tokens, d_model
        ).to(dtype=packed_expert_outputs.dtype)

        return (pool.reshape(batch, seq_len, d_model), expert_count)
