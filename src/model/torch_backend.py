import torch
import torch.nn as nn
from torch.profiler import record_function

from src.config import ModelConfig


class TorchMoEBackend:
    def __init__(self, cfg: ModelConfig, experts: nn.ModuleList, router: nn.Linear):
        self.cfg = cfg
        self.router = router
        self.experts = experts

    def forward(
        self,
        x: torch.Tensor,
    ):
        batch, seq_len, d_model = x.shape
        # flatten residual stream tokens into a 2D tensor of shape (num_tokens, d_model)
        flat_tokens = x.reshape(-1, d_model)

        # get the expert logits
        router_dtype = self.router.weight.dtype
        with record_function("moe/router"):
            expert_logits = self.router(flat_tokens.to(router_dtype))

        with record_function("moe/topK"):
            expert_probs = expert_logits.softmax(dim=-1)
            topk_weights, topk_expert_idx = torch.topk(expert_probs, dim=-1, k=self.cfg.top_k)
        expert_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        assert expert_weights.dtype == torch.float32, "Expert topk weights should be in fp32"

        tokens_per_expert = torch.bincount(
            topk_expert_idx.reshape(-1), minlength=self.cfg.num_experts
        )
        expert_offsets = torch.empty(self.cfg.num_experts + 1, dtype=torch.long, device=x.device)
        expert_offsets[0] = 0
        expert_offsets[1:] = torch.cumsum(tokens_per_expert, dim=0)

        total_assignments = flat_tokens.shape[0] * self.cfg.top_k
        packed_X = torch.empty(
            (total_assignments, d_model), dtype=flat_tokens.dtype, device=x.device
        )
        packed_token_ids = torch.empty(total_assignments, dtype=torch.long, device=x.device)
        packed_weights = torch.empty(total_assignments, dtype=expert_weights.dtype, device=x.device)

        for expert in range(self.cfg.num_experts):
            indices = torch.argwhere(topk_expert_idx == expert)
            if indices.numel() == 0:
                continue
            start = expert_offsets[expert].item()
            end = expert_offsets[expert + 1].item()
            packed_X[start:end] = flat_tokens[indices[:, 0]]
            packed_token_ids[start:end] = indices[:, 0]
            packed_weights[start:end] = expert_weights[indices[:, 0], indices[:, 1]]

        packed_outputs = self.experts(packed_X, expert_offsets)
        pool = torch.zeros_like(flat_tokens)
        weighted_outputs = (packed_outputs * packed_weights.unsqueeze(1)).to(pool.dtype)
        pool.index_add_(0, packed_token_ids, weighted_outputs)

        return (pool.reshape(batch, seq_len, d_model), tokens_per_expert)
