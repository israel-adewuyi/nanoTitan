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

        # accumulate the weighted results of each expert per token
        pool = torch.zeros_like(flat_tokens)
        tokens_per_expert = torch.zeros((self.cfg.num_experts), dtype=torch.long, device=x.device)

        for expert in range(self.cfg.num_experts):
            indices = torch.argwhere(topk_expert_idx == expert)
            # if no token chose current expert, continue
            if indices.numel() == 0:
                continue
            # get the residual stream vector for each token by indexing into flat tokens with all tokens that chose current expert
            expert_toks = flat_tokens[indices[:, 0]]

            tokens_per_expert[expert] += expert_toks.shape[0]

            pool[indices[:, 0]] += self.experts[expert](expert_toks) * expert_weights[
                indices[:, 0], indices[:, 1]
            ].unsqueeze(1)

        return (pool.reshape(batch, seq_len, d_model), tokens_per_expert)
