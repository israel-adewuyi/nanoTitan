import torch
import torch.nn as nn
from jaxtyping import Float

from src.config import ModelConfig
from src.model.cuda_backend import CUDAMoEBackend
from src.model.torch_backend import TorchMoEBackend


class FFN(nn.Module):
    """Feedforward network implementation"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # SwiGLU uses two parallel input projections
        self.W_gate = nn.Linear(cfg.d_model, cfg.ffn_in)
        self.W_val = nn.Linear(cfg.d_model, cfg.ffn_in)
        self.W_out = nn.Linear(cfg.ffn_in, cfg.d_model)
        self.silu = nn.SiLU()

    def forward(
        self, x: Float[torch.Tensor, "batch seq_len d_model"]
    ) -> Float[torch.Tensor, "batch seq_len d_model"]:
        # SwiGLU: silu(gate) * value
        gate = self.silu(self.W_gate(x))
        value = self.W_val(x)
        ffn_out = gate * value  # element-wise multiplication

        return self.W_out(ffn_out)

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())


class MoE(nn.Module):
    """
    A Mixture of Experts layer
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleList(FFN(cfg) for _ in range(self.cfg.num_experts))
        self.router = nn.Linear(self.cfg.d_model, self.cfg.num_experts, bias=False)
        if cfg.moe_backend == "cuda":
            self.moe_backend = CUDAMoEBackend(cfg, self.experts, self.router)
        else:
            self.moe_backend = TorchMoEBackend(cfg, self.experts, self.router)

    def active_parameter_count(self) -> int:
        router_params = sum(param.numel() for param in self.router.parameters())
        expert_params = self.experts[0].parameter_count()
        return router_params + self.cfg.top_k * expert_params

    def forward(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> tuple:
        return self.moe_backend.forward(x)
        # batch, seq_len, d_model = x.shape
        # # flatten residual stream tokens into a 2D tensor of shape (num_tokens, d_model)
        # flat_tokens = x.reshape(-1, d_model)  # (num_tokens d_model)

        # # get the expert logits
        # with record_function("moe/router"):
        #     expert_logits = self.router(flat_tokens)

        # with record_function("moe/topK"):
        #     expert_probs = expert_logits.softmax(dim=-1)
        #     topk_weights, topk_expert_idx = torch.topk(expert_probs, dim=-1, k=self.cfg.top_k)
        # expert_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # # accumulate the weighted results of each expert per token
        # pool = torch.zeros_like(flat_tokens)
        # tokens_per_expert = torch.zeros((self.cfg.num_experts), dtype=torch.long, device=x.device)

        # for expert in range(self.cfg.num_experts):
        #     with record_function("moe/dispatch"):
        #         # Return a 2D tensor of which token chose the current expert and the index / position in the topK
        #         indices = torch.argwhere(topk_expert_idx == expert)
        #         # if no token chose current expert, continue
        #         if indices.numel() == 0:
        #             continue
        #         # get the residual stream vector for each token by indexing into flat tokens with all tokens that chose current expert
        #         expert_toks = flat_tokens[indices[:, 0]]
        #     tokens_per_expert[expert] += expert_toks.shape[0]

        #     with record_function("moe/expert_combine"):
        #         pool[indices[:, 0]] += self.experts[expert](expert_toks) * expert_weights[
        #             indices[:, 0], indices[:, 1]
        #         ].unsqueeze(1)

        # return (pool.reshape(batch, seq_len, d_model), tokens_per_expert)
