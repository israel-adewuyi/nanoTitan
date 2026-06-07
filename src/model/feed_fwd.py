import torch
import torch.nn as nn
from jaxtyping import Float

from src.config import ModelConfig


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


class MoE(nn.Module):
    """
    A Mixture of Experts layer
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.experts = nn.ModuleList(FFN(cfg) for _ in range(self.cfg.num_experts))
        self.router = nn.Linear(self.cfg.d_model, self.cfg.num_experts, bias=False)

    def forward(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> tuple:
        batch, seq_len, d_model = x.shape
        # flatten residual stream tokens into a 2D tensor of shape (num_tokens, d_model)
        flat_tokens = x.reshape(-1, d_model)  # (num_tokens d_model)

        # get the expert logits
        expert_logits = self.router(flat_tokens)
        expert_probs = expert_logits.softmax(dim=-1)
        topk_weights, topk_expert_idx = torch.topk(expert_probs, dim=-1, k=self.cfg.top_k)
        expert_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # accumulate the weighted results of each expert per token
        pool = torch.zeros_like(flat_tokens)
        tokens_per_expert = torch.zeros((self.cfg.num_experts), dtype=torch.long, device=x.device)

        for expert in range(self.cfg.num_experts):
            # Return a 2D tensor of which token chose the current expert and the index / position in the topK
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
