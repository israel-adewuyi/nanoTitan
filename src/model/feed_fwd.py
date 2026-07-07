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
        self.W_gate = nn.Linear(cfg.d_model, cfg.ffn_in, dtype=cfg.dtype, bias=False)
        self.W_val = nn.Linear(cfg.d_model, cfg.ffn_in, dtype=cfg.dtype, bias=False)
        self.W_out = nn.Linear(cfg.ffn_in, cfg.d_model, dtype=cfg.dtype, bias=False)
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


class ExpertFFN(nn.Module):
    """Feedforward network implementation"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        # SwiGLU uses two parallel input projections
        self.W_gate = nn.Parameter(
            torch.empty(
                cfg.num_experts,
                cfg.d_model,
                cfg.ffn_in,
                dtype=cfg.dtype,
            )
        )
        self.W_val = nn.Parameter(
            torch.empty(
                cfg.num_experts,
                cfg.d_model,
                cfg.ffn_in,
                dtype=cfg.dtype,
            )
        )
        self.W_out = nn.Parameter(
            torch.empty(
                cfg.num_experts,
                cfg.ffn_in,
                cfg.d_model,
                dtype=cfg.dtype,
            )
        )
        self.silu = nn.SiLU()
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.W_gate)
        nn.init.normal_(self.W_val)
        nn.init.normal_(self.W_out)

    def forward(
        self,
        packed_X: Float[torch.Tensor, "num_assignments d_model"],
        expert_offset,  # TODO: shape anotation
    ) -> Float[torch.Tensor, "num_assignments d_model"]:
        out = torch.empty_like(packed_X)

        for e in range(self.cfg.num_experts):
            start = expert_offset[e].item()
            end = expert_offset[e + 1].item()  # TODO: Verify that expert offset is num_experts + 1

            expert_input = packed_X[start:end]

            gate = self.silu(expert_input @ self.W_gate[e])
            val = expert_input @ self.W_val[e]
            temp_out = gate * val
            out[start:end] = temp_out @ self.W_out[e]

        return out

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
        self.router = nn.Linear(
            self.cfg.d_model, self.cfg.num_experts, bias=False, dtype=cfg.moe_router_dtype
        )
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
