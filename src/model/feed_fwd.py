import torch
import torch.nn as nn
from jaxtyping import Float

from src.config import ModelConfig
from src.model.cuda_backend import CUDAMoEBackend
from src.model.moe_ops import grouped_gemm_fn
from src.model.torch_backend import TorchMoEBackend
from src.model.utils import ModelShardSpec


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

    def __init__(self, cfg: ModelConfig, spec: ModelShardSpec):
        super().__init__()
        self.cfg = cfg
        self.spec = spec

        # SwiGLU uses two parallel input projections
        self.W_gate = nn.Parameter(
            torch.empty(
                spec.per_rank_expert,
                cfg.d_model,
                cfg.ffn_in,
                dtype=cfg.dtype,
            )
        )
        self.W_val = nn.Parameter(
            torch.empty(
                spec.per_rank_expert,
                cfg.d_model,
                cfg.ffn_in,
                dtype=cfg.dtype,
            )
        )
        self.W_out = nn.Parameter(
            torch.empty(
                spec.per_rank_expert,
                cfg.ffn_in,
                cfg.d_model,
                dtype=cfg.dtype,
            )
        )
        self.silu = nn.SiLU()
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_normal_(self.W_gate, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.W_val, mode="fan_in", nonlinearity="relu")
        nn.init.kaiming_normal_(self.W_out, mode="fan_in", nonlinearity="relu")

    def _forward_torch(
        self,
        packed_X: Float[torch.Tensor, "num_assignments d_model"],
        expert_offset: Float[torch.Tensor, "num_experts + 1"],
    ):
        out = torch.empty_like(packed_X)

        for e in range(self.cfg.num_experts):  # TODO: torch backend not compatible with dist MoE
            start = expert_offset[e].item()
            end = expert_offset[e + 1].item()

            expert_input = packed_X[start:end]

            gate = self.silu(expert_input @ self.W_gate[e])
            val = expert_input @ self.W_val[e]
            temp_out = gate * val
            out[start:end] = temp_out @ self.W_out[e]

        return out

    def _forward_cuda(
        self,
        packed_X: Float[torch.Tensor, "num_assignments d_model"],
        expert_offset: Float[torch.Tensor, "num_experts + 1"],
    ):
        gate_logits = grouped_gemm_fn(packed_X, self.W_gate, expert_offset)
        value = grouped_gemm_fn(packed_X, self.W_val, expert_offset)

        hidden = torch.nn.functional.silu(gate_logits) * value

        return grouped_gemm_fn(hidden, self.W_out, expert_offset)

    def forward(
        self,
        packed_X: Float[torch.Tensor, "num_assignments d_model"],
        expert_offset: Float[torch.Tensor, "num_experts + 1"],
    ) -> Float[torch.Tensor, "num_assignments d_model"]:
        if self.cfg.moe_backend == "cuda":
            return self._forward_cuda(packed_X, expert_offset)
        elif self.cfg.moe_backend == "torch":
            return self._forward_torch(packed_X, expert_offset)

    def parameter_count(self) -> int:
        return sum(param.numel() for param in self.parameters())


class MoE(nn.Module):
    """
    A Mixture of Experts layer
    """

    def __init__(self, cfg: ModelConfig, spec: ModelShardSpec):
        super().__init__()
        self.cfg = cfg
        self.experts = ExpertFFN(cfg, spec)
        self.router = nn.Linear(
            self.cfg.d_model, self.cfg.num_experts, bias=False, dtype=cfg.moe_router_dtype
        )
        if cfg.moe_backend == "cuda":
            self.moe_backend = CUDAMoEBackend(cfg, self.experts, self.router)
        else:
            self.moe_backend = TorchMoEBackend(cfg, self.experts, self.router)

    def active_parameter_count(self) -> int:
        router_params = sum(param.numel() for param in self.router.parameters())
        expert_params = self.experts.parameter_count()
        return router_params + self.cfg.top_k * (expert_params / self.cfg.num_experts)

    def forward(self, x: Float[torch.Tensor, "batch seq_len d_model"]) -> tuple:
        return self.moe_backend.forward(x)
