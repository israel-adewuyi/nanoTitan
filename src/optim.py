import torch
from torch.optim import AdamW

from src.config import OptimizerConfig


class Float32AdamW(AdamW):
    """Keep half-precision parameters' master weights and Adam moments in FP32."""

    def __init__(self, parameters, **kwargs):
        self.model_parameters = list(parameters)
        self.master_parameters = [
            p if p.dtype == torch.float32 else torch.nn.Parameter(p.detach().float())
            for p in self.model_parameters
        ]
        super().__init__(self.master_parameters, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Float32AdamW does not support closures")
        for model, master in zip(self.model_parameters, self.master_parameters, strict=True):
            if model is not master:
                master.grad = model.grad.float() if model.grad is not None else None
        result = super().step()
        for model, master in zip(self.model_parameters, self.master_parameters, strict=True):
            if model is not master:
                model.copy_(master)
        return result

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        for model, master in zip(self.model_parameters, self.master_parameters, strict=True):
            if model is not master and model.grad is not None:
                if set_to_none:
                    model.grad = None
                else:
                    model.grad.zero_()

    def state_dict(self):
        state = super().state_dict()
        state["master_weights"] = [p.detach().clone() for p in self.master_parameters]
        return state

    def load_state_dict(self, state_dict):
        state_dict = state_dict.copy()
        masters = state_dict.pop("master_weights")
        if len(masters) != len(self.master_parameters):
            raise ValueError("Master parameter count differs from the checkpoint")
        super().load_state_dict(state_dict)
        with torch.no_grad():
            for saved, model, master in zip(masters, self.model_parameters, self.master_parameters, strict=True):
                master.copy_(saved)
                model.copy_(master)


def setup_optimizer(cfg: OptimizerConfig, model):
    if cfg.type == "adam":
        return Float32AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1)

    raise ValueError(f"Optimizer type {cfg.type} is not supported")
