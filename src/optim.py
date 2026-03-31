from torch.optim import AdamW

from src.config import OptimizerConfig


def setup_optimizer(cfg: OptimizerConfig, model):
    if cfg.type == "adam":
        return AdamW(model.parameters(), lr=cfg.lr)

    raise ValueError(f"Optimizer type {cfg.type} is not supported")
