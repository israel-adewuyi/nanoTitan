import logging
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grads_with_norm_, get_total_norm

from src.config import AppConfig, load_config, TrainerConfig
from src.metrics import MetricsLogger
from src.model.model import NanoTitanModel
from src.parallel_dims import ParallelDims


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def setup_tensorboard(run_name: str, log_root: str = "runs") -> MetricsLogger:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(log_root) / f"{run_name}-{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return MetricsLogger(str(log_dir))


def normalize_config_arg(config_arg: str) -> str:
    return config_arg[1:] if config_arg.startswith("@") else config_arg


def load_run_config(config_arg: str) -> AppConfig:
    config_path = normalize_config_arg(config_arg)
    app_config = load_config(config_path)
    return app_config


def resolve_device(device_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_id}")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reduce_scalars(scalar_tensor, pp_group):
    dist.all_reduce(scalar_tensor, op=dist.ReduceOp.SUM, group=pp_group, async_op=False)


def resolve_dtype(dtype):
    if dtype is None or isinstance(dtype, torch.dtype):
        return dtype
    return getattr(torch, dtype)


def get_profiler_trace_dir() -> Path | None:
    if self.log_dir is None or not self.is_main_rank():
        return None
    return self.log_dir / "profiler"


def compute_grad_norm(model: NanoTitanModel, dims: ParallelDims) -> torch.Tensor:
    expert_params, nonexpert_params = [], []
    expert_param_name_slug = ["W_gate", "W_val", "W_out"]

    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        found = any(pname in name for pname in expert_param_name_slug)
        if found:
            expert_params.append(param.grad)
        else:
            nonexpert_params.append(param.grad)

    ep_squared_norm = get_total_norm(expert_params) ** 2
    nonep_squared_norm = get_total_norm(nonexpert_params) ** 2

    dist.all_reduce(ep_squared_norm, op=dist.ReduceOp.SUM, group=dims.ep_group)
    dist.all_reduce(ep_squared_norm, op=dist.ReduceOp.SUM, group=dims.pp_group)
    dist.all_reduce(nonep_squared_norm, op=dist.ReduceOp.SUM, group=dims.pp_group)

    global_norm = torch.sqrt(ep_squared_norm + nonep_squared_norm)
    assert isinstance(global_norm, torch.Tensor)
    return global_norm


def clip_gradients(model: NanoTitanModel, grad_norm: torch.Tensor, cfg: TrainerConfig):
    clip_grads_with_norm_(model.parameters(), max_norm=cfg.grad_norm, total_norm=grad_norm)
