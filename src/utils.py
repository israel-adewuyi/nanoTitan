import logging
from datetime import datetime
from pathlib import Path

import torch

from src.config import AppConfig, load_config
from src.metrics import MetricsLogger


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


def resolve_device(cfg: AppConfig) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{cfg.trainer.device_id}")
    return torch.device("cpu")
