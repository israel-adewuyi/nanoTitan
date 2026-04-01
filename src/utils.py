import torch
from src.config import AppConfig, load_config

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