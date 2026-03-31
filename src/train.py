from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import AppConfig, load_config
from src.data.dataset import PackedTokenDataset
from src.model import NanoTitanModel
from src.optim import setup_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a NanoTitan TOML config and instantiate the model."
    )
    parser.add_argument(
        "--single_gpu",
        action="store_true",
        help="Reserved training mode flag. The model wiring is identical for now.",
    )
    parser.add_argument(
        "config",
        help="TOML config path. Prefix with '@' to match the planned launcher style.",
    )
    return parser.parse_args()


def normalize_config_arg(config_arg: str) -> str:
    return config_arg[1:] if config_arg.startswith("@") else config_arg


def load_dataloader(cfg: AppConfig):
    train_dataset = PackedTokenDataset(
        path=cfg.data.train_tokens_path, seq_len=cfg.model.max_seq_len
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.model.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


def build_from_config(config_arg: str) -> tuple[AppConfig, NanoTitanModel]:
    config_path = normalize_config_arg(config_arg)
    app_config = load_config(config_path)
    model = NanoTitanModel(app_config.model)
    return app_config, model


def resolve_device(cfg: AppConfig) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{cfg.trainer.device_id}")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    app_config, model = build_from_config(args.config)
    device = resolve_device(app_config)

    print(f"Loaded model config from {normalize_config_arg(args.config)}")
    print(app_config.model.model_dump())
    print(f"Instantiated {model.__class__.__name__}")
    print(f"Number of parameters are {sum(p.numel() for p in model.parameters())}")
    if args.single_gpu:
        print("single_gpu mode enabled")

    train_loader = load_dataloader(app_config)
    model = model.to(device)
    optimizer = setup_optimizer(app_config.optim, model)
    print(next(model.parameters()).device)

    iter = 50

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        y = y.unsqueeze(2)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        print(f"Loss is {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iter -= 1

        if iter == 0:
            break

if __name__ == "__main__":
    main()
