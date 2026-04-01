from __future__ import annotations

import argparse
import logging

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import AppConfig
from src.data.dataset import PackedTokenDataset
from src.model import NanoTitanModel
from src.optim import setup_optimizer
from src.utils import (
    load_run_config,
    normalize_config_arg,
    resolve_device,
    setup_logging,
    setup_tensorboard,
)

logger = logging.getLogger(__name__)


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


def load_train_dataloader(cfg: AppConfig):
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


def load_val_dataloader(cfg: AppConfig):
    val_dataset = PackedTokenDataset(path=cfg.data.val_tokens_path, seq_len=cfg.model.max_seq_len)

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return val_loader


def main() -> None:
    setup_logging()
    args = parse_args()

    # get configs and setup tb logger
    cfg = load_run_config(args.config)
    metrics_logger = setup_tensorboard(cfg.run_name)
    device = resolve_device(cfg)

    # Setup the model
    model = NanoTitanModel(cfg.model).to(device)
    metrics_logger.log_config(cfg.model_dump())

    # Some detail logs about model and args
    logger.info("Loaded model config from %s", normalize_config_arg(args.config))
    logger.info("Model config: %s", cfg.model.model_dump())
    logger.info("Number of parameters: %s", sum(p.numel() for p in model.parameters()))
    if args.single_gpu:
        logger.info("single_gpu mode enabled")

    # Setup the data loader for train and test
    train_loader = load_train_dataloader(cfg)
    val_loader = load_val_dataloader(cfg)

    # Setup the optimizer
    optimizer = setup_optimizer(cfg.optim, model)
    logger.info("Model device: %s", next(model.parameters()).device)

    iter = 50
    num_batches = len(train_loader)

    step = 0
    try:
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            logger.info("[Step %s/%s] Loss: %.6f", step, num_batches, loss.item())

            optimizer.zero_grad()
            loss.backward()

            total_grad_norm_sq = 0.0
            for param in model.parameters():
                if param.grad is None:
                    continue
                grad_norm = param.grad.detach().norm(2)
                total_grad_norm_sq += grad_norm.item() ** 2
            total_grad_norm = total_grad_norm_sq**0.5

            metrics_logger.log(
                step,
                {
                    "train/loss": loss.item(),
                    "train/grad_norm": total_grad_norm,
                },
            )

            optimizer.step()

            # iter -= 1

            # if iter == 0:
            #     break

            if step % cfg.trainer.eval_every_step == 0:
                model.eval()
                total_loss = 0.0
                num_val_batches = 0

                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x = val_x.to(device)
                        val_y = val_y.to(device)

                        logits = model(val_x)
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), val_y.reshape(-1)
                        )

                        total_loss += loss.item()
                        num_val_batches += 1

                model.train()
                val_loss = total_loss / num_val_batches
                logger.info("[Step %s] Validation loss: %.6f", step, val_loss)
                metrics_logger.log(step, {"val/loss": val_loss})

            step += 1
    finally:
        metrics_logger.close()


if __name__ == "__main__":
    main()
