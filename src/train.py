from __future__ import annotations

import argparse
import logging
import time

import torch
import torch.nn.functional as F

from src.config import AppConfig
from src.data.dataset import PackedTokenDataset
from src.model import NanoTitanModel
from src.optim import setup_optimizer
from src.runtime import DDPRuntimeRef, SingleDeviceRuntime
from src.runtime.base import ScalarMetric
from src.utils import load_run_config, normalize_config_arg, seed_everything, setup_logging

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


def build_runtime(cfg: AppConfig):
    if cfg.runtime.name == "single":
        return SingleDeviceRuntime(cfg)
    elif cfg.runtime.name == "ddp_reference":
        return DDPRuntimeRef(cfg)
    else:
        raise ValueError(f"Runtime of name {cfg.runtime.name} hasn't been implemented yet.")


def main() -> None:
    setup_logging()
    args = parse_args()

    # get configs, setup runtime, seed everything
    cfg = load_run_config(args.config)
    runtime = build_runtime(cfg)
    seed_everything(cfg.trainer.seed)

    # Setup the data loader for train and test
    train_dataset = PackedTokenDataset(
        path=cfg.data.train_tokens_path, seq_len=cfg.model.max_seq_len
    )
    val_dataset = PackedTokenDataset(path=cfg.data.val_tokens_path, seq_len=cfg.model.max_seq_len)
    train_loader = runtime.prepare_trainloader(train_dataset)
    val_loader = runtime.prepare_valloader(val_dataset)

    # Setup the model
    model = runtime.prepare_model(NanoTitanModel(cfg.model))

    if runtime.is_main_process():
        # Some detail logs about model and args
        logger.info("Loaded model config from %s", normalize_config_arg(args.config))
        logger.info("Model config: %s", cfg.model.model_dump())
        logger.info("Number of parameters: %s", sum(p.numel() for p in model.parameters()))
        logger.info(f"Runtime is {cfg.runtime.name}")

    # Setup the optimizer
    optimizer = setup_optimizer(cfg.optim, model)

    iter = 1000
    num_batches = len(train_loader)

    step = 0
    try:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(0)

        step_start_time = time.perf_counter()
        for x, y in train_loader:
            # move data to device
            x = x.to(runtime.device)
            y = y.to(runtime.device)

            # forward pass
            logits = model(x)

            # compute loss
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            if runtime.is_main_process():
                logger.info("[Step %s/%s] Loss: %.6f", step, num_batches, loss.item())

            # backward pass
            optimizer.zero_grad()
            runtime.backward(loss)
            runtime.finalize_backward()

            total_grad_norm_sq = 0.0
            for param in model.parameters():
                if param.grad is None:
                    continue
                grad_norm = param.grad.detach().norm(2)
                total_grad_norm_sq += grad_norm.item() ** 2
            total_grad_norm = total_grad_norm_sq**0.5

            optimizer.step()

            iter -= 1

            train_step_time = time.perf_counter() - step_start_time
            tokens = (step + 1) * runtime.tokens_per_step

            runtime.log(
                step,
                {
                    "stats/tokens": ScalarMetric(tokens, reduce="none"),
                    "stats/train_step_time": ScalarMetric(train_step_time, reduce="max"),
                    "train/loss": ScalarMetric(loss.item(), reduce="mean"),
                    "train/grad_norm": ScalarMetric(total_grad_norm, reduce="mean"),
                },
            )

            if iter == 0:
                break

            if cfg.trainer.eval_every_step != -1 and (
                cfg.trainer.eval_every_step == 0 or (step + 1) % cfg.trainer.eval_every_step == 0
            ):
                model.eval()
                total_loss = 0.0
                num_val_batches = 0
                val_start_time = time.perf_counter()

                with torch.no_grad():
                    for val_x, val_y in val_loader:
                        val_x = val_x.to(runtime.device)
                        val_y = val_y.to(runtime.device)

                        logits = model(val_x)
                        loss = F.cross_entropy(
                            logits.reshape(-1, logits.size(-1)), val_y.reshape(-1)
                        )

                        total_loss += loss.item()
                        num_val_batches += 1

                val_time = time.perf_counter() - val_start_time
                val_loss = total_loss / num_val_batches
                if runtime.is_main_process():
                    logger.info("[Step %s] Validation loss: %.6f", step, val_loss)
                runtime.log(
                    step,
                    {
                        "val/loss": ScalarMetric(val_loss, reduce="mean"),
                        "val/time": ScalarMetric(val_time, reduce="max"),
                    },
                )
                model.train()

            step += 1
            step_start_time = time.perf_counter()
    finally:
        runtime.cleanup()


if __name__ == "__main__":
    main()
