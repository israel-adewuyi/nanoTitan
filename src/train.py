from __future__ import annotations

import argparse
import logging
import time

import torch
import torch.nn.functional as F

from src.config import AppConfig
from src.data.dataset import PackedTokenDataset
from src.dist_env import get_world_size, init_distributed
from src.model.model import NanoTitanModel
from src.model_utils import get_model_shard_specs
from src.optim import setup_optimizer
from src.parallel_dims import get_parallel_dims
from src.profiler import build_profiler
from src.runtime import (
    DDPRuntime,
    DDPRuntimeRef,
    GPipePipelineParallel,
    NaivePipelineParallel,
    SingleDeviceRuntime,
)
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
        "--log.level",
        default="INFO",
        dest="log_level",
        help="Python logging level to use, e.g. DEBUG, INFO, WARNING.",
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
    elif cfg.runtime.name == "ddp":
        return DDPRuntime(cfg)
    elif cfg.runtime.name == "naive_pp":
        return NaivePipelineParallel(cfg)
    elif cfg.runtime.name == "pp_gpipe":
        return GPipePipelineParallel(cfg)
    else:
        raise ValueError(f"Runtime of name {cfg.runtime.name} hasn't been implemented yet.")


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    # get configs, run sanity checks for device mesh
    cfg = load_run_config(args.config)
    init_distributed()
    world_size = get_world_size()
    assert world_size == cfg.runtime.dp_size * cfg.runtime.pp_size
    assert cfg.model.n_layers % cfg.runtime.pp_size == 0
    # assert cfg.trainer.per_device_batch_size % cfg.trainer.microbatches == 0

    # setup runtime, seed everything
    runtime = build_runtime(cfg)
    seed_everything(cfg.trainer.seed)

    # Setup the data loader for train and test
    train_dataset = PackedTokenDataset(
        path=cfg.data.train_tokens_path, seq_len=cfg.model.max_seq_len
    )
    val_dataset = PackedTokenDataset(path=cfg.data.val_tokens_path, seq_len=cfg.model.max_seq_len)
    train_loader = runtime.prepare_trainloader(train_dataset)
    val_loader = runtime.prepare_valloader(val_dataset)

    dims = get_parallel_dims(cfg.runtime)
    spec = get_model_shard_specs(dims, cfg)
    # Setup the model
    raw_model = NanoTitanModel.from_specs(cfg.model, spec)
    runtime.register_model_stats(raw_model)
    model = runtime.prepare_model(raw_model)

    if runtime.is_main_rank():
        total_params = raw_model.total_parameter_count()
        active_params = raw_model.active_parameter_count()
        # Some detail logs about model and args
        logger.info("Loaded model config from %s", normalize_config_arg(args.config))
        logger.info("Model config: %s", cfg.model.model_dump())
        logger.info(
            "Number of parameters: %.2fM total, %.2fM active",
            total_params / 1e6,
            active_params / 1e6,
        )
        logger.info(f"Runtime is {cfg.runtime.name}")

    # Setup the optimizer
    optimizer = setup_optimizer(cfg.optim, model)

    iter = 10
    profiler = build_profiler(runtime, cfg.profiler)

    step = 0
    try:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(0)

        with profiler as prof:
            step_start_time = time.perf_counter()
            for x, y in train_loader:
                step_metrics = runtime.train_step(model, (x, y), optimizer)

                iter -= 1

                train_step_time = time.perf_counter() - step_start_time
                tokens = (step + 1) * runtime.tokens_per_step

                metrics = {
                    "stats/tokens": ScalarMetric(tokens, reduce="none"),
                    "stats/train_step_time": ScalarMetric(train_step_time, reduce="max"),
                }

                metrics.update(step_metrics)

                runtime.log(step, metrics)

                prof.step()

                if iter == 0:
                    break

                if cfg.trainer.eval_every_step != -1 and (
                    cfg.trainer.eval_every_step == 0
                    or (step + 1) % cfg.trainer.eval_every_step == 0
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
                    if runtime.is_main_rank():
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
