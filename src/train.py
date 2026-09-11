from __future__ import annotations

import argparse
import logging
import os
import time
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist

from src.data.dataset import PackedTokenDataset
from src.dist_env import cleanup, get_world_size, init_distributed
from src.metrics import HistogramMetric, MetricsLogger, ScalarMetric
from src.model.model import NanoTitanModel
from src.model.utils import get_model_shard_specs
from src.optim import setup_optimizer
from src.parallel import (
    DataParallel,
    PipelineParallel,
)
from src.parallel_dims import get_parallel_dims
from src.profiler import build_profiler
from src.utils import (
    SUCCESS,
    load_run_config,
    resolve_dtype,
    seed_everything,
    setup_logging,
)

logger = logging.getLogger(__name__)


def reduce_metrics(
    metrics: dict[str, ScalarMetric | HistogramMetric], device: torch.device
) -> dict[str, float | torch.Tensor]:
    reduced = {}
    world_size = dist.get_world_size()

    for name, metric in metrics.items():
        value = torch.as_tensor(metric.value, dtype=torch.float64, device=device)

        if metric.reduce == "sum":
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        elif metric.reduce == "mean":
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= world_size
        elif metric.reduce == "max":
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
        elif metric.reduce != "none":
            raise ValueError(f"Unknown metric reduction: {metric.reduce}")

        if value.ndim == 0:
            reduced[name] = value.item()
        else:
            reduced[name] = value.detach().cpu().to(torch.float32)

    return reduced


def run_validation(
    model: NanoTitanModel,
    pipeline: PipelineParallel,
    val_loader,
    num_val_batches: int,
    metric_device: torch.device,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    token_count = 0
    start_time = time.perf_counter()

    for batch in islice(val_loader, num_val_batches):
        batch_loss_sum, batch_token_count = pipeline.val_step(model, batch)
        loss_sum += batch_loss_sum
        token_count += batch_token_count

    model.train()

    reduced = reduce_metrics(
        {
            "val/loss_sum": ScalarMetric(loss_sum, reduce="sum"),
            "val/token_count": ScalarMetric(token_count, reduce="sum"),
            "time/validation_time": ScalarMetric(
                time.perf_counter() - start_time,
                reduce="max",
            ),
        },
        metric_device,
    )

    return {
        "val/ce_loss": reduced["val/loss_sum"] / reduced["val/token_count"],
        "val/token_count": reduced["val/token_count"],
        "time/validation_time": reduced["time/validation_time"],
    }


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


def main() -> None:
    args = parse_args()

    # get configs, run sanity checks for device mesh
    cfg = load_run_config(args.config)
    log_dir = Path("runs") / cfg.run_name / "logs"
    rank = int(os.environ.get("RANK", "0"))
    log_file = log_dir / f"rank-{rank}.log"
    setup_logging(args.log_level, log_file=log_file)
    logger.info("Writing logs to %s", log_file)
    init_distributed()
    world_size = get_world_size()
    assert world_size == cfg.runtime.dp_size * cfg.runtime.pp_size * cfg.runtime.ep_size
    assert cfg.model.n_layers % cfg.runtime.pp_size == 0
    assert cfg.trainer.per_device_batch_size % cfg.runtime.num_microbatches == 0

    # seed everything
    seed_everything(cfg.trainer.seed)

    # divide ranks into their respective process groups, based on the parallelism config
    dims = get_parallel_dims(cfg.runtime)
    logger.debug(f"At rank {dims.global_rank}, {repr(dims)}")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # allocate model layers to different ranks, including token, pos embed and umembed layer
    spec = get_model_shard_specs(dims, cfg)
    logger.debug(f"At rank {dims.global_rank}, spec is {spec}")

    cfg.model.dtype = resolve_dtype(cfg.model.dtype)
    cfg.model.moe_router_dtype = resolve_dtype(cfg.model.moe_router_dtype)

    # Setup the model
    model = NanoTitanModel.from_specs(cfg.model, spec)
    dp = DataParallel(cfg, dims)
    dp.prepare_model(model)
    pp = PipelineParallel(cfg, dims, dp.get_reducers())
    logger.debug(model)

    if dims.local_rank == 0:
        logger.debug(f"There are {len(list(model.parameters()))} parameters in the model")

    metrics_logger = None
    if dims.local_rank == 0:
        metrics_logger = MetricsLogger(cfg.run_name)

    # Setup the data loaders for training and validation.
    train_dataset = PackedTokenDataset(
        name=cfg.data.dataset_name,
        seq_len=cfg.model.max_seq_len,
        seed=cfg.trainer.seed,
        rank=dims.data_rank,
        world_size=dims.data_world_size,
        dataset_path=cfg.data.dataset_path,
    )
    train_loader = dp.prepare_trainloader(train_dataset)
    val_loader = None
    if cfg.trainer.val_interval != -1:
        val_dataset = PackedTokenDataset(
            name=cfg.data.dataset_name,
            seq_len=cfg.model.max_seq_len,
            seed=cfg.trainer.seed,
            rank=dims.data_rank,
            world_size=dims.data_world_size,
            split="validation",
            shuffle=False,
        )
        val_loader = dp.prepare_valloader(val_dataset)

    parameter_groups = model.parameter_sync_groups()
    non_expert_params = sum(param.numel() for param in parameter_groups["non_expert"])
    expert_params = sum(param.numel() for param in parameter_groups["expert"])

    if dims.dp_rank == 0:
        non_expert_params = non_expert_params if dims.ep_rank == 0 else 0
        local_total_params = non_expert_params + expert_params
        local_active_params = (
            non_expert_params + expert_params * cfg.model.top_k / cfg.model.num_experts
        )
    else:
        local_total_params = local_active_params = 0

    parameter_counts = torch.tensor(
        [local_total_params, local_active_params],
        dtype=torch.float64,
        device=next(model.parameters()).device,
    )
    dist.all_reduce(parameter_counts, op=dist.ReduceOp.SUM)
    total_params, active_params = parameter_counts.tolist()

    if dims.global_rank == 0:
        logger.info(
            "Number of parameters: %.2fM total, %.2fM active",
            total_params / 1e6,
            active_params / 1e6,
        )
        logger.info(f"Model parameters and activations will be in {cfg.model.dtype} datatype")

    # Setup the optimizer
    optimizer = setup_optimizer(cfg.optim, model)

    iter = 0
    profiler = build_profiler(cfg.run_name, cfg.profiler, dims)  # TODO: Fixx
    metric_device = torch.device(f"cuda:{dims.local_rank}" if torch.cuda.is_available() else "cpu")
    logger.info("Attempting to begin training")

    try:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(0)

        with profiler as prof:
            for batch in train_loader:
                metrics = pp.train_step(model, batch, optimizer)
                num_tokens = (
                    cfg.trainer.per_device_batch_size
                    * cfg.runtime.dp_size
                    * cfg.model.max_seq_len
                    * cfg.runtime.ep_size
                )
                metrics.update(
                    {
                        "train/tokens_per_step": ScalarMetric(num_tokens, reduce="none"),
                        "train/total_tokens_seen": ScalarMetric(
                            num_tokens * (iter + 1), reduce="none"
                        ),
                    }
                )
                metrics = reduce_metrics(metrics, metric_device)
                metrics["train/tokens_per_second"] = (
                    metrics["train/tokens_per_step"] / metrics["time/step_time"]
                )
                metrics["train/ce_loss"] = metrics["train/ce_loss"] / (dims.dp_size * dims.ep_size)

                # Log metrics to tensorboard on rank 0
                if dims.local_rank == 0:
                    logger.log(
                        SUCCESS,
                        "Step %s | rank=%s | ce_loss=%.6f | max_vio=%.4f | grad_norm=%.4f",
                        iter + 1,
                        dims.global_rank,
                        metrics["train/ce_loss"],
                        metrics["moe/max_vio"],
                        metrics["train/grad_norm"],
                    )
                    metrics_logger.log(step=iter, metrics=metrics)

                completed_steps = iter + 1
                if val_loader is not None and completed_steps % cfg.trainer.val_interval == 0:
                    val_metrics = run_validation(
                        model,
                        pp,
                        val_loader,
                        cfg.trainer.num_val_batches,
                        metric_device,
                    )
                    if dims.local_rank == 0:
                        logger.info(
                            "Step %s validation CE loss: %.6f",
                            completed_steps,
                            val_metrics["val/ce_loss"],
                        )
                        metrics_logger.log(step=iter, metrics=val_metrics)

                iter += 1

                prof.step()

                if iter == cfg.max_steps:
                    break
    finally:
        cleanup()
    cleanup()


if __name__ == "__main__":
    main()
