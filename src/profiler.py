from __future__ import annotations

import logging
from pathlib import Path

import torch

from src.config import ProfilerConfig

# from src.runtime.base import Runtime

logger = logging.getLogger(__name__)


class NoOpProfiler:
    def __enter__(self) -> NoOpProfiler:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        return False

    def step(self) -> None:
        pass


# def _resolve_runtime_device(runtime: Runtime) -> torch.device:
#     device = runtime.device
#     if isinstance(device, torch.device):
#         return device
#     return torch.device(device)


def _get_activities() -> list[torch.profiler.ProfilerActivity]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    activities.append(torch.profiler.ProfilerActivity.CUDA)
    return activities


def build_profiler(run_name: str, cfg: ProfilerConfig, dims):
    if not cfg.enabled:
        return NoOpProfiler()

    if dims.local_rank != 0:
        return NoOpProfiler()

    trace_dir = Path("runs") / f"{run_name}" / "profiler"
    if trace_dir is None:
        return NoOpProfiler()

    trace_dir.mkdir(parents=True, exist_ok=True)
    worker_name = f"rank_{dims.global_rank}_trace"
    logger.info(
        "Torch profiler enabled for %s. Traces will be written to %s",
        worker_name,
        trace_dir,
    )

    return torch.profiler.profile(
        activities=_get_activities(),
        schedule=torch.profiler.schedule(
            wait=cfg.wait_steps,
            warmup=cfg.warmup_steps,
            active=cfg.active_steps,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(trace_dir), worker_name=worker_name
        ),
        record_shapes=cfg.record_shapes,
        profile_memory=cfg.profile_memory,
        with_stack=cfg.with_stack,
        with_flops=cfg.with_flops,
    )
