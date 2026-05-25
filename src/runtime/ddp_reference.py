import logging
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.dataset import PackedTokenDataset
from src.dist_env import (
    cleanup,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
)
from src.model import NanoTitanModel
from src.runtime.base import Runtime, ScalarMetric
from src.utils import setup_tensorboard

logger = logging.getLogger(__name__)


class DDPRuntimeRef(Runtime):
    """Reference DDP Implementation with torch.nn.parallel.DistributedDataParallel"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup()

    def setup(self):
        init_distributed()
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = torch.device(f"cuda:{self.local_rank}")
        logger.debug(f"World size = {self.world_size}, rank = {self.rank}")

        if is_main_process():
            self.metrics_logger = setup_tensorboard(self.cfg.run_name)
            self.log_dir = self.metrics_logger.log_dir
            self.metrics_logger.log_config(self.cfg.model_dump())

    def log(self, step: int, values_to_log: dict[str, float]) -> None:
        reduced = {}
        for k, metric in values_to_log.items():
            reduced[k] = self.reduce_scalar(metric.value, metric.reduce)

        self.add_throughput_metrics(reduced)
        if is_main_process():
            if "train/loss" in reduced:
                logger.info("[Step %s] Loss: %.6f", step, reduced["train/loss"])
            self.metrics_logger.log(step, reduced)

    def reduce_scalar(self, value: float | int, reduce: str) -> float:
        value = torch.tensor([value], device=self.device, dtype=torch.float32)

        if reduce == "mean":
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
            value /= self.world_size
        elif reduce == "max":
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
        elif reduce == "sum":
            dist.all_reduce(value, op=dist.ReduceOp.SUM)
        elif reduce == "none":
            # all ranks have the same val e.g tokens
            pass
        else:
            raise ValueError(f"Unknown reduce type: {reduce}")

        return value.item()

    def prepare_model(self, model: NanoTitanModel):
        model = model.to(self.device)
        return DDP(model, device_ids=[self.local_rank], output_device=self.local_rank)

    def prepare_trainloader(self, train_dataset: PackedTokenDataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.trainer.per_device_batch_size,
            shuffle=False,
            sampler=DistributedSampler(dataset=train_dataset, shuffle=True),
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

    def prepare_valloader(self, val_dataset: PackedTokenDataset):
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.trainer.per_device_batch_size,
            num_workers=self.cfg.data.num_workers,
            sampler=DistributedSampler(
                dataset=val_dataset,
                shuffle=False,
            ),
            pin_memory=True,
            drop_last=False,
        )
        return val_loader

    def train_step(self, model, batch, optimizer):
        x, y = batch
        # move data to device
        x = x.to(self.device)
        y = y.to(self.device)

        # zero the grad tensor for all trainable params and reset the mem stats
        optimizer.zero_grad()
        self._reset_peak_memory_stats()

        # forward pass
        logits = model(x)

        # compute loss
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        # log loss here

        # backward pass
        if self.cfg.track_backward_time:
            torch.cuda.synchronize(self.device)
            backward_start_time = time.perf_counter()

        self.backward(loss)
        self.finalize_backward()

        if self.cfg.track_backward_time:
            torch.cuda.synchronize(self.device)
            backward_time = time.perf_counter() - backward_start_time

        total_grad_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_norm = param.grad.detach().norm(2)
            total_grad_norm_sq += grad_norm.item() ** 2
        total_grad_norm = total_grad_norm_sq**0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        metrics = {
            "train/loss": ScalarMetric(loss.item(), reduce="mean"),
            "train/grad_norm": ScalarMetric(total_grad_norm, reduce="mean"),
        }
        metrics.update(self._peak_memory_metrics())

        if self.cfg.track_backward_time:
            metrics["stats/backward_time"] = ScalarMetric(backward_time, reduce="max")

        return metrics

    def backward(self, loss):
        loss.backward()

    def is_main_rank(self):
        return is_main_process()

    def finalize_backward(self):
        pass

    # TODO: maybe move these to base? If a couple of runtimes need these
    def _reset_peak_memory_stats(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_memory_mb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024**2)

    # TODO: Reason about this metric... might be the wrong formulation to use
    def _peak_memory_metrics(self) -> dict[str, ScalarMetric]:
        peak_memory_mb = self._peak_memory_mb()
        return {
            f"stats/peak_memory_rank_{rank}_mb": ScalarMetric(
                peak_memory_mb if rank == self.rank else 0.0,
                reduce="sum",
            )
            for rank in range(self.world_size)
        }

    @property
    def tokens_per_step(self):
        return self.cfg.trainer.per_device_batch_size * self.cfg.model.max_seq_len * self.world_size

    @property
    def num_mfu_devices(self):
        return self.world_size

    def cleanup(self):
        if is_main_process():
            self.metrics_logger.close()
        cleanup()
