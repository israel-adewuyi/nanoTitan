import logging
import math
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from src.pipeline_model import PipelineStageModel

# TODO: two is_main_process is bad code broo
from src.runtime.base import Runtime, ScalarMetric
from src.utils import setup_tensorboard

logger = logging.getLogger(__name__)


class NaivePipelineParallel(Runtime):
    """Naive pipeline parallelism implementation"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup()

    def setup(self):
        init_distributed()
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = torch.device(f"cuda:{self.local_rank}")

        if self.is_main_rank():
            self.metrics_logger = setup_tensorboard(self.cfg.run_name)
            self.log_dir = self.metrics_logger.log_dir
            self.metrics_logger.log_config(self.cfg.model_dump())

        # For communication during fwd/bwd pass
        self.prev_rank = self.rank - 1
        self.next_rank = self.rank + 1

        # For tied embeddings
        self.tie_embed_group = dist.new_group(ranks=[0, self.world_size - 1])

    def get_rank_bounds(self):
        # return the layers that the current rank should process
        per_rank_layers = self.cfg.model.n_layers // self.world_size
        start_idx = self.rank * per_rank_layers
        end_idx = self.rank * per_rank_layers + per_rank_layers
        return (start_idx, end_idx)

    def prepare_model(self, model: NanoTitanModel):
        assert self.cfg.model.n_layers % self.world_size == 0, (
            "The number of GPUs should be divisible by the number of layers of the model"
        )

        self.start_idx, self.end_idx = self.get_rank_bounds()

        # instantiate a pipeline stage model with the exact layers / modules the current rank should process
        return PipelineStageModel(
            model=model,
            rank=self.rank,
            cfg=self.cfg,
            start_idx=self.start_idx,
            end_idx=self.end_idx,
            device=self.device,
            is_first_stage=is_main_process(),
            is_last_stage=self.is_last_rank,
        )

    def train_step(self, model, batch, optimizer):
        x, y = batch
        optimizer.zero_grad()
        self._reset_peak_memory_stats()

        if self.is_main_rank():
            x = x.to(self.device)
        else:
            x = torch.empty(
                (
                    self.cfg.trainer.per_device_batch_size,
                    self.cfg.model.max_seq_len,
                    self.cfg.model.d_model,
                ),
                device=self.device,
            )
            logger.debug(f"At rank {self.rank}!!! Receiving activations from rank {self.prev_rank}")
            dist.recv(x, self.prev_rank)

        x = model(x)
        loss, backward_time = None, None

        if self.is_last_rank:
            # compute loss here
            y = y.to(self.device)
            loss = F.cross_entropy(x.reshape(-1, x.size(-1)), y.reshape(-1))
        else:
            logger.debug(f"Sending activations from rank {self.rank} to rank {self.next_rank}")
            dist.send(x, self.next_rank)

        if self.cfg.track_backward_time:
            torch.cuda.synchronize(self.device)
            backward_start_time = time.perf_counter()

        self.backward(loss, model)
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
        # This value is only for the params the current rank is holding.

        self._clip_grad_norm(model, total_grad_norm_sq)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        metrics = {
            "train/loss": ScalarMetric(
                loss.item() if self.is_last_rank else 0.0,
                reduce="sum",
            ),
            "train/stage_grad_norm": ScalarMetric(total_grad_norm_sq, reduce="sum"),
        }
        metrics.update(self._peak_memory_metrics())

        if self.cfg.track_backward_time:
            metrics["stats/backward_time"] = ScalarMetric(backward_time, reduce="max")

        return metrics

    def _clip_grad_norm(self, model, grad_norm_sq: float):
        grad_norm_sq = torch.tensor([grad_norm_sq], device=self.device, dtype=torch.float32)
        dist.all_reduce(grad_norm_sq, op=dist.ReduceOp.SUM)
        # TODO: max norm is harddcoded
        # TODO: .item() is blocking here. is this the right implementation?
        scale = min(2.0 / (math.sqrt(grad_norm_sq.item()) + 1e-8), 1.0)
        for param in model.parameters():
            if param.grad is not None:
                param.grad.mul_(scale)

    def _reset_peak_memory_stats(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_memory_mb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024**2)

    def _peak_memory_metrics(self) -> dict[str, ScalarMetric]:
        peak_memory_mb = self._peak_memory_mb()
        return {
            f"stats/peak_memory_rank_{rank}_mb": ScalarMetric(
                peak_memory_mb if rank == self.rank else 0.0,
                reduce="sum",
            )
            for rank in range(self.world_size)
        }

    # TODO: Blind copying the dataset fn from ddp. Will have to fix later
    def prepare_trainloader(self, train_dataset: PackedTokenDataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.trainer.per_device_batch_size,
            shuffle=False,
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
            pin_memory=True,
            drop_last=False,
        )
        return val_loader

    def log(self, step: int, values_to_log: dict[str, float]) -> dict:
        reduced = {}
        for k, metric in values_to_log.items():
            reduced[k] = self.reduce_scalar(metric.value, metric.reduce)

        if "stats/train_step_time" in reduced:
            reduced["stats/tokens_per_sec"] = (
                self.tokens_per_step / reduced["stats/train_step_time"]
            )
        grad_norm_sq = reduced.pop("train/stage_grad_norm", None)
        if grad_norm_sq is not None:
            reduced["train/grad_norm"] = math.sqrt(grad_norm_sq)
        if is_main_process():
            if "train/loss" in reduced:
                logger.info("[Step %s] Loss: %.6f", step, reduced["train/loss"])
            self.metrics_logger.log(step, reduced)
        return reduced

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
            pass
        else:
            raise ValueError(f"Unknown reduce type: {reduce}")

        return value.item()

    def backward(self, loss, model):
        if self.rank == self.world_size - 1:
            loss.backward()
        else:
            out_acts = model.get_outgoing_acts()
            out_acts_grad = torch.empty(
                (
                    self.cfg.trainer.per_device_batch_size,
                    self.cfg.model.max_seq_len,
                    self.cfg.model.d_model,
                ),
                device=self.device,
            )
            dist.recv(out_acts_grad, self.next_rank)
            out_acts.backward(out_acts_grad)

        # incoming.backward()
        if not self.is_main_rank():
            dist.send(model.get_incoming_acts_grad(), self.prev_rank)

        if self.is_main_rank() or self.is_last_rank:
            grad = model.token_embed.token_embed.weight.grad
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.tie_embed_group)
            grad.div_(dist.get_world_size(self.tie_embed_group))

    def finalize_backward(self):
        pass

    def cleanup(self):
        if is_main_process():
            self.metrics_logger.close()
        cleanup()

    def is_main_rank(self):
        return is_main_process()

    @property
    def is_last_rank(self):
        return self.rank == self.world_size - 1

    @property
    def tokens_per_step(self):
        return self.cfg.trainer.per_device_batch_size * self.cfg.model.max_seq_len
