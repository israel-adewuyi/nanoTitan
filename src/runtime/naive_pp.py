import logging
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

# TODO: two is_main_process is bad code broo.
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

        if self.is_main_process():
            self.metrics_logger = setup_tensorboard(self.cfg.run_name)
            self.log_dir = self.metrics_logger.log_dir
            self.metrics_logger.log_config(self.cfg.model_dump())

        # For communication during fwd/bwd pass
        self.prev_rank = self.rank - 1
        self.next_rank = self.rank + 1

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

        if self.is_main_process():
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
        total_grad_norm = total_grad_norm_sq**0.5
        # This value is only for the params the current rank is holding.

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        metrics = {
            "train/loss": ScalarMetric(
                loss.item() if self.is_last_rank else 0.0,
                reduce="sum",
            ),
            "train/stage_grad_norm": ScalarMetric(total_grad_norm, reduce="max"),
        }

        if self.cfg.track_backward_time:
            metrics["stats/backward_time"] = ScalarMetric(backward_time, reduce="max")

        return metrics

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

    # TODO: Blind copying log fns too.
    def log(self, step: int, values_to_log: dict[str, float]) -> None:
        reduced = {}
        for k, metric in values_to_log.items():
            reduced[k] = self.reduce_scalar(metric.value, metric.reduce)

        if "stats/train_step_time" in reduced:
            reduced["stats/tokens_per_sec"] = (
                self.tokens_per_step / reduced["stats/train_step_time"]
            )
        if is_main_process():
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
        if not self.is_main_process():
            dist.send(model.get_incoming_acts_grad(), self.prev_rank)

    def finalize_backward(self):
        pass

    def cleanup(self):
        if is_main_process():
            self.metrics_logger.close()
        cleanup()

    def is_main_process(self):
        return is_main_process()

    @property
    def is_last_rank(self):
        return self.rank == self.world_size - 1

    @property
    def tokens_per_step(self):
        return self.cfg.trainer.per_device_batch_size * self.cfg.model.max_seq_len
