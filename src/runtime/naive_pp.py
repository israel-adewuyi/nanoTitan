import torch
import torch.nn.functional as F
import torch.distributed as dist
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
from src.pipeline_model import PipelineStageModel

# TODO: two is_main_process is bad code broo.
from src.runtime.base import Runtime
from src.utils import setup_tensorboard


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
        per_rank_layers = self.cfg.model.n_layers // self.world_size
        start_idx = self.rank * per_rank_layers
        end_idx = self.rank * per_rank_layers + per_rank_layers
        return (start_idx, end_idx)

    def prepare_model(self, model: NanoTitanModel):
        assert self.cfg.model.n_layers % self.world_size == 0, (
            "The number of GPUs should be divisible by the number of layers of the model"
        )

        self.start_idx, self.end_idx = self.get_rank_bounds()

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
            dist.recv(x, self.prev_rank)

        x = model(x)

        if self.is_last_rank:
            # compute loss here
            y = y.to(self.device)
            loss = F.cross_entropy(x.reshape(-1, x.size(-1)), y.reshape(-1))
            return loss
        else:
            dist.send(x, self.next_rank)

        return None

    # TODO: Blind copying the dataset fn from ddp. Will have to fix later.
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
        elif reduce == "none":
            pass
        else:
            raise ValueError(f"Unknown reduce type: {reduce}")

        return value.item()

    def backward(self, loss):
        loss.backward()

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
