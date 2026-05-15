import torch
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
from src.runtime.base import Runtime
from src.runtime.reducer import ReducerV0, ReducerV1
from src.utils import setup_tensorboard


class DDPRuntime(Runtime):
    """Mini-DDP implementation"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup()

    def setup(self):
        init_distributed()
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = torch.device(f"cuda:{self.local_rank}")

        if is_main_process():
            self.metrics_logger = setup_tensorboard(self.cfg.run_name)
            self.log_dir = self.metrics_logger.log_dir
            self.metrics_logger.log_config(self.cfg.model_dump())

    def prepare_model(self, model: NanoTitanModel):
        model = model.to(self.device)
        if self.cfg.runtime.reducer == "v0":
            self.reducer = ReducerV0(model, self.world_size)
        else:
            self.reducer = ReducerV1(model, self.world_size, self.cfg.runtime.bucket_size)
        return model

    def log(self, step: int, values_to_log: dict[str, float]) -> None:
        reduced = {}
        for k, metric in values_to_log.items():
            reduced[k] = self.reduce_scalar(metric.value, metric.reduce)

        self.add_throughput_metrics(reduced)
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

    def backward(self, loss):
        loss.backward()

    def is_main_process(self):
        return is_main_process()

    def finalize_backward(self):
        self.reducer.finalize_backward()

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


MiniDDP = DDPRuntime
