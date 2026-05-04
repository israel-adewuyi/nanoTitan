import torch

from src.dist_env import get_local_rank, get_rank, get_world_size, init_distributed, is_main_process, cleanup
#TODO: two is_main_process is bad code broo.
from src.runtime.base import Runtime
from src.model import NanoTitanModel
from src.utils import setup_tensorboard
from src.data.dataset import PackedTokenDataset

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


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

    def get_rank_bounds(self):
        per_rank_layers = self.cfg.model.n_layers // self.world_size
        start_idx = self.rank * per_rank_layers
        end_idx = self.rank * per_rank_layers + per_rank_layers
        return (start_idx, end_idx)

    def model_partition(self, model: NanoTitanModel):
        assert self.cfg.model.n_layers % self.world_size == 0, (
            "The number of GPUs should be divisible by the number of layers of the model"
        )

        self.start_idx, self.end_idx = self.get_rank_bounds()
        for layer in range(self.cfg.model.n_layers):
            if layer >= self.start_idx and layer < self.end_idx:
                model.layers[layer].to(self.device)

        if self.is_main_process():
            model.token_embed.to(self.device)
            model.position_embed.to(self.device)

        # for layer in range(self.cfg.n_layers):
        #     model.layers[layer].to(f"cuda:{GPU_IDS[layer]}")

        return model

    def prepare_model(self, model: NanoTitanModel):
        return self.model_partition(model)

    # TODO: Blind copying the dataset fn from ddp. Will have to fix later.
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
