import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.config import AppConfig
from src.data.dataset import PackedTokenDataset
from src.parallel_dims import ParallelDims
from src.runtime.reducer import ReducerV1


class DataParallel:
    def __init__(self, cfg: AppConfig, dims: ParallelDims):
        self.cfg = cfg
        self.dims = dims
        self.device = f"cuda:{dims.local_rank}"

    def prepare_model(self, model):
        model = model.to(self.device)

        src = self.dims.dp_group_ranks[0]
        for params in model.parameters():
            dist.broadcast(params, src=src, group=self.dims.dp_group)

        self.reducer = ReducerV1(
            model, self.dims.dp_size, self.dims.dp_group_ranks, self.cfg.runtime.bucket_size
        )

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
