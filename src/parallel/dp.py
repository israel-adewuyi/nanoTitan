import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import ProcessGroup
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.config import AppConfig
from src.data.dataset import PackedTokenDataset
from src.parallel.reducer import ReducerV1
from src.parallel_dims import ParallelDims


class DataParallel:
    """
    The DataParallel class handles all things related to data parallel, from registering hooks
    on params and dividing params in to buckets, to running AllReduce after .grad has been
    computed for all the params.
    """

    def __init__(self, cfg: AppConfig, dims: ParallelDims):
        self.cfg = cfg
        self.dims = dims
        self.device = f"cuda:{dims.local_rank}"

    def prepare_model(self, model):
        """
        This method
        1. syncs the parameters across the dp_group(s) to ensure state_dict is equal at the starts
        2. calls the reducer which registers hooks and divy parameters into buckets
        """
        model = model.to(self.device)

        groups = model.parameter_sync_groups()
        self.broadcast_parameters(
            params=groups["shared"],
            src_rank=self.dims.shared_dp_group_ranks[0],
            group=self.dims.shared_dp_group,
        )

        self.broadcast_parameters(
            params=groups["expert"],
            src_rank=self.dims.expert_dp_group_ranks[0],
            group=self.dims.expert_dp_group,
        )

        self.shared_reducer = ReducerV1(
            groups["shared"],
            len(self.dims.shared_dp_group_ranks),
            self.dims.shared_dp_group,
            self.cfg.runtime.bucket_size,
        )
        self.expert_reducer = ReducerV1(
            groups["expert"],
            len(self.dims.expert_dp_group_ranks),
            self.dims.expert_dp_group,
            self.cfg.runtime.bucket_size,
        )

    def broadcast_parameters(
        self,
        params: tuple[nn.Parameter, ...],
        src_rank: int,
        group: ProcessGroup,
        async_op: bool = False,
    ):
        with torch.no_grad():
            for param in params:
                dist.broadcast(param, src=src_rank, group=group, async_op=async_op)

    def finalize_backward(self):
        self.shared_reducer.finalize_backward()
        self.expert_reducer.finalize_backward()

    def get_reducers(self) -> dict:
        return {"shared": self.shared_reducer, "expert": self.expert_reducer}

    def prepare_trainloader(self, train_dataset: PackedTokenDataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.trainer.per_device_batch_size,
            shuffle=False,
            sampler=DistributedSampler(
                dataset=train_dataset,
                shuffle=True,
                num_replicas=self.dims.data_world_size,
                rank=self.dims.data_rank,
            ),
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
