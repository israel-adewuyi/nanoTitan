import torch
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
from src.runtime.base import Runtime
from src.runtime.reducer import ReducerV0
from src.utils import setup_tensorboard


class MiniDDP(Runtime):
    """Mini-DDP Implementation"""

    def __init__(self, cfg):
        super().__init__(cfg)

    def setup(self):
        init_distributed()
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = torch.device(f"cuda:{self.local_rank}")

        if is_main_process():
            self.metrics_logger = setup_tensorboard(self.cfg.run_name)

    def prepare_model(self, model: NanoTitanModel):
        model = model.to(self.device)
        self.reducer = ReducerV0(model, self.world_size)
        return model

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

    def is_main_process():
        return is_main_process()

    def finalize_backward(self):
        pass

    def cleanup(self):
        if is_main_process():
            self.metrics_logger.close()
        cleanup()
