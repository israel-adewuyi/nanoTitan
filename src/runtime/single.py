import torch
from torch.utils.data import DataLoader

from src.data.dataset import PackedTokenDataset
from src.model import NanoTitanModel
from src.runtime.base import Runtime


class SingleDeviceRuntime(Runtime):
    """Runtime for Single GPU training"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup()

    def setup(self):
        # TODO: Think on who sets this. Config?
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def prepare_model(self, model: NanoTitanModel):
        return model.to(self.device)

    def prepare_trainloader(self, train_dataset: PackedTokenDataset):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.trainer.per_device_batch_size,
            shuffle=True,
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
            drop_last=True,
        )
        return val_loader

    def backward(self, loss):
        loss.backward()

    def finalize_backward(self):
        pass

    def cleanup(self):
        pass
