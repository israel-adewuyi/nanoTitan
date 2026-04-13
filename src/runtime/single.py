import torch
from torch.utils.data import DataLoader

from src.data.dataset import PackedTokenDataset
from src.model import NanoTitanModel
from src.runtime.base import Runtime, ScalarMetric
from src.utils import setup_tensorboard


class SingleDeviceRuntime(Runtime):
    """Runtime for Single GPU training"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup()

    def setup(self):
        # TODO: Think on who sets this. Config?
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.metrics_logger = setup_tensorboard(self.cfg.run_name)
        self.metrics_logger.log_config(self.cfg.model_dump())

    def log(self, step: int, values_to_log: dict[str, ScalarMetric]) -> None:
        payload = {k: v.value for k, v in values_to_log.items()}

        if "stats/tokens" in payload and "stats/train_step_time" in payload:
            payload["stats/tokens_per_sec"] = (
                payload["stats/tokens"] / payload["stats/train_step_time"]
            )
        self.metrics_logger.log(step, payload)

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
            shuffle=False,
            batch_size=self.cfg.trainer.per_device_batch_size,
            num_workers=self.cfg.data.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return val_loader

    def backward(self, loss):
        loss.backward()

    def is_main_process(self):
        return True

    def finalize_backward(self):
        pass

    @property
    def tokens_per_step(self):
        return self.cfg.trainer.per_device_batch_size * self.cfg.model.max_seq_len

    def cleanup(self):
        self.metrics_logger.close()
