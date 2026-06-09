import logging
import time

import torch
import torch.nn.functional as F
from torch.profiler import record_function
from torch.utils.data import DataLoader

from src.data.dataset import PackedTokenDataset
from src.model.model import NanoTitanModel
from src.runtime.base import Runtime, ScalarMetric
from src.utils import resolve_device, setup_tensorboard

logger = logging.getLogger(__name__)


class SingleDeviceRuntime(Runtime):
    """Runtime for Single GPU training"""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup()

    def setup(self):
        self.device = resolve_device(self.cfg.trainer.device_id)
        self.metrics_logger = setup_tensorboard(self.cfg.run_name)
        self.log_dir = self.metrics_logger.log_dir
        self.metrics_logger.log_config(self.cfg.model_dump())

    def log(self, step: int, values_to_log: dict[str, ScalarMetric]) -> None:
        payload = {k: v.value for k, v in values_to_log.items()}

        self.add_throughput_metrics(payload)
        if "train/loss" in payload:
            logger.info("[Step %s] Loss: %.6f", step, payload["train/loss"])
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

    def train_step(self, model, batch, optimizer):
        x, y = batch
        optimizer.zero_grad()
        self._reset_peak_memory_stats()
        # move data to device
        x = x.to(self.device)
        y = y.to(self.device)

        # forward pass
        with record_function("forward"):
            logits = model(x)

        # compute loss
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if self.cfg.track_backward_time and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        backward_start_time = time.perf_counter()

        with record_function("backward"):
            self.backward(loss, model)
            self.finalize_backward()

        if self.cfg.track_backward_time and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        backward_time = time.perf_counter() - backward_start_time

        total_grad_norm_sq = 0.0
        for param in model.parameters():
            if param.grad is None:
                continue
            grad_norm = param.grad.detach().norm(2)
            total_grad_norm_sq += grad_norm.item() ** 2
        total_grad_norm = total_grad_norm_sq**0.5

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        with record_function("optim_step"):
            optimizer.step()

        metrics = {
            "train/loss": ScalarMetric(loss.item(), reduce="none"),
            "train/grad_norm": ScalarMetric(total_grad_norm, reduce="none"),
            "stats/peak_memory_mb": ScalarMetric(self._peak_memory_mb(), reduce="none"),
        }

        if self.cfg.track_backward_time:
            metrics["stats/backward_time"] = ScalarMetric(backward_time, reduce="none")

        return metrics

    def _reset_peak_memory_stats(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _peak_memory_mb(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return torch.cuda.max_memory_allocated(self.device) / (1024**2)

    def backward(self, loss, model):
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
