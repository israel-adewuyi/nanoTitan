from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ReduceOp = Literal["mean", "max", "sum", "none"]


@dataclass(frozen=True)
class ScalarMetric:
    value: float
    reduce: ReduceOp = "mean"


class Runtime(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.log_dir: Path | None = None
        self.num_model_params: int | None = None
        self.flops_per_token: int | None = None

    def register_model_stats(self, model) -> None:
        self.num_model_params = sum(p.numel() for p in model.parameters())
        cfg = self.cfg.model
        self.flops_per_token = (
            6 * self.num_model_params
            + 12 * cfg.n_layers * cfg.n_heads * cfg.d_head * cfg.max_seq_len
        )

    def add_throughput_metrics(self, metrics: dict[str, float]) -> None:
        if "stats/train_step_time" not in metrics:
            return

        train_step_time = metrics["stats/train_step_time"]
        tokens_per_sec = self.tokens_per_step / train_step_time
        metrics["stats/tokens_per_sec"] = tokens_per_sec

        peak_flops_per_gpu = self.cfg.hardware.peak_flops_tflops_per_gpu * 1e12
        if not self.flops_per_token or peak_flops_per_gpu <= 0:
            return

        model_flops_per_sec = self.flops_per_token * tokens_per_sec
        hardware_flops_per_sec = self.num_mfu_devices * peak_flops_per_gpu
        mfu = model_flops_per_sec / hardware_flops_per_sec
        metrics["stats/mfu"] = mfu
        metrics["stats/mfu_percent"] = 100 * mfu

    @property
    def num_mfu_devices(self) -> int:
        return 1

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def prepare_model(self, model):
        pass

    @abstractmethod
    def log(self, step: int, values_to_log: dict[str, ScalarMetric]):
        pass

    @abstractmethod
    def prepare_trainloader(self, train_dataset):
        pass

    @abstractmethod
    def prepare_valloader(self, val_dataset):
        pass

    @abstractmethod
    def train_step(self, model, batch, optimizer):
        pass

    @abstractmethod
    def backward(self, loss):
        pass

    @abstractmethod
    def finalize_backward(self):
        pass

    @abstractmethod
    def is_main_rank(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    def get_profiler_trace_dir(self) -> Path | None:
        if self.log_dir is None or not self.is_main_rank():
            return None
        return self.log_dir / "profiler"
