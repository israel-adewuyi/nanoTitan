from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ReduceOp = Literal["mean", "max", "none"]


@dataclass(frozen=True)
class ScalarMetric:
    value: float
    reduce: ReduceOp = "mean"


class Runtime(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.log_dir: Path | None = None

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
    def backward(self, loss):
        pass

    @abstractmethod
    def finalize_backward(self):
        pass

    @abstractmethod
    def is_main_process(self):
        pass

    @abstractmethod
    def cleanup(self):
        pass

    def get_profiler_trace_dir(self) -> Path | None:
        if self.log_dir is None or not self.is_main_process():
            return None
        return self.log_dir / "profiler"
