import json
from pathlib import Path
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def log(self, step: int, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_config(self, config_dict: dict) -> None:
        self.writer.add_text("config", json.dumps(config_dict, indent=2), 0)

    def close(self) -> None:
        self.writer.close()


ReduceOp = Literal["mean", "max", "sum", "none"]


@dataclass(frozen=True)
class ScalarMetric:
    value: float
    reduce: ReduceOp = "mean"

