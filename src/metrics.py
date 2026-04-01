import json

from torch.utils.tensorboard import SummaryWriter


class MetricsLogger:
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir=log_dir)

    def log(self, step: int, metrics: dict[str, float]) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)

    def log_config(self, config_dict: dict) -> None:
        self.writer.add_text("config", json.dumps(config_dict, indent=2), 0)

    def close(self) -> None:
        self.writer.close()
