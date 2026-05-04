import torch

from src.runtime.base import Runtime
from src.utils import setup_tensorboard
from src.dist_env import (
    init_distributed, 
    get_world_size, 
    get_rank, 
    get_local_rank
)

class NaivePipelineParallel(Runtime):
    """Naive pipeline parallelism implementation"""

    def __init__(self, cfg):
        super().__init(cfg)
        self.setup()

    def setup(self):
        init_distributed()
        self.world_size = get_world_size()
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = torch.device(f"cuda:{self.local_rank}")

        if is_main_process():
            self.metrics_logger = setup_tensorboard(self.cfg.run_name)
            self.log_dir = self.metrics_logger.log_dir
            self.metrics_logger.log_config(self.cfg.model_dump())
            
    def get_rank_bounds(self):
        per_rank_layers = self.cfg.n_laeyrs / self.world_size
        start_idx = self.rank * per_rank_layers
        end_idx = self.rank * per_rank_layers + per_rank_layers
        return (start_idx, end_idx)

    def model_partition(self, model: NanoTitanModel):
        assert self.cfg.n_layers % self.world_size == 0, "The number of GPUs should be divisible by the number of layers of the model"

        self.start_idx, self.end_idx = self.get_rank_bounds()
        for layer in range(self.cfg.n_layers):
            if layer >= self.start_idx and layer <= self.end_idx:
                model.layer[layer].to(self.device)
                
        if is_main_process():
            model.token_embed.to(self.device)

        # for layer in range(self.cfg.n_layers):
        #     model.layers[layer].to(f"cuda:{GPU_IDS[layer]}")

        return model

    def prepare_model(self, model: NanoTitanModel):
        return self.model_partition(model)

    def cleanup(self):
        if is_main_process():
            self.metrics_logger.close()
        cleanup()

    def is_main_process(self):
        return is_main_process()

