from dataclasses import dataclass

from src.config import RuntimeConfig
from src.dist_env import get_local_rank, get_rank, get_world_size


@dataclass
class ParallelDims:
    def __init__(self, config: RuntimeConfig):
        self.world_size = get_world_size()
        self.global_rank = get_rank()
        self.local_rank = get_local_rank()

        self.dp_size = config.dp_size
        self.pp_size = config.pp_size

        assert self.pp_size * self.dp_size == self.world_size

        self.dp_rank = self.global_rank // self.pp_size
        self.pp_rank = self.global_rank % self.pp_size

        self.is_pp_first_stage = self.pp_rank == 0
        self.is_pp_last_stage = self.pp_rank == self.pp_size - 1

        self.prev_pp_rank = -1 if self.is_pp_first_stage else self.global_rank + 1
        self.next_pp_rank = -1 if self.is_pp_last_stage else self.global_rank + 1

        self.dp_group_ranks = [dp * self.pp_size + self.pp_rank for dp in range(self.dp_size)]
        self.pp_group_ranks = [pp + self.dp_rank * self.pp_size for pp in range(self.pp_size)]


def get_parallel_dims(config: RuntimeConfig):
    return ParallelDims(config)
