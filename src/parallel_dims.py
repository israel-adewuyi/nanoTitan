from dataclasses import dataclass

import torch.distributed as dist

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
        self.ep_size = config.ep_size
        self.num_experts = config.num_expert  # TODO: In config, post

        assert self.pp_size * self.dp_size * self.ep_size == self.world_size
        assert self.num_experts % self.ep_size == 0, (
            "Number of experts should be divisible by the size of expert group"
        )

        self.dp_rank = self.global_rank // (self.pp_size * self.ep_size)
        self.pp_rank = (self.global_rank // self.ep_size) % self.pp_size
        self.ep_rank = self.global_rank % self.ep_size

        self.is_pp_first_stage = self.pp_rank == 0
        self.is_pp_last_stage = self.pp_rank == self.pp_size - 1

        # self.ep_group_ranks = [((self.dp_rank * self.pp_size) + self.pp_rank) * self.ep_size + ep for ep in range(self.ep_size)]
        self.ep_group = None
        for dp in range(self.dp_size):
            for pp in range(self.pp_size):
                ranks = [
                    ((dp * self.pp_size) + pp) * self.ep_size + ep for ep in range(self.ep_size)
                ]
                group = dist.new_group(ranks=ranks)

                if self.dp_rank == dp and self.pp_rank == pp:
                    self.ep_group = group

        self.pp_group_ranks = [
            ((self.dp_rank * self.pp_size) + pp) * self.ep_size + self.ep_rank
            for pp in range(self.pp_size)
        ]
        self.pp_group = None
        for dp in range(self.dp_size):
            for ep in range(self.ep_size):
                ranks = [
                    ((dp * self.pp_size) + pp) * self.ep_size + ep for pp in range(self.pp_size)
                ]
                group = dist.new_group(ranks=ranks)

                if self.dp_rank == dp and self.ep_rank == ep:
                    self.pp_group = group

        self.shared_dp_group = None
        for pp in range(self.pp_size):
            ranks = [
                ((dp * self.pp_size) + pp) * self.ep_size + ep
                for dp in range(self.dp_size)
                for ep in range(self.ep_size)
            ]
            group = dist.new_group(ranks=ranks)

            if self.pp_rank == pp:
                self.shared_dp_group = group

        self.expert_dp_group = None
        for pp in range(self.pp_size):
            for ep in range(self.ep_size):
                ranks = [
                    ((dp * self.pp_size) + pp) * self.ep_size + ep for dp in range(self.dp_size)
                ]
                group = dist.new_group(ranks=ranks)

                if self.pp_rank == pp and self.ep_rank == ep:
                    self.expert_dp_group = group

        self.prev_pp_rank = -1 if self.is_pp_first_stage else self.global_rank - self.ep_size
        self.next_pp_rank = -1 if self.is_pp_last_stage else self.global_rank + self.ep_size


def get_parallel_dims(config: RuntimeConfig):
    return ParallelDims(config)
