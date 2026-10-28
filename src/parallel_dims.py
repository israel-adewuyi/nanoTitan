from dataclasses import dataclass

import torch.distributed as dist

from src.config import AppConfig
from src.dist_env import get_local_rank, get_rank, get_world_size


@dataclass
class ParallelContext:
    def __init__(self, config: AppConfig):
        self.config = config
        self.world_size = get_world_size()

        self._initialize_rank_identity()
        self._initialize_groups()

    def _initialize_rank_identity(self):
        # What rank am I?
        self.global_rank = get_rank()
        self.local_rank = get_local_rank()

        # Size of the different parallelism groups I am in?
        self.dp_size = self.config.runtime.dp_size
        self.pp_size = self.config.runtime.pp_size
        self.ep_size = self.config.runtime.ep_size
        self.tp_size = self.config.runtime.tp_size

        assert self.pp_size * self.dp_size * self.ep_size * self.tp_size == self.world_size
        # TODO: Maybe this test should move out of here
        assert self.config.model.num_experts % self.ep_size == 0, (
            "Number of experts should be divisible by the size of expert group"
        )

        self.dp_rank = self.global_rank // (self.pp_size * self.ep_size)
        self.pp_rank = (self.global_rank // self.ep_size) % self.pp_size
        self.ep_rank = self.global_rank % self.ep_size

        self.data_world_size = self.dp_size * self.ep_size
        self.data_rank = self.dp_rank * self.ep_size + self.ep_rank

        self.is_pp_first_stage = self.pp_rank == 0
        self.is_pp_last_stage = self.pp_rank == self.pp_size - 1

        self.has_token_embed = self.is_pp_first_stage
        self.has_pos_embed = self.is_pp_first_stage
        self.has_unembed_head = self.is_pp_last_stage

        self._layer_bounds()
        self._logical_expert_bounds()

    def _layer_bounds(self):
        """
        We compute the layer bounds for each PP stage.
        The format is [start_layer, end_layer)
        """
        per_rank_layers = self.config.model.n_layers // self.config.runtime.pp_size
        self.start_layer_idx = self.pp_rank * per_rank_layers
        self.end_layer_idx = (self.pp_rank + 1) * per_rank_layers

    def _logical_expert_bounds(self):
        """
        Each EP rank holds a subset of experts and we return the indices of the experts.
        """
        self.num_experts_per_rank = self.config.model.num_experts // self.ep_size
        self.start_expert_id = self.ep_rank * self.num_experts_per_rank
        self.end_expert_id = self.start_expert_id + self.num_experts_per_rank

    def _initialize_groups(self):
        """Create groups on every rank in the same order, retaining local memberships."""

        # Rank layout: [DP, PP, EP, TP], with TP varying fastest.
        def global_rank(dp, pp, ep, tp):
            return (((dp * self.pp_size) + pp) * self.ep_size + ep) * self.tp_size + tp

        def create_group(name, ranks):
            # All ranks must call new_group, including nonmembers.
            group = dist.new_group(ranks=ranks)
            if self.global_rank in ranks:
                setattr(self, f"{name}_group_ranks", ranks)
                setattr(self, f"{name}_group", group)

        # EP exchanges tokens between expert owners at a fixed TP shard.
        for dp in range(self.dp_size):
            for pp in range(self.pp_size):
                for tp in range(self.tp_size):
                    ranks = [global_rank(dp, pp, ep, tp) for ep in range(self.ep_size)]
                    create_group("ep", ranks)

        # PP connects the same data/expert/tensor coordinates across stages.
        for dp in range(self.dp_size):
            for ep in range(self.ep_size):
                for tp in range(self.tp_size):
                    ranks = [global_rank(dp, pp, ep, tp) for pp in range(self.pp_size)]
                    create_group("pp", ranks)

        # Only experts are tensor-sharded; shared parameters replicate over DP/EP/TP.
        for pp in range(self.pp_size):
            ranks = [
                global_rank(dp, pp, ep, tp)
                for dp in range(self.dp_size)
                for ep in range(self.ep_size)
                for tp in range(self.tp_size)
            ]
            create_group("shared_dp", ranks)

        # Expert replicas share both the expert identity and the tensor shard.
        for pp in range(self.pp_size):
            for ep in range(self.ep_size):
                for tp in range(self.tp_size):
                    ranks = [global_rank(dp, pp, ep, tp) for dp in range(self.dp_size)]
                    create_group("expert_dp", ranks)

        # TP combines weight shards of the same experts within a pipeline stage.
        for dp in range(self.dp_size):
            for pp in range(self.pp_size):
                for ep in range(self.ep_size):
                    ranks = [global_rank(dp, pp, ep, tp) for tp in range(self.tp_size)]
                    create_group("tp", ranks)

        pp_rank = self.pp_group_ranks.index(self.global_rank)
        self.prev_pp_rank = -1 if pp_rank == 0 else self.pp_group_ranks[pp_rank - 1]
        self.next_pp_rank = -1 if pp_rank == self.pp_size - 1 else self.pp_group_ranks[pp_rank + 1]

    def __repr__(self):
        return f"""ParallelContext(ep_group_rank={self.ep_group_ranks}\n
            pp_group_ranks={self.pp_group_ranks}\n
            shared_dp_group_rank={self.shared_dp_group_ranks}\n
            self.expert_dp_group_rank={self.expert_dp_group_ranks})"""


def get_parallel_dims(config: AppConfig):
    return ParallelContext(config)
