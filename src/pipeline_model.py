from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float

from src.dist_env import is_main_process
from src.model import NanoTitanModel


class PipelineStageModel(nn.Module):
    """A pipeline staged model. It holds and is responsible for a stage on a device"""

    def __init__(
        self,
        model: NanoTitanModel,
        rank: int,
        cfg,  # I am not sure of the type of this guy
        start_idx: int,
        end_idx: int,
        device: str,
    ):
        super().__init__()
        self.model = model
        self.rank = rank
        self.device = device
        self.cfg = cfg
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.partition_params()

    def partition_params(self):
        self.stage = nn.ModuleList()

        if is_main_process():
            # rank 0 should hold the embedding layer and the pos embed layer
            self.token_embed = self.model.token_embed.to(self.device)
            self.pos_embed = self.model.position_embed.to(self.device)

        # each device holds a subset of layers in the model
        for layer in range(self.cfg.model.n_layers):
            if layer >= self.start_idx and layer < self.end_idx:
                self.stage.append(self.model.layers[layer].to(self.device))

    def forward(
        self, x: torch.Tensor[Float, "batch seq_len"]
    ) -> torch.Tensor[Float, "batch seq_len"]:
        # forward pass on the specific stage each device is holding
        if is_main_process():
            token_embed = self.token_embed(x)
            x = token_embed + self.pos_embed(x)

        for layer in self.stage:
            x = layer(x)

        if is_main_process():
            x = self.token_embed.project(x)

        return x
