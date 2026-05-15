from __future__ import annotations

import torch
import torch.nn as nn

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
        is_first_stage: bool,
        is_last_stage: bool,
    ):
        super().__init__()
        self.rank = rank
        self.device = device
        self.cfg = cfg
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage

        self.partition_params(model)

        self.stage_inputs, self.stage_outputs = [], []
        self.microbatch_idx = 0

    def partition_params(self, model):
        self.stage = nn.ModuleList()

        if self.is_first_stage:
            # rank 0 should hold the embedding layer and the pos embed layer
            self.token_embed = model.token_embed.to(self.device)
            self.pos_embed = model.position_embed.to(self.device)

        # each device holds a subset of layers in the model
        for layer in range(self.cfg.model.n_layers):
            if layer >= self.start_idx and layer < self.end_idx:
                self.stage.append(model.layers[layer].to(self.device))

        if self.is_last_stage:
            self.token_embed = model.token_embed.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on the specific stage each device is holding"""
        if not self.is_first_stage:
            self.stage_input = x
            self.stage_input.requires_grad_()
            self.stage_inputs.append(self.stage_input)

        if self.is_first_stage:
            token_embed = self.token_embed(x)
            x = token_embed + self.pos_embed(x)

        for layer in self.stage:
            x = layer(x)

        if self.is_last_stage:
            x = self.token_embed.project(x)

        self.stage_outputs.append(x)

        return x

    def get_incoming_acts_grad(self, microbatch_idx: int = 0):
        return self.stage_inputs[microbatch_idx].grad

    def get_outgoing_acts(self, microbatch_idx: int = 0):
        return self.stage_outputs[microbatch_idx]

    def clear_cache_acts(self) -> None:
        self.stage_inputs = []
        self.stage_outputs = []
