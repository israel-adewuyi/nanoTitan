from dataclasses import dataclass

import torch

from src.config import ModelConfig


@dataclass
class MoELayerStats:
    tokens_per_expert: torch.Tensor
    probs_per_expert: torch.Tensor
    cfg: ModelConfig
    aux_loss: torch.Tensor | None = None

    def __post_init__(
        self,
    ):
        self.num_tokens = self.probs_per_expert.shape[0]
        self.total_assignments = self.tokens_per_expert.sum()
        self.ass_frac_per_expert = self.tokens_per_expert.float() / self.total_assignments

        self.probs_per_expert = self.probs_per_expert.mean(dim=0)

        assert self.ass_frac_per_expert.shape == self.probs_per_expert.shape
        assert self.ass_frac_per_expert.ndim == 1

        self.aux_loss = (
            self.cfg.router_alpha
            * self.cfg.num_experts
            * torch.sum(self.ass_frac_per_expert * self.probs_per_expert)
        )
