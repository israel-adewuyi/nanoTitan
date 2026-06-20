from __future__ import annotations

import math

import einops
import torch
import torch.nn as nn
from jaxtyping import Float

from src.config import ModelConfig
from src.model.feed_fwd import MoE
from src.model_utils import ModelShardSpec


def _parameter_count(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters())


class TokenEmbed(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.token_embed = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.d_model,
        )

    def forward(
        self, input_ids: torch.Tensor[Float, "batch seq_len"]
    ) -> torch.Tensor[Float, "batch seq_len d_model"]:
        return self.token_embed(input_ids)

    def project(
        self, hidden_states: torch.Tensor[Float, "batch seq_len d_model"]
    ) -> torch.Tensor[Float, "batch seq_len vocab_size"]:
        return einops.einsum(
            self.token_embed.weight,
            hidden_states,
            "vocab_size d_model, batch seq_len d_model -> batch seq_len vocab_size",
        )


class PositionEmbed(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.max_seq_len = cfg.max_seq_len
        self.register_buffer("pos_embed", self._build_sinusoidal_pos_embed(cfg))

    @staticmethod
    def _build_sinusoidal_pos_embed(cfg: ModelConfig) -> torch.Tensor:
        indices = torch.arange(cfg.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, cfg.d_model, 2) * (-math.log(10000.0) / cfg.d_model))
        pos_embed = torch.zeros(cfg.max_seq_len, cfg.d_model)
        pos_embed[:, 0::2] = torch.sin(indices * div_term)
        pos_embed[:, 1::2] = torch.cos(indices * div_term)
        return pos_embed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds configured max_seq_len={self.max_seq_len}"
            )
        return self.pos_embed[:seq_len].unsqueeze(0)


class EmbeddingBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = TokenEmbed(self.cfg)
        self.position_embed = PositionEmbed(self.cfg)

    def forward(self, x: torch.Tensor):
        return self.token_embed(x) + self.position_embed(x)


class Unembed(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.unembed = nn.Linear(self.cfg.d_model, self.cfg.vocab_size, bias=False)

    def forward(self, x: torch.Tensor):
        return self.unembed(x)


class LayerNorm(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.gamma = nn.Parameter(torch.ones(self.d_model))
        self.beta = nn.Parameter(torch.zeros(self.d_model))

    def forward(
        self, x: torch.Tensor[Float, "batch seq_len d_model"]
    ) -> torch.Tensor[Float, "batch seq_len d_model"]:
        x_mean = x.mean(dim=-1, keepdim=True)
        x_var = x.var(dim=-1, keepdim=True)
        normalized_x = (x - x_mean) / torch.sqrt(x_var + 1e-8)
        return normalized_x * self.gamma + self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.Tensor(cfg.d_model, cfg.d_head * cfg.n_heads))
        self.W_K = nn.Parameter(torch.Tensor(cfg.d_model, cfg.d_head * cfg.n_heads))
        self.W_V = nn.Parameter(torch.Tensor(cfg.d_model, cfg.d_head * cfg.n_heads))
        self.W_O = nn.Parameter(torch.Tensor(cfg.d_head * cfg.n_heads, cfg.d_model))

        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)
        nn.init.xavier_uniform_(self.W_V)
        nn.init.xavier_uniform_(self.W_O)

        # Register mask. This implementation assumes that all residual streams have the same sequence length
        mask = torch.tril(torch.ones(self.cfg.max_seq_len, self.cfg.max_seq_len, dtype=torch.bool))[
            None, None, :, :
        ]
        self.register_buffer("mask", mask)

    def forward(
        self, x: torch.Tensor[Float, "batch seq_len d_model"]
    ) -> torch.Tensor[Float, "batch seq_len d_model"]:
        _, seq_len, _ = x.shape
        Q = einops.einsum(
            x,
            self.W_Q,
            "batch seq_len d_model, d_model d_out -> batch seq_len d_out",
        )
        Q = einops.rearrange(
            Q,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=self.cfg.n_heads,
            d_head=self.cfg.d_head,
        )
        K = einops.einsum(
            x,
            self.W_K,
            "batch seq_len d_model, d_model d_out -> batch seq_len d_out",
        )
        K = einops.rearrange(
            K,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=self.cfg.n_heads,
            d_head=self.cfg.d_head,
        )
        V = einops.einsum(
            x,
            self.W_V,
            "batch seq_len d_model, d_model d_out -> batch seq_len d_out",
        )
        V = einops.rearrange(
            V,
            "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=self.cfg.n_heads,
            d_head=self.cfg.d_head,
        )
        attn_scores = einops.einsum(
            Q,
            K,
            "batch seq_Q n_heads d_head, batch seq_K n_heads d_head -> batch n_heads seq_Q seq_K",
        ) / math.sqrt(self.cfg.d_head)
        masked_attn_scores = attn_scores.masked_fill(self.mask == 0, float("-inf"))
        attn_pattern = torch.softmax(masked_attn_scores, dim=-1)
        attn_out = einops.einsum(
            attn_pattern,
            V,
            "batch n_heads seq_Q seq_K, batch seq_K n_heads d_head -> batch seq_Q n_heads d_head",
        )
        attn_out = einops.rearrange(
            attn_out, "batch seq_Q n_heads d_head -> batch seq_Q (n_heads d_head)"
        )

        out = einops.einsum(attn_out, self.W_O, "batch seq d_in, d_in d_model -> batch seq d_model")
        return out


class TransformerLayer(nn.Module):
    """A transformer layer"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.attn = MultiHeadAttention(cfg)
        self.ffn = MoE(cfg)
        self.attn_norm = LayerNorm(cfg)
        self.ffn_norm = LayerNorm(cfg)

    def active_parameter_count(self) -> int:
        return (
            _parameter_count(self.attn)
            + _parameter_count(self.attn_norm)
            + _parameter_count(self.ffn_norm)
            + self.ffn.active_parameter_count()
        )

    def forward(
        self,
        x: torch.Tensor[Float, "batch seq_len d_model"],
        return_moe_stats: bool = False,
    ) -> torch.Tensor[Float, "batch seq_len d_model"] | tuple[torch.Tensor, torch.Tensor]:
        x = self.attn(self.attn_norm(x)) + x
        out, tokens_per_expert = self.ffn(self.ffn_norm(x))
        x = out + x
        if return_moe_stats:
            return x, tokens_per_expert
        return x


class NanoTitanModel(nn.Module):
    def __init__(self, cfg: ModelConfig, spec: ModelShardSpec | None = None):
        super().__init__()
        self.cfg = cfg
        self.spec = spec
        self.blocks = nn.ModuleList()

        if self.spec is None:
            self.token_embed = TokenEmbed(self.cfg)
            self.position_embed = PositionEmbed(self.cfg)
            self.layers = nn.ModuleList(
                TransformerLayer(self.cfg) for _ in range(self.cfg.n_layers)
            )
        else:
            if self.spec.has_token_embed and self.spec.has_pos_embed:
                self.blocks.append(EmbeddingBlock(cfg))

            for _ in range(self.spec.layer_start, self.spec.layer_end):
                self.blocks.append(TransformerLayer(cfg))

            if self.spec.has_unembed_head:
                self.blocks.append(Unembed(self.cfg))

    @classmethod
    def from_specs(cls, cfg: ModelConfig, spec: ModelShardSpec):
        return cls(cfg, spec)

    def total_parameter_count(self) -> int:
        return _parameter_count(self)

    def active_parameter_count(self) -> int:
        return _parameter_count(self.token_embed) + sum(
            layer.active_parameter_count() for layer in self.layers
        )

    def forward(
        self, x: torch.Tensor, return_moe_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        for block in self.blocks:
            x = block(x)
        # moe_stats = []
        # for layer in self.layers:
        #     if return_moe_stats:
        #         x, tokens_per_expert = layer(x, return_moe_stats=True)
        #         moe_stats.append(tokens_per_expert.detach())
        #     else:
        #         x = layer(x)
        # x = self.token_embed.project(x)
        # if return_moe_stats:
        #     return x, moe_stats
        return x
