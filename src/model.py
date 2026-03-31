from __future__ import annotations

import math

import einops
import torch
import torch.nn as nn
from jaxtyping import Float

from src.config import ModelConfig


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
            Q, "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=self.cfg.n_heads, d_head=self.cfg.d_head
        )
        K = einops.einsum(
            x,
            self.W_K,
            "batch seq_len d_model, d_model d_out -> batch seq_len d_out",
        )
        K = einops.rearrange(
            K, "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=self.cfg.n_heads, d_head=self.cfg.d_head
        )
        V = einops.einsum(
            x,
            self.W_V,
            "batch seq_len d_model, d_model d_out -> batch seq_len d_out",
        )
        V = einops.rearrange(
            V, "batch seq_len (n_heads d_head) -> batch seq_len n_heads d_head",
            n_heads=self.cfg.n_heads, d_head=self.cfg.d_head
        )
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))[None, None, :, :].to(
            x.device
        )
        attn_scores = einops.einsum(
            Q,
            K,
            "batch seq_Q n_heads d_head, batch seq_K n_heads d_head -> batch n_heads seq_Q seq_K",
        ) / math.sqrt(self.cfg.d_head)
        masked_attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
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


class FFN(nn.Module):
    """Feedforward network implementation"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.W_in = nn.Linear(cfg.d_model, cfg.ffn_in)
        self.W_out = nn.Linear(cfg.ffn_in, cfg.d_model)
        self.ReLU = nn.ReLU()

    def forward(
        self, x: torch.Tensor[Float, "batch seq_len d_model"]
    ) -> torch.Tensor[Float, "batch seq_len d_model"]:
        return self.W_out(self.ReLU(self.W_in(x)))


class TransformerLayer(nn.Module):
    """A transformer layer"""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.attn = MultiHeadAttention(cfg)
        self.ffn = FFN(cfg)

    def forward(
        self, x: torch.Tensor[Float, "batch seq_len d_model"]
    ) -> torch.Tensor[Float, "batch seq_len d_model"]:
        x = self.attn(x) + x
        x = self.ffn(x) + x
        return x


class NanoTitanModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embed = TokenEmbed(cfg)
        self.position_embed = PositionEmbed(cfg)
        self.layers = nn.ModuleList(TransformerLayer(cfg) for _ in range(cfg.n_layers))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_emb = self.token_embed(input_ids)
        pos_embed = self.position_embed(token_emb)
        x = token_emb + pos_embed
        for layer in self.layers:
            x = layer(x)
        x = self.token_embed.project(x)
        return x
