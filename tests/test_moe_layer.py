import torch

from src.config import ModelConfig
from src.model.feed_fwd import MoE


def make_test_config(d_model=8, num_experts=4, top_k=2):
    return ModelConfig(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        vocab_size=1,
        d_head=1,
        n_heads=1,
        n_layers=1,
        max_seq_len=1,
        ffn_in=2 * d_model,
    )


def test_moe_output_shape():
    if torch.cuda.is_available():
        cfg = make_test_config(
            d_model=64,
            num_experts=16,
            top_k=4,
        )
        device = "cuda"
        x = torch.randn(4, 16, 64, device=device)
    else:
        cfg = make_test_config(
            d_model=8,
            num_experts=4,
            top_k=2,
        )
        device = "cpu"
        x = torch.randn(2, 4, 8, device=device)

    moe = MoE(cfg).to(device)

    y, tokens_per_expert = moe(x)

    assert y.shape == x.shape
    assert tokens_per_expert.shape == (cfg.num_experts,)
