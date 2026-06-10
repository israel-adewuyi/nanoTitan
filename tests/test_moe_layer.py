import torch

from src.config import ModelConfig
from src.model.feed_fwd import MoE
from src.model.model import NanoTitanModel


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


def test_model_can_return_moe_stats():
    cfg = make_test_config()
    model = NanoTitanModel(cfg)
    input_ids = torch.zeros((2, cfg.max_seq_len), dtype=torch.long)

    logits, moe_stats = model(input_ids, return_moe_stats=True)

    assert logits.shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert len(moe_stats) == cfg.n_layers
    assert moe_stats[0].shape == (cfg.num_experts,)
    assert moe_stats[0].sum().item() == input_ids.numel() * cfg.top_k
