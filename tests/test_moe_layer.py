import torch
import pytest

from src.config import ModelConfig
from src.model.feed_fwd import MoE
from src.utils import resolve_dtype
from src.model.model import NanoTitanModel
from src.model_utils import ModelShardSpec


def make_test_config(d_model=8, num_experts=4, top_k=2, moe_backend="torch"):
    cfg = ModelConfig(
        d_model=d_model,
        num_experts=num_experts,
        top_k=top_k,
        vocab_size=1,
        d_head=1,
        n_heads=1,
        n_layers=1,
        max_seq_len=1,
        ffn_in=2 * d_model,
        moe_backend=moe_backend,
    )
    cfg.dtype = resolve_dtype(cfg.dtype)
    return cfg


def test_moe_output_shape():
    if torch.cuda.is_available():
        cfg = make_test_config(
            d_model=64,
            num_experts=16,
            top_k=4,
            moe_backend="cuda",
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

@pytest.mark.skip(reason="WIP")
def test_model_can_return_moe_stats():
    cfg = make_test_config()
    
    spec = ModelShardSpec(
        layer_start=0,
        layer_end=2,
        has_token_embed=False,
        has_pos_embed=False,
        has_unembed_head=False
    )
    
    model = NanoTitanModel(cfg, spec)
    input_ids = torch.zeros((2, cfg.max_seq_len), dtype=torch.long)

    logits, moe_stats = model(input_ids, return_moe_stats=True)

    assert logits.shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert len(moe_stats) == cfg.n_layers
    assert moe_stats[0].shape == (cfg.num_experts,)
    assert moe_stats[0].sum().item() == input_ids.numel() * cfg.top_k


def test_model_active_parameter_count_uses_top_k_experts():
    cfg = make_test_config()
    spec = ModelShardSpec(
        layer_start=0,
        layer_end=cfg.n_layers,
        has_token_embed=False,
        has_pos_embed=False,
        has_unembed_head=False
    )
    
    model = NanoTitanModel(cfg, spec)

    attention_params = 4 * cfg.d_model * cfg.d_head * cfg.n_heads
    layer_norm_params = 4 * cfg.d_model
    router_params = cfg.d_model * cfg.num_experts
    expert_params = 3 * cfg.d_model * cfg.ffn_in
    expected_active_params = cfg.n_layers * (
        attention_params + layer_norm_params + router_params + cfg.top_k * expert_params
    )

    assert model.total_parameter_count() == sum(param.numel() for param in model.parameters())
    assert model.active_parameter_count() == expected_active_params
