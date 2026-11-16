import copy
from types import SimpleNamespace

import pytest
import torch

from src.config import load_config
from src.model.model import LayerNorm, NanoTitanModel, RMSNorm
from src.optim import Float32AdamW
from src.utils import squared_gradient_norm


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_adam_preserves_float32_updates_and_resumes(dtype):
    parameter = torch.nn.Parameter(torch.ones(3, dtype=dtype))
    reference = torch.nn.Parameter(parameter.detach().float())
    optimizer = Float32AdamW([parameter], lr=5e-4, weight_decay=0.1)
    expected = torch.optim.AdamW([reference], lr=5e-4, weight_decay=0.1)
    for _ in range(3):
        gradient = torch.tensor([0, 1e-5, -1e-4], dtype=dtype)
        parameter.grad = gradient.clone()
        reference.grad = gradient.float()
        optimizer.step()
        expected.step()
        torch.testing.assert_close(parameter, reference.to(dtype), rtol=0, atol=0)
        assert bool(parameter.isfinite().all())
        assert optimizer.state[optimizer.master_parameters[0]]["exp_avg_sq"].dtype == torch.float32
        saved = copy.deepcopy(optimizer.state_dict())
        optimizer.load_state_dict(saved)
        optimizer.zero_grad()
        assert parameter.grad is None


@pytest.mark.parametrize("norm", [LayerNorm, RMSNorm])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_normalization_handles_zero_and_large_half_inputs(norm, dtype):
    layer = norm(SimpleNamespace(d_model=4, dtype=dtype))
    value = torch.tensor([[0, 0, 0, 0], [1000, -1000, 1, -1]], dtype=dtype, requires_grad=True)
    result = layer(value)
    assert result.dtype == dtype
    assert bool(result.isfinite().all())
    result.float().square().sum().backward()
    assert bool(value.grad.isfinite().all())


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_squared_gradient_norm_avoids_half_precision_overflow(dtype):
    result = squared_gradient_norm([torch.full((4,), 1000.0, dtype=dtype)])
    assert result.dtype == torch.float32
    assert result.item() == pytest.approx(4_000_000.0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_tiny_moe_cpu_optimizer_steps_remain_finite(dtype):
    cfg = load_config("configs/dist.toml").model.model_copy(update=dict(
        vocab_size=32, d_model=8, d_head=4, n_heads=2, n_layers=2,
        max_seq_len=8, ffn_in=16, num_experts=4, top_k=2,
        moe_backend="torch", dtype=dtype, moe_router_dtype=torch.float32))
    spec = SimpleNamespace(has_token_embed=True, has_pos_embed=True, has_unembed_head=True,
        start_layer_idx=0, end_layer_idx=2, num_experts_per_rank=4, ep_size=1)
    torch.manual_seed(42)
    model = NanoTitanModel.from_specs(cfg, spec)
    optimizer = Float32AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    tokens = torch.randint(0, 32, (1, 8))
    for _ in range(3):
        optimizer.zero_grad()
        logits, _ = model(tokens)
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, 32), tokens.flatten())
        assert bool(loss.isfinite())
        loss.backward()
        assert all(bool(p.grad.isfinite().all()) for p in model.parameters() if p.grad is not None)
        optimizer.step()
        assert all(bool(p.isfinite().all()) for p in model.parameters())
