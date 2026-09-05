import pytest
import torch

from src.config import ModelConfig
from src.model.feed_fwd import MoE
from src.model.model import NanoTitanModel
from src.model.utils import ModelShardSpec, MoELayerStats, topk_with_capacity
from src.utils import resolve_dtype


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
        router_alpha=0.01,
    )
    cfg.dtype = resolve_dtype(cfg.dtype)
    cfg.moe_router_dtype = resolve_dtype(cfg.moe_router_dtype)
    return cfg


def make_test_spec(cfg, layer_end=None):
    return ModelShardSpec(
        layer_start=0,
        layer_end=cfg.n_layers if layer_end is None else layer_end,
        has_token_embed=False,
        has_pos_embed=False,
        has_unembed_head=False,
        per_rank_expert=cfg.num_experts,
        start_expert_id=0,
        end_expert_id=cfg.num_experts,
    )


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

    moe = MoE(cfg, make_test_spec(cfg)).to(device)

    y, _ = moe(x)

    assert y.shape == x.shape


@pytest.mark.parametrize(
    ("counts", "top_k", "expected"),
    [
        ([2, 2, 2, 2], 2, 0.0),
        ([3, 3, 1, 1], 2, 0.5),
        ([4, 4, 0, 0], 2, 1.0),
        ([4, 0, 0, 0], 1, 3.0),
        ([1, 0, 0, 0], 1, 3.0),
    ],
)
def test_moe_max_vio_is_scalar_relative_overload(counts, top_k, expected):
    cfg = make_test_config(top_k=top_k)
    num_tokens = sum(counts) // top_k
    logits = torch.zeros(num_tokens, cfg.num_experts, requires_grad=True)
    stats = MoELayerStats(torch.tensor(counts), logits.softmax(dim=-1), cfg)

    assert stats.max_vio.ndim == 0
    assert stats.max_vio.item() == pytest.approx(expected)
    assert not stats.max_vio.requires_grad
    assert stats.aux_loss.requires_grad


def test_capacity_routing_reports_raw_counts_for_aux_loss():
    cfg = make_test_config(d_model=4, num_experts=4, top_k=1)
    cfg.capacity_factor = 1.0
    moe = MoE(cfg, make_test_spec(cfg))

    with torch.no_grad():
        moe.router.weight.copy_(torch.eye(4))

    x = torch.tensor([[[8.0, 6.0, 2.0, 0.0]]]).repeat(1, 4, 1)
    _, stats = moe(x)

    torch.testing.assert_close(
        stats.tokens_per_expert,
        torch.tensor([4, 0, 0, 0], dtype=stats.tokens_per_expert.dtype),
    )
    assert stats.max_vio.item() == pytest.approx(3.0)


def test_capacity_routing_normalizes_selected_logits_without_nan():
    logits = torch.tensor([[1000.0, 0.0, -1000.0, -2000.0]]).repeat(4, 1)

    weights, _, _, _ = topk_with_capacity(logits, top_k=2, capacity_factor=1.0)

    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(4))


@pytest.mark.cuda
def test_cuda_moe_matches_torch_forward_and_backward():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pytest.importorskip("nanotitan_cuda")

    torch.manual_seed(0)
    cfg_torch = make_test_config()
    cfg_cuda = make_test_config(moe_backend="cuda")
    moe_torch = MoE(cfg_torch, make_test_spec(cfg_torch)).to("cuda")
    moe_cuda = MoE(cfg_cuda, make_test_spec(cfg_cuda)).to("cuda")

    with torch.no_grad():
        moe_torch.router.weight.copy_(torch.eye(cfg_torch.num_experts, cfg_torch.d_model))
    moe_cuda.load_state_dict(moe_torch.state_dict())

    x_torch = torch.tensor(
        [
            [
                [4, 3, 0, 0, 0, 0, 0, 0],
                [0, 0, 4, 3, 0, 0, 0, 0],
                [4, 0, 3, 0, 0, 0, 0, 0],
                [0, 4, 0, 3, 0, 0, 0, 0],
            ]
        ],
        dtype=torch.float32,
        device="cuda",
        requires_grad=True,
    )
    x_cuda = x_torch.detach().clone().requires_grad_(True)

    out_torch, stats_torch = moe_torch(x_torch)
    out_cuda, stats_cuda = moe_cuda(x_cuda)
    tolerance = {"rtol": 1e-4, "atol": 5e-4}

    torch.testing.assert_close(out_cuda, out_torch, **tolerance)
    torch.testing.assert_close(
        stats_cuda.tokens_per_expert, stats_torch.tokens_per_expert, check_dtype=False
    )
    torch.testing.assert_close(stats_cuda.probs_per_expert, stats_torch.probs_per_expert)
    torch.testing.assert_close(stats_cuda.aux_loss, stats_torch.aux_loss)
    torch.testing.assert_close(stats_cuda.max_vio, stats_torch.max_vio)

    out_grad = torch.linspace(-1, 1, out_torch.numel(), device="cuda").reshape_as(out_torch)
    ((out_torch * out_grad).sum() + stats_torch.aux_loss).backward()
    ((out_cuda * out_grad).sum() + stats_cuda.aux_loss).backward()

    torch.testing.assert_close(x_cuda.grad, x_torch.grad, **tolerance)
    cuda_params = dict(moe_cuda.named_parameters())
    for name, param in moe_torch.named_parameters():
        torch.testing.assert_close(cuda_params[name].grad, param.grad, **tolerance)


@pytest.mark.skip(reason="WIP")
def test_model_can_return_moe_stats():
    cfg = make_test_config()

    spec = make_test_spec(cfg, layer_end=2)

    model = NanoTitanModel(cfg, spec)
    input_ids = torch.zeros((2, cfg.max_seq_len), dtype=torch.long)

    logits, moe_stats = model(input_ids, return_moe_stats=True)

    assert logits.shape == (2, cfg.max_seq_len, cfg.vocab_size)
    assert len(moe_stats) == cfg.n_layers
    assert moe_stats[0].shape == (cfg.num_experts,)
    assert moe_stats[0].sum().item() == input_ids.numel() * cfg.top_k


def test_model_active_parameter_count_uses_top_k_experts():
    cfg = make_test_config()
    spec = make_test_spec(cfg)

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


def test_parameter_sync_groups_handle_pipeline_boundary_blocks():
    cfg = make_test_config()
    spec = ModelShardSpec(
        layer_start=0,
        layer_end=cfg.n_layers,
        has_token_embed=True,
        has_pos_embed=True,
        has_unembed_head=True,
        per_rank_expert=cfg.num_experts,
        start_expert_id=0,
        end_expert_id=cfg.num_experts,
    )
    model = NanoTitanModel(cfg, spec)

    groups = model.parameter_sync_groups()
    shared_ids = {id(param) for param in groups["shared"]}
    expert_ids = {id(param) for param in groups["expert"]}
    trainable_ids = {id(param) for param in model.parameters() if param.requires_grad}

    assert shared_ids
    assert expert_ids
    assert shared_ids.isdisjoint(expert_ids)
    assert shared_ids | expert_ids == trainable_ids
