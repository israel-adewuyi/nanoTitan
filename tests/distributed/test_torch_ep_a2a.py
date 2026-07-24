import os

import pytest
import torch
import torch.distributed as dist

from src.config import ModelConfig
from src.model.feed_fwd import MoE
from src.model.utils import ModelShardSpec
from src.utils import resolve_dtype

WORLD_SIZE = 2
NUM_EXPERTS = 4
EXPERTS_PER_RANK = NUM_EXPERTS // WORLD_SIZE
LOCAL_BATCH_SIZE = 1


def _make_config() -> ModelConfig:
    cfg = ModelConfig(
        d_model=4,
        num_experts=NUM_EXPERTS,
        top_k=2,
        vocab_size=1,
        d_head=1,
        n_heads=1,
        n_layers=1,
        max_seq_len=4,
        ffn_in=8,
        moe_backend="torch",
        dtype="float32",
        moe_router_dtype="float32",
        router_alpha=0.01,
    )
    cfg.dtype = resolve_dtype(cfg.dtype)
    cfg.moe_router_dtype = resolve_dtype(cfg.moe_router_dtype)
    return cfg


def _make_reference_spec() -> ModelShardSpec:
    return ModelShardSpec(
        layer_start=0,
        layer_end=1,
        has_token_embed=False,
        has_pos_embed=False,
        has_unembed_head=False,
        per_rank_expert=NUM_EXPERTS,
        start_expert_id=0,
        end_expert_id=NUM_EXPERTS,
    )


def _make_sharded_spec(rank: int) -> ModelShardSpec:
    start_expert_id = rank * EXPERTS_PER_RANK
    return ModelShardSpec(
        layer_start=0,
        layer_end=1,
        has_token_embed=False,
        has_pos_embed=False,
        has_unembed_head=False,
        per_rank_expert=EXPERTS_PER_RANK,
        start_expert_id=start_expert_id,
        end_expert_id=start_expert_id + EXPERTS_PER_RANK,
        ep_size=WORLD_SIZE,
        ep_group=dist.group.WORLD,
    )


def _build_models(cfg: ModelConfig, rank: int, device: torch.device) -> tuple[MoE, MoE]:
    # Construct on CPU with the same seed on every rank so that each process has
    # an identical, full reference model.
    torch.manual_seed(2026)
    reference = MoE(cfg, _make_reference_spec())
    sharded = MoE(cfg, _make_sharded_spec(rank))

    start = rank * EXPERTS_PER_RANK
    end = start + EXPERTS_PER_RANK

    with torch.no_grad():
        # Identity routing plus the inputs below guarantees assignments to both
        # expert ranks from each source rank.
        reference.router.weight.copy_(torch.eye(NUM_EXPERTS, cfg.d_model))
        sharded.router.weight.copy_(reference.router.weight)

        for name in ("W_gate", "W_val", "W_out"):
            reference_weight = getattr(reference.experts, name)
            getattr(sharded.experts, name).copy_(reference_weight[start:end])

    return reference.to(device), sharded.to(device)


def _make_global_input(device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [
                [4.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 3.0],
                [4.0, 0.0, 3.0, 0.0],
                [0.0, 4.0, 0.0, 3.0],
            ],
            [
                [3.0, 4.0, 0.0, 0.0],
                [0.0, 0.0, 3.0, 4.0],
                [3.0, 0.0, 4.0, 0.0],
                [0.0, 3.0, 0.0, 4.0],
            ],
        ],
        device=device,
    )


@pytest.mark.distributed
def test_two_rank_torch_ep_matches_unsharded_reference():
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        pytest.skip("launch with torchrun --standalone --nproc-per-node=2")

    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        use_nccl = torch.cuda.is_available() and dist.is_nccl_available()
        backend = "nccl" if use_nccl else "gloo"
        if use_nccl:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group(backend=backend, init_method="env://")

    rank = dist.get_rank()
    if dist.get_world_size() != WORLD_SIZE:
        pytest.fail(f"expected {WORLD_SIZE} ranks, got {dist.get_world_size()}")

    if dist.get_backend() == "nccl":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    try:
        cfg = _make_config()
        reference, sharded = _build_models(cfg, rank, device)

        reference_input = _make_global_input(device).requires_grad_(True)
        local_start = rank * LOCAL_BATCH_SIZE
        local_end = local_start + LOCAL_BATCH_SIZE
        sharded_input = reference_input.detach()[local_start:local_end].clone().requires_grad_(True)

        reference_output, reference_counts = reference(reference_input)
        sharded_output, local_counts = sharded(sharded_input)

        output_gradient = torch.linspace(
            -1.0,
            1.0,
            steps=reference_output.numel(),
            device=device,
            dtype=reference_output.dtype,
        ).reshape_as(reference_output)

        reference_output.backward(output_gradient)
        sharded_output.backward(output_gradient[local_start:local_end].contiguous())

        global_counts = local_counts.clone()
        dist.all_reduce(global_counts, group=dist.group.WORLD)

        global_router_gradient = sharded.router.weight.grad.clone()
        dist.all_reduce(global_router_gradient, group=dist.group.WORLD)

        expected_output = reference_output.detach()[local_start:local_end]
        expected_input_gradient = reference_input.grad[local_start:local_end]
        start_expert = rank * EXPERTS_PER_RANK
        end_expert = start_expert + EXPERTS_PER_RANK

        torch.testing.assert_close(sharded_output, expected_output, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            sharded_input.grad,
            expected_input_gradient,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(global_counts, reference_counts)
        torch.testing.assert_close(
            global_router_gradient,
            reference.router.weight.grad,
            rtol=1e-5,
            atol=1e-6,
        )

        for name in ("W_gate", "W_val", "W_out"):
            sharded_gradient = getattr(sharded.experts, name).grad
            reference_gradient = getattr(reference.experts, name).grad[start_expert:end_expert]
            torch.testing.assert_close(
                sharded_gradient,
                reference_gradient,
                rtol=1e-5,
                atol=1e-6,
            )

        assignments_per_destination = local_counts.view(WORLD_SIZE, EXPERTS_PER_RANK).sum(dim=1)
        assert torch.all(assignments_per_destination > 0), (
            "each source rank must send assignments to both expert ranks"
        )
    finally:
        if owns_process_group:
            dist.destroy_process_group()
