from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.config import ModelConfig, RuntimeConfig
from src.model.model import NanoTitanModel
from src.parallel import DataParallel, PipelineParallel
from src.parallel_dims import ParallelContext
from tests.helpers import make_single_rank_context

pytestmark = [
    pytest.mark.distributed,
    pytest.mark.parametrize(
        "dtype", [torch.float32, torch.float16, torch.bfloat16], ids=["fp32", "fp16", "bf16"]
    ),
]
TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float16: {"rtol": 1e-2, "atol": 1e-3},
    torch.bfloat16: {"rtol": 5e-2, "atol": 5e-3},
}


def _setup(device, dtype, *, dp_size=1, pp_size=1):
    if device.type == "cuda" and dtype == torch.bfloat16:
        supported = torch.tensor(int(torch.cuda.is_bf16_supported()), device=device)
        dist.all_reduce(supported, op=dist.ReduceOp.MIN)
        if not supported.item():
            pytest.skip("BF16 requires support on both ranks")

    model_cfg = ModelConfig(
        vocab_size=16,
        d_model=8,
        d_head=4,
        n_heads=2,
        n_layers=2,
        max_seq_len=4,
        ffn_in=16,
        num_experts=4,
        top_k=2,
        # Each expert can accept every token, so splitting the batch cannot reroute it.
        capacity_factor=2.0,
        router_alpha=0.01,
    )
    model_cfg.dtype = dtype
    # The MoE backend requires FP32 routing weights, including for FP16/BF16 models.
    model_cfg.moe_router_dtype = torch.float32
    cfg = SimpleNamespace(
        model=model_cfg,
        runtime=RuntimeConfig(dp_size=dp_size, pp_size=pp_size, num_expert=4, bucket_size=1),
    )
    dims = ParallelContext(cfg)
    reference_spec = make_single_rank_context(model_cfg)

    torch.manual_seed(2026)
    reference = NanoTitanModel(model_cfg, reference_spec).to(device)
    model = NanoTitanModel(model_cfg, dims).to(device)
    # The full model has an embedding block before its transformer layers.
    start = dims.start_layer_idx + int(not dims.has_token_embed)
    reference_blocks = reference.blocks[start : start + len(model.blocks)]
    model.blocks.load_state_dict(reference_blocks.state_dict())
    return cfg, dims, reference, model, reference_blocks


def _assert_gradients(model, reference_blocks, dtype):
    for name, param in model.blocks.named_parameters():
        assert param.grad is not None, name
        torch.testing.assert_close(
            param.grad, reference_blocks.get_parameter(name).grad, **TOLERANCES[dtype]
        )


def test_two_rank_dp_training_step_matches_reference(two_ranks, dtype):
    rank, device = two_ranks
    cfg, dims, reference, model, reference_blocks = _setup(device, dtype, dp_size=2)
    dp = DataParallel(cfg, dims)
    dp.device = device
    dp.prepare_model(model)

    x = torch.arange(8, device=device).reshape(2, 4)
    targets = (x + 1) % cfg.model.vocab_size
    reference_output, _ = reference(x)
    # CE alone is additive across ranks; rank-local auxiliary losses need not be.
    F.cross_entropy(reference_output.flatten(0, 1), targets.flatten()).backward()

    for reducer in dp.get_reducers().values():
        reducer.backward_grad_sync = True
    output, _ = model(x[rank : rank + 1])
    F.cross_entropy(output.flatten(0, 1), targets[rank : rank + 1].flatten()).backward()
    for reducer in dp.get_reducers().values():
        reducer.prepare_missing_grad()
    dp.finalize_backward()

    assert output.dtype == dtype
    torch.testing.assert_close(output, reference_output[rank : rank + 1], **TOLERANCES[dtype])
    _assert_gradients(model, reference_blocks, dtype)

    # SGD preserves gradient magnitude, making incorrect averaging visible in the update.
    for candidate in (model, reference):
        torch.optim.SGD(candidate.parameters(), lr=0.1).step()
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, reference.get_parameter(name), **TOLERANCES[dtype])


def test_two_stage_pp_forward_backward_matches_reference(two_ranks, dtype):
    _, device = two_ranks
    cfg, dims, reference, model, reference_blocks = _setup(device, dtype, pp_size=2)
    pipeline = PipelineParallel(cfg, dims, reducers={})
    pipeline.device = device

    x = torch.arange(8, device=device).reshape(2, 4)
    targets = (x + 1) % cfg.model.vocab_size
    reference_output, stats = reference(x)
    reference_loss = F.cross_entropy(reference_output.flatten(0, 1), targets.flatten())
    reference_aux_loss = torch.stack([layer.aux_loss for layer in stats]).mean()
    (reference_loss + reference_aux_loss).backward()

    # Exercise one microbatch through the production activation/gradient transfers.
    pipeline.microbatch_size = x.shape[0]
    pipeline.ce_losses = []
    pipeline.moe_aux_losses = []
    pipeline.moe_route_counts = []
    stage_input = (
        x
        if dims.is_pp_first_stage
        else pipeline.comms.recv_forward(0, pipeline._activation_buffer())
    )
    output = pipeline.forward_microbatch(0, model, stage_input, targets)
    if dims.is_pp_last_stage:
        input_grad = pipeline.backward_microbatch(0)
        pipeline.comms.send_backward(0, input_grad)
    else:
        pipeline.comms.send_forward(0, output)
        output_grad = pipeline.comms.recv_backward(0, torch.empty_like(output))
        pipeline.backward_microbatch(0, output_grad)

    assert output.dtype == dtype
    if dims.is_pp_last_stage:
        torch.testing.assert_close(output, reference_output, **TOLERANCES[dtype])
        torch.testing.assert_close(
            pipeline.ce_losses[0], reference_loss.detach(), **TOLERANCES[dtype]
        )
    _assert_gradients(model, reference_blocks, dtype)
