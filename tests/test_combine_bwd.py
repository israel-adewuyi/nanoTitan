import pytest
import torch

from src.model.moe_ops import combine_tokens_fn

TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float16: {"rtol": 1e-2, "atol": 1e-2},
    torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens,assignments,d_model", [(3, 6, 8), (5, 11, 7)])
def test_combine_backward_matches_reference(dtype, num_tokens, assignments, d_model):
    torch.manual_seed(0)
    device = "cuda"

    expert_out_ref = torch.randn(
        assignments,
        d_model,
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    gate_ref = torch.randn(
        assignments,
        device=device,
        requires_grad=True,
    )

    packed_token_ids = torch.tensor(
        [i % num_tokens for i in range(assignments)],
        device=device,
        dtype=torch.int32,
    )

    expert_out_cuda = expert_out_ref.detach().clone().requires_grad_(True)
    gate_cuda = gate_ref.detach().clone().requires_grad_(True)

    y_ref = torch.zeros(num_tokens, d_model, device=device, dtype=torch.float32)
    y_ref.index_add_(
        0,
        packed_token_ids.long(),
        expert_out_ref.float() * gate_ref[:, None],
    )

    y_cuda = combine_tokens_fn(expert_out_cuda, packed_token_ids, gate_cuda, num_tokens, d_model)

    torch.testing.assert_close(y_cuda, y_ref, **TOLERANCES[dtype])

    grad_out = torch.randn_like(y_ref)

    y_ref.backward(grad_out)
    y_cuda.backward(grad_out)

    torch.testing.assert_close(
        expert_out_cuda.grad,
        expert_out_ref.grad,
        **TOLERANCES[dtype],
    )

    torch.testing.assert_close(
        gate_cuda.grad,
        gate_ref.grad,
        **TOLERANCES[dtype],
    )
