import pytest
import torch

from src.model.moe_ops import combine_tokens_fn, pack_tokens_fn

TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float16: {"rtol": 1e-2, "atol": 1e-2},
    torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens,top_k,d_model", [(3, 2, 8), (5, 3, 7)])
def test_pack_then_combine_backward_matches_reference(dtype, num_tokens, top_k, d_model):
    torch.manual_seed(0)
    device = "cuda"

    x_ref = torch.randn(num_tokens, d_model, dtype=dtype, device=device, requires_grad=True)
    x_cuda = x_ref.detach().clone().requires_grad_(True)
    weights_ref = torch.rand(num_tokens, top_k, device=device, requires_grad=True)
    weights_cuda = weights_ref.detach().clone().requires_grad_(True)
    topk_experts = (
        (torch.arange(num_tokens * top_k, device=device) % 3)
        .reshape(num_tokens, top_k)
        .to(torch.int32)
    )
    counts = torch.bincount(topk_experts.reshape(-1).long(), minlength=3)
    expert_offset = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(torch.int32)

    packed_x, packed_token_id, _, packed_weights = pack_tokens_fn(
        x_cuda, weights_cuda, topk_experts, expert_offset.clone()
    )
    y_cuda = combine_tokens_fn(packed_x, packed_token_id, packed_weights, num_tokens, d_model)

    y_ref = torch.zeros(num_tokens, d_model, dtype=torch.float32, device=device)
    for k in range(top_k):
        y_ref += x_ref.float() * weights_ref[:, k, None]

    torch.testing.assert_close(y_cuda, y_ref, **TOLERANCES[dtype])

    grad_out = torch.randn_like(y_ref)
    y_ref.backward(grad_out)
    y_cuda.backward(grad_out)

    torch.testing.assert_close(x_cuda.grad, x_ref.grad, **TOLERANCES[dtype])
    torch.testing.assert_close(weights_cuda.grad, weights_ref.grad, **TOLERANCES[dtype])
