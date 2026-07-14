import pytest
import torch

import random_ext


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(4, 2, 4, 4), (128, 4, 16, 32), (8192, 3, 1, 8192)])
def test_grouped_gemm_kernel_small_smoke(dtype, shape):
    assignments, num_experts, d_model, d_ffn_in = shape

    X = torch.randn((assignments, d_model), dtype=dtype, device="cuda")
    weights = torch.randn((num_experts, d_model, d_ffn_in), dtype=X.dtype, device=X.device)
    expert_offset = torch.tensor(
        [i * assignments // num_experts for i in range(num_experts + 1)],
        dtype=torch.int32,
        device=X.device,
    )

    out = random_ext.grouped_gemm_kernel(
        X,
        expert_offset,
        weights,
    )

    torch.cuda.synchronize()
    assert isinstance(out, torch.Tensor)

    expected = torch.empty_like(out)
    for expert_id in range(num_experts):
        start = expert_offset[expert_id].item()
        end = expert_offset[expert_id + 1].item()
        expected[start:end] = X[start:end] @ weights[expert_id]

    torch.testing.assert_close(out, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_grouped_gemm_backward_matches_autograd(dtype):
    counts = [3, 1, 4]
    offsets = [0, 3, 4, 8]
    assignments, d_model, d_ffn_in = sum(counts), 7, 11
    expert_offset = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    X = torch.randn(assignments, d_model, dtype=dtype, device="cuda", requires_grad=True)
    weights = torch.randn(
        len(counts),
        d_model,
        d_ffn_in,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    d_out = torch.randn(assignments, d_ffn_in, dtype=dtype, device="cuda")

    out = torch.cat([X[offsets[e] : offsets[e + 1]] @ weights[e] for e in range(len(counts))])
    out.backward(d_out)

    dX = random_ext.bwd_grouped_gemm_up_proj_dX_kernel(weights.detach(), expert_offset, d_out)
    dW = random_ext.bwd_grouped_gemm_up_proj_dW_kernel(X.detach(), expert_offset, d_out)

    torch.testing.assert_close(dX, X.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dW, weights.grad, rtol=2e-2, atol=2e-2)
