import torch

import random_ext


def test_grouped_gemm_kernel_small_smoke():
    assignments, num_experts, d_model, d_ffn_in = 4, 2, 4, 4

    X = torch.randn((assignments, d_model), dtype=torch.float32, device="cuda")
    weights = torch.randn((num_experts, d_model, d_ffn_in), dtype=X.dtype, device=X.device)
    expert_offset = torch.tensor([0, 2, 4], dtype=torch.int32, device=X.device)

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

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)
