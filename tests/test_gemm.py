import einops
import pytest
import random_ext
import torch


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_gemm_kernel_small(dtype):
    M, N, K = 10, 20, 30

    A = torch.randn((M, K), dtype=dtype, device="cuda")
    B = torch.randn((K, N), dtype=dtype, device=A.device)

    C = random_ext.naive_gemm_kernel(A, B)

    torch.cuda.synchronize()

    expected = einops.einsum(A, B, "M K, K N -> M N")

    torch.testing.assert_close(C, expected, rtol=1e-5, atol=1e-6)
