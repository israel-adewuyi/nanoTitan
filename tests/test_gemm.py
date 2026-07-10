import einops
import pytest
import torch

import random_ext

TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-4},
    torch.float16: {"rtol": 1e-2, "atol": 1e-2},
    torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("implementation", [None, "tiled", "naive"])
@pytest.mark.parametrize("shape", [(10, 20, 30), (33, 45, 65), (33, 2, 65)])
def test_gemm_kernel_small(dtype, implementation, shape):
    M, N, K = shape

    A = torch.randn((M, K), dtype=dtype, device="cuda")
    B = torch.randn((K, N), dtype=dtype, device=A.device)

    if implementation is None:
        C = random_ext.gemm_kernel(A, B)
    else:
        C = random_ext.gemm_kernel(A, B, implementation)

    torch.cuda.synchronize()

    expected = einops.einsum(A, B, "M K, K N -> M N")

    torch.testing.assert_close(C, expected, **TOLERANCES[dtype])


def test_naive_gemm_kernel_wrapper():
    A = torch.randn((10, 30), device="cuda")
    B = torch.randn((30, 20), device=A.device)

    C = random_ext.naive_gemm_kernel(A, B)

    torch.testing.assert_close(C, A @ B, rtol=1e-5, atol=1e-6)

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_tiled_gemm_equals_naive_gemm_kernel(dtype):
    A = torch.randn((10, 30), dtype=dtype, device="cuda")
    B = torch.randn((30, 20), dtype=dtype, device=A.device)

    naive_C = random_ext.naive_gemm_kernel(A, B)
    tiled_C = random_ext.tiled_gemm_kernel(A, B)

    torch.testing.assert_close(naive_C, tiled_C, rtol=1e-5, atol=1e-6)


def test_tiled_gemm_kernel_rejects_unknown_implementation():
    A = torch.randn((10, 30), device="cuda")
    B = torch.randn((30, 20), device=A.device)

    with pytest.raises(RuntimeError, match="implementation must be either 'tiled' or 'naive'"):
        random_ext.gemm_kernel(A, B, "wat")
