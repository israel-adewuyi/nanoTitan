from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

random_ext = pytest.importorskip("random_ext")
feed_fwd = pytest.importorskip("src.model.feed_fwd")
moe_ops = pytest.importorskip("src.model.moe_ops")

ExpertFFN = feed_fwd.ExpertFFN
grouped_gemm_fn = moe_ops.grouped_gemm_fn

SUPPORTED_DTYPES = [
    pytest.param(torch.float32, id="float32"),
    pytest.param(torch.float16, id="float16"),
    pytest.param(torch.bfloat16, id="bfloat16"),
]

# ExpertFFN stores its projection matrices as:
#   W_gate/W_val: [num_experts, d_model, ffn_in]
#   W_out:        [num_experts, ffn_in, d_model]
# These dimensions match the smaller CUDA model config in configs/dist.toml.
MOE_PROJECTION_SHAPES = [
    pytest.param(128, 512, id="up-projection"),
    pytest.param(512, 128, id="down-projection-w-out"),
]

TOLERANCES = {
    torch.float32: {"rtol": 1e-4, "atol": 1e-4},
    torch.float16: {"rtol": 2e-2, "atol": 2e-2},
    torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
}


def torch_grouped_gemm(X, weights, offsets):
    return torch.cat([X[offsets[e] : offsets[e + 1]] @ weights[e] for e in range(len(offsets) - 1)])


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
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


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(("input_features", "output_features"), MOE_PROJECTION_SHAPES)
def test_grouped_gemm_forward_matches_moe_projection(dtype, input_features, output_features):
    counts = [3, 1, 4]
    offsets = [0, 3, 4, 8]
    assignments = sum(counts)
    expert_offset = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    X = torch.randn(assignments, input_features, dtype=dtype, device="cuda")
    weights = torch.randn(len(counts), input_features, output_features, dtype=dtype, device="cuda")

    actual = random_ext.grouped_gemm_kernel(X, expert_offset, weights)
    expected = torch_grouped_gemm(X, weights, offsets)

    assert actual.shape == (assignments, output_features)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(("input_features", "output_features"), MOE_PROJECTION_SHAPES)
def test_grouped_gemm_backward_matches_moe_projection_autograd(
    dtype, input_features, output_features
):
    counts = [3, 1, 4]
    offsets = [0, 3, 4, 8]
    assignments = sum(counts)
    expert_offset = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    X = torch.randn(assignments, input_features, dtype=dtype, device="cuda", requires_grad=True)
    weights = torch.randn(
        len(counts),
        input_features,
        output_features,
        dtype=dtype,
        device="cuda",
        requires_grad=True,
    )
    d_out = torch.randn(assignments, output_features, dtype=dtype, device="cuda")

    out = torch_grouped_gemm(X, weights, offsets)
    out.backward(d_out)

    dX = random_ext.bwd_grouped_gemm_dX_kernel(weights.detach(), expert_offset, d_out)
    dW = random_ext.bwd_grouped_gemm_dW_kernel(X.detach(), expert_offset, d_out)

    assert dX.shape == X.shape
    assert dW.shape == weights.shape
    torch.testing.assert_close(dX, X.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dW, weights.grad, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize(("input_features", "output_features"), MOE_PROJECTION_SHAPES)
def test_grouped_gemm_autograd_wrapper_matches_torch(dtype, input_features, output_features):
    offsets = [0, 3, 4, 8]
    expert_offset = torch.tensor(offsets, dtype=torch.int32, device="cuda")

    X_ref = torch.randn(8, input_features, dtype=dtype, device="cuda", requires_grad=True)
    X_cuda = X_ref.detach().clone().requires_grad_(True)
    weights_ref = torch.randn(
        3, input_features, output_features, dtype=dtype, device="cuda", requires_grad=True
    )
    weights_cuda = weights_ref.detach().clone().requires_grad_(True)

    expected = torch_grouped_gemm(X_ref, weights_ref, offsets)
    actual = grouped_gemm_fn(X_cuda, weights_cuda, expert_offset)

    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])

    d_out = torch.randn(output_features, 8, dtype=dtype, device="cuda").T
    expected.backward(d_out)
    actual.backward(d_out)

    torch.testing.assert_close(X_cuda.grad, X_ref.grad, **TOLERANCES[dtype])
    torch.testing.assert_close(weights_cuda.grad, weights_ref.grad, **TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_expert_ffn_cuda_matches_torch_forward_and_backward(dtype):
    torch.manual_seed(0)
    cfg_torch = SimpleNamespace(
        num_experts=3,
        d_model=8,
        ffn_in=16,
        dtype=dtype,
        moe_backend="torch",
    )
    cfg_cuda = SimpleNamespace(**vars(cfg_torch))
    cfg_cuda.moe_backend = "cuda"

    expert_ref = ExpertFFN(cfg_torch).to("cuda")
    expert_cuda = ExpertFFN(cfg_cuda).to("cuda")
    expert_cuda.load_state_dict(expert_ref.state_dict())

    offsets = torch.tensor([0, 3, 4, 8], dtype=torch.int32, device="cuda")
    X_ref = torch.randn(8, cfg_torch.d_model, dtype=dtype, device="cuda", requires_grad=True)
    X_cuda = X_ref.detach().clone().requires_grad_(True)

    expected = expert_ref(X_ref, offsets)
    actual = expert_cuda(X_cuda, offsets)

    torch.testing.assert_close(actual, expected, **TOLERANCES[dtype])

    d_out = torch.randn_like(expected)
    expected.backward(d_out)
    actual.backward(d_out)

    torch.testing.assert_close(X_cuda.grad, X_ref.grad, **TOLERANCES[dtype])
    for name in ("W_gate", "W_val", "W_out"):
        torch.testing.assert_close(
            getattr(expert_cuda, name).grad,
            getattr(expert_ref, name).grad,
            **TOLERANCES[dtype],
        )
