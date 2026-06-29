import torch

from src.model.moe_ops import combine_tokens


def test_combine_backward_matches_reference():
    torch.manual_seed(0)
    device = "cuda"

    num_tokens = 3
    assignments = 6
    d_model = 8

    expert_out_ref = torch.randn(
        assignments,
        d_model,
        device=device,
        requires_grad=True,
    )
    gate_ref = torch.randn(
        assignments,
        device=device,
        requires_grad=True,
    )

    packed_token_ids = torch.tensor(
        [0, 0, 1, 1, 2, 2],
        device=device,
        dtype=torch.int32,
    )

    expert_out_cuda = expert_out_ref.detach().clone().requires_grad_(True)
    gate_cuda = gate_ref.detach().clone().requires_grad_(True)

    # Torch reference
    y_ref = torch.zeros(num_tokens, d_model, device=device)
    y_ref.index_add_(
        0,
        packed_token_ids.long(),
        expert_out_ref * gate_ref[:, None],
    )

    # CUDA custom op
    y_cuda = combine_tokens(expert_out_cuda, packed_token_ids, gate_cuda, num_tokens, d_model).to(
        dtype=expert_out_ref.dtype
    )

    torch.testing.assert_close(y_cuda, y_ref, rtol=1e-4, atol=1e-4)

    grad_out = torch.randn_like(y_ref)

    y_ref.backward(grad_out)
    y_cuda.backward(grad_out)

    torch.testing.assert_close(
        expert_out_cuda.grad,
        expert_out_ref.grad,
        rtol=1e-4,
        atol=1e-4,
    )

    torch.testing.assert_close(
        gate_cuda.grad,
        gate_ref.grad,
        rtol=1e-4,
        atol=1e-4,
    )
