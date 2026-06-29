import torch
import pytest

from src.model.moe_ops import pack_tokens


def test_pack_backward_matches_reference():
    torch.manual_seed(0)
    device = "cuda"

    num_tokens = 3
    assignments = 6
    d_model = 8

    packed_token_ids = torch.tensor(
        [0, 0, 1, 1, 2, 2],
        device=device,
        dtype=torch.int32,
    )

    x_ref = torch.randn(num_tokens, d_model, device=device, requires_grad=True)
    x_cuda = x_ref.detach().clone().requires_grad_(True)

    # Just for correctness
    topk_experts = torch.zeros(
        (num_tokens, assignments // num_tokens), dtype=torch.int32, device=device
    )
    topk_weights = torch.zeros(
        (num_tokens, assignments // num_tokens), dtype=torch.float32, device=device
    )
    expert_offset = torch.tensor(
        [0, 2, 4, 6],
        device=device,
        dtype=torch.int32,
    )

    packed_cuda, packed_tokenId, packed_expert, packed_topk_weights = pack_tokens(
        x_cuda, topk_weights, topk_experts, expert_offset
    )

    packed_ref = x_ref[packed_tokenId.long()]

    torch.testing.assert_close(
        packed_cuda,
        packed_ref,
        rtol=1e-4,
        atol=1e-4,
    )

    grad_packed = torch.randn_like(packed_ref)

    packed_ref.backward(grad_packed)
    packed_cuda.backward(grad_packed)

    torch.testing.assert_close(
        x_cuda.grad,
        x_ref.grad,
        rtol=1e-4,
        atol=1e-4,
    )
