import pytest
import torch

pytestmark = pytest.mark.cuda

if not torch.cuda.is_available():
    pytest.skip("CUDA unavailable", allow_module_level=True)

moe_ops = pytest.importorskip("src.model.moe_ops")

pack_tokens_fn = moe_ops.pack_tokens_fn

TOLERANCES = {
    torch.float32: {"rtol": 1e-5, "atol": 1e-6},
    torch.float16: {"rtol": 1e-2, "atol": 1e-2},
    torch.bfloat16: {"rtol": 2e-2, "atol": 2e-2},
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("num_tokens,top_k,d_model", [(3, 2, 8), (5, 3, 7)])
def test_pack_backward_matches_reference(dtype, num_tokens, top_k, d_model):
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

    packed_cuda, packed_token_id, packed_expert, packed_weight_cuda = pack_tokens_fn(
        x_cuda, weights_cuda, topk_experts, expert_offset.clone()
    )
    token_id = packed_token_id.long()
    topk_slot = (topk_experts[token_id] == packed_expert[:, None]).long().argmax(dim=1)
    packed_ref = x_ref[token_id]
    packed_weight_ref = weights_ref[token_id, topk_slot]

    torch.testing.assert_close(packed_cuda, packed_ref, **TOLERANCES[dtype])
    torch.testing.assert_close(packed_weight_cuda, packed_weight_ref)

    grad_packed = torch.randn_like(packed_ref)
    grad_weight = torch.randn_like(packed_weight_ref)
    (packed_ref * grad_packed).sum().add((packed_weight_ref * grad_weight).sum()).backward()
    (packed_cuda * grad_packed).sum().add((packed_weight_cuda * grad_weight).sum()).backward()

    torch.testing.assert_close(x_cuda.grad, x_ref.grad, **TOLERANCES[dtype])
    torch.testing.assert_close(weights_cuda.grad, weights_ref.grad, rtol=1e-5, atol=1e-6)
