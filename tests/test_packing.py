import pytest
import torch

import random_ext

# Will extend to include pack and combine when combine is fully implemented.


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pack_tokens_correctness(dtype):
    hidden_dim = 2

    X = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.8, 0.8],
        ],
        device="cuda",
        dtype=dtype,
    )  # TODO: The fn should read the cuda device I want to use.

    topk_experts = torch.tensor(
        [
            [0, 1],
            [0, 2],
            [1, 0],
            [0, 2],
        ],
        device="cuda",
        dtype=torch.int32,
    )

    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.6, 0.4], [0.9, 0.1], [0.1, 0.9]],
        device="cuda",
        dtype=torch.float32,
    )

    topK = 2
    total_assignments = X.shape[0] * topK

    expert_offset_cpy = torch.tensor(
        ([0, 4, 6, 8]), device="cuda", dtype=torch.int32
    )  # TODO: I should call count_expert kernel and do the prefix sum for this. Leave for now.

    packed_X, packed_tokenId, packed_expert, packed_topk_weights = random_ext.pack_tokens_kernel(
        X,
        topk_weights,
        topk_experts,
        expert_offset_cpy,
    )

    torch.cuda.synchronize()

    assert packed_X.shape == (8, 2)
    assert packed_tokenId.shape == (8,)
    assert packed_expert.shape == (8,)
    assert packed_topk_weights.shape == (8,)

    # Is the correct token stored in each packed row?
    for idx in range(total_assignments):
        token_id = packed_tokenId[idx]
        torch.testing.assert_close(packed_X[idx], X[token_id])

    # Did packing follow the expert offset?
    expert_offsets = torch.tensor([0, 4, 6, 8], device="cuda", dtype=torch.int32)
    for e in range(3):
        start = expert_offsets[e].item()
        end = expert_offsets[e + 1].item()

        assert torch.all(packed_expert[start:end] == e)
