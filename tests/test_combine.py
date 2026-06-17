import torch
import random_ext


def test_combine_top1():
    """
    topK = 1
    num_expert = 3
    num_tokens = 3
    num_assignments = 3
    """

    expert_outputs = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], device="cuda")

    packed_tokenId = torch.tensor([1, 0, 2], device=expert_outputs.device, dtype=torch.int32)
    packed_topk_weights = torch.tensor([1.0, 1.0, 1.0], device=expert_outputs.device)
    resid_stream = torch.zeros_like(expert_outputs)

    random_ext.combine_tokens_kernel(
        expert_outputs, packed_tokenId, packed_topk_weights, 2, resid_stream
    )

    torch.cuda.synchronize()

    for token_idx in range(3):
        torch.testing.assert_close(
            expert_outputs[token_idx], resid_stream[packed_tokenId[token_idx]]
        )

    expected = torch.zeros_like(resid_stream)
    expected[1] = expert_outputs[0]
    expected[0] = expert_outputs[1]
    expected[2] = expert_outputs[2]

    torch.testing.assert_close(resid_stream, expected)


def test_combine_top2():
    """
    topK = 2
    num_expert = 3
    num_tokens = 3
    num_assignments = 6
    """

    expert_outputs = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        device="cuda",
    )

    packed_tokenId = torch.tensor(
        [1, 2, 0, 2, 0, 1], device=expert_outputs.device, dtype=torch.int32
    )
    packed_topk_weights = torch.tensor([0.3, 0.4, 0.4, 0.5, 0.3, 0.6], device=expert_outputs.device)
    resid_stream = torch.zeros((3, 2), device=expert_outputs.device, dtype=torch.float32)

    random_ext.combine_tokens_kernel(
        expert_outputs, packed_tokenId, packed_topk_weights, 2, resid_stream
    )

    torch.cuda.synchronize()

    expected = torch.zeros((3, 2), device=expert_outputs.device, dtype=torch.float32)
    expected[0] = 0.4 * expert_outputs[2] + 0.3 * expert_outputs[4]
    expected[1] = 0.3 * expert_outputs[0] + 0.6 * expert_outputs[5]
    expected[2] = 0.4 * expert_outputs[1] + 0.5 * expert_outputs[3]

    torch.testing.assert_close(resid_stream, expected, rtol=1e-5, atol=1e-6)


def test_combine_top2_bf16_expert_output():
    """
    topK = 2
    num_expert = 3
    num_tokens = 3
    num_assignments = 6
    """

    expert_outputs = torch.tensor(
        [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6],
        ],
        device="cuda",
        dtype=torch.bfloat16,
    )

    packed_tokenId = torch.tensor(
        [1, 2, 0, 2, 0, 1], device=expert_outputs.device, dtype=torch.int32
    )
    packed_topk_weights = torch.tensor([0.3, 0.4, 0.4, 0.5, 0.3, 0.6], device=expert_outputs.device)
    resid_stream = torch.zeros((3, 2), device=expert_outputs.device, dtype=torch.float32)

    random_ext.combine_tokens_kernel(
        expert_outputs, packed_tokenId, packed_topk_weights, 2, resid_stream
    )

    torch.cuda.synchronize()

    expected = torch.zeros((3, 2), device=expert_outputs.device, dtype=torch.float32)
    expected[0] = 0.4 * expert_outputs[2] + 0.3 * expert_outputs[4]
    expected[1] = 0.3 * expert_outputs[0] + 0.6 * expert_outputs[5]
    expected[2] = 0.4 * expert_outputs[1] + 0.5 * expert_outputs[3]

    torch.testing.assert_close(resid_stream, expected, rtol=1e-5, atol=1e-6)
