import torch

import random_ext


class CombineTokensFN(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, packed_expert_outputs, packed_tokenId, packed_topk_weights, num_tokens, hidden_dim
    ) -> torch.Tensor:
        pool = random_ext.combine_tokens_kernel(
            packed_expert_outputs, packed_tokenId, packed_topk_weights, num_tokens, hidden_dim
        )

        ctx.save_for_backward(packed_expert_outputs, packed_tokenId, packed_topk_weights)
        ctx.hidden_dim = hidden_dim
        ctx.num_tokens = num_tokens

        return pool

    @staticmethod
    def backward(ctx, d_resid_stream) -> tuple:
        packed_expert_outputs, packed_tokenId, packed_topk_weights = ctx.saved_tensors

        expert_output_grad, packed_topk_weights_grad = random_ext.combine_kernel_backward(
            packed_expert_outputs,
            packed_tokenId,
            packed_topk_weights,
            d_resid_stream.contiguous(),
            ctx.hidden_dim,
            ctx.num_tokens,
        )

        return expert_output_grad, None, packed_topk_weights_grad, None


def combine_tokens(
    packed_expert_outputs, packed_tokenId, packed_topk_weights, num_tokens, hidden_dim
):
    return CombineTokensFn.apply(
        packed_expert_outputs, packed_tokenId, packed_topk_weights, num_tokens, hidden_dim
    )
