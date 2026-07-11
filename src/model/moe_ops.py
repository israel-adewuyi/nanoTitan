from typing import Any

import torch

import random_ext


def combine_tokens_fn(
    packed_expert_outputs, packed_tokenId, packed_topk_weights, num_tokens, hidden_dim
):
    return CombineTokensFN.apply(
        packed_expert_outputs, packed_tokenId, packed_topk_weights, num_tokens, hidden_dim
    )


def pack_tokens_fn(X, topk_weights, topk_experts, expert_offset_cpy):
    return PackTokensFN.apply(X, topk_weights, topk_experts, expert_offset_cpy)


class PackTokensFN(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X,
        topk_weights,
        topk_experts,
        expert_offset_cpy,
    ) -> tuple:
        packed_X, packed_tokenId, packed_expert, packed_topk_weights = (
            random_ext.pack_tokens_kernel(X, topk_weights, topk_experts, expert_offset_cpy)
        )

        ctx.save_for_backward(packed_tokenId, packed_expert, topk_experts)

        return packed_X, packed_tokenId, packed_expert, packed_topk_weights

    @staticmethod
    def backward(
        ctx, packed_X_grad, packed_tokenId_grad, packed_expert_grad, packed_topk_weights_grad
    ) -> Any:
        packed_tokenId, packed_expert, topk_experts = ctx.saved_tensors

        X_grad, topk_weights_grad = random_ext.pack_kernel_backward(
            packed_X_grad.contiguous(),
            packed_topk_weights_grad.contiguous(),
            packed_tokenId,
            packed_expert,
            topk_experts,
        )

        return X_grad, topk_weights_grad, None, None


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

        return pool

    @staticmethod
    def backward(ctx, d_resid_stream) -> tuple:
        packed_expert_outputs, packed_tokenId, packed_topk_weights = ctx.saved_tensors

        expert_output_grad, packed_topk_weights_grad = random_ext.combine_kernel_backward(
            packed_expert_outputs,
            packed_tokenId,
            packed_topk_weights,
            d_resid_stream.to(torch.float32).contiguous(),
            ctx.hidden_dim,
        )

        return expert_output_grad, None, packed_topk_weights_grad, None, None
