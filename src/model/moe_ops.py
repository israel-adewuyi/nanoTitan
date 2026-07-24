from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from src.model.cuda_extension import get_cuda_extension


def combine_tokens_fn(
    packed_expert_outputs, packed_tokenId, packed_topk_weights, num_tokens, hidden_dim
):
    return CombineTokensFN.apply(
        packed_expert_outputs, packed_tokenId, packed_topk_weights, num_tokens, hidden_dim
    )


def pack_tokens_fn(X, topk_weights, topk_experts, expert_offset_cpy):
    return PackTokensFN.apply(X, topk_weights, topk_experts, expert_offset_cpy)


def grouped_gemm_fn(X, weight, expert_offset):
    return GroupedGEMM_FN.apply(X, expert_offset, weight)


def torch_backend_all_to_all(
    x: torch.Tensor,
    input_splits,
    output_splits,
    group: ProcessGroup,
) -> torch.Tensor:
    return TorchBackendAll2ALL.apply(
        x,
        input_splits,
        output_splits,
        group,
    )


class PackTokensFN(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X,
        topk_weights,
        topk_experts,
        expert_offset_cpy,
    ) -> tuple:
        nanotitan_cuda = get_cuda_extension()
        packed_X, packed_tokenId, packed_expert, packed_topk_weights = (
            nanotitan_cuda.pack_tokens_kernel(X, topk_weights, topk_experts, expert_offset_cpy)
        )

        ctx.save_for_backward(packed_tokenId, packed_expert, topk_experts)

        return packed_X, packed_tokenId, packed_expert, packed_topk_weights

    @staticmethod
    def backward(
        ctx, packed_X_grad, packed_tokenId_grad, packed_expert_grad, packed_topk_weights_grad
    ) -> Any:
        nanotitan_cuda = get_cuda_extension()
        packed_tokenId, packed_expert, topk_experts = ctx.saved_tensors

        X_grad, topk_weights_grad = nanotitan_cuda.pack_kernel_backward(
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
        nanotitan_cuda = get_cuda_extension()
        pool = nanotitan_cuda.combine_tokens_kernel(
            packed_expert_outputs, packed_tokenId, packed_topk_weights, num_tokens, hidden_dim
        )

        ctx.save_for_backward(packed_expert_outputs, packed_tokenId, packed_topk_weights)
        ctx.hidden_dim = hidden_dim

        return pool

    @staticmethod
    def backward(ctx, d_resid_stream) -> tuple:
        nanotitan_cuda = get_cuda_extension()
        packed_expert_outputs, packed_tokenId, packed_topk_weights = ctx.saved_tensors

        expert_output_grad, packed_topk_weights_grad = nanotitan_cuda.combine_kernel_backward(
            packed_expert_outputs,
            packed_tokenId,
            packed_topk_weights,
            d_resid_stream.to(torch.float32).contiguous(),
            ctx.hidden_dim,
        )

        return expert_output_grad, None, packed_topk_weights_grad, None, None


class GroupedGEMM_FN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, expert_offset, weight) -> torch.Tensor:
        nanotitan_cuda = get_cuda_extension()
        ctx.save_for_backward(X, expert_offset, weight)

        return nanotitan_cuda.grouped_gemm_kernel(X, expert_offset, weight)

    @staticmethod
    def backward(ctx, out_grad) -> tuple:
        nanotitan_cuda = get_cuda_extension()
        X, expert_offset, weight = ctx.saved_tensors
        out_grad = out_grad.contiguous()

        dX = nanotitan_cuda.bwd_grouped_gemm_dX_kernel(weight, expert_offset, out_grad)

        dW = nanotitan_cuda.bwd_grouped_gemm_dW_kernel(X, expert_offset, out_grad)

        return dX, None, dW


class TorchBackendAll2ALL(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x: torch.Tensor, input_splits, output_splits, group: ProcessGroup
    ) -> torch.Tensor:
        ctx.input_splits = input_splits
        ctx.output_splits = output_splits
        ctx.group = group

        output = x.new_empty((sum(output_splits), *x.shape[1:]))

        dist.all_to_all_single(
            output,
            x.contiguous(),
            input_split_sizes=input_splits,
            output_split_sizes=output_splits,
            group=group,
        )

        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_input = grad_output.new_empty((sum(ctx.input_splits), *grad_output.shape[1:]))

        dist.all_to_all_single(
            grad_input,
            grad_output.contiguous(),
            input_split_sizes=ctx.output_splits,
            output_split_sizes=ctx.input_splits,
            group=ctx.group,
        )
        return grad_input, None, None, None
