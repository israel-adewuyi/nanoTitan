#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cstdint>
#include <cuda_runtime.h>


template <typename T>
__global__ void pack_kernel_backward_cu(
    const T* __restrict__ packed_X_grad,
    const float* __restrict__ packed_topk_weights_grad,
    T* __restrict__ X_grad,
    float* __restrict__ topk_weights_grad,
    const int32_t* __restrict__ packed_tokenId,
    const int32_t* __restrict__ packed_expert,
    const int32_t* __restrict__ topk_experts,
    int64_t topK,
    int64_t hidden_dim
){
    int64_t assignment = blockIdx.x;
    int64_t h = threadIdx.x;

    int32_t token_id = packed_tokenId[assignment];
    int32_t expert = packed_expert[assignment];

    for(int64_t hh = h; hh < hidden_dim; hh += blockDim.x){
        gpuAtomicAddNoReturn(
            &X_grad[token_id * hidden_dim + hh],
            packed_X_grad[assignment * hidden_dim + hh]
        );
    }

    if(h == 0){
        for(int64_t expert_idx = 0; expert_idx < topK; ++expert_idx){
            int64_t idx = token_id * topK + expert_idx;
            if(topk_experts[idx] == expert){
                atomicAdd(&topk_weights_grad[idx], packed_topk_weights_grad[assignment]);
                break;
            }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor> pack_kernel_backward(
    torch::Tensor packed_X_grad,
    torch::Tensor packed_topk_weights_grad,
    torch::Tensor packed_tokenId,
    torch::Tensor packed_expert,
    torch::Tensor topk_experts
){
    TORCH_CHECK(packed_X_grad.is_contiguous(), "packed_X_grad must be contiguous");
    TORCH_CHECK(packed_topk_weights_grad.is_contiguous(), "packed_topk_weights_grad must be contiguous");
    TORCH_CHECK(packed_tokenId.is_contiguous(), "packed_tokenId must be contiguous");
    TORCH_CHECK(packed_expert.is_contiguous(), "packed_expert must be contiguous");
    TORCH_CHECK(topk_experts.is_contiguous(), "topk_experts must be contiguous");
    TORCH_CHECK(packed_topk_weights_grad.scalar_type() == torch::kFloat32, "packed_topk_weights_grad must be float32");

    int64_t total_assignments = packed_X_grad.size(0);
    int64_t hidden_dim = packed_X_grad.size(1);
    int64_t num_tokens = topk_experts.size(0);
    int64_t topK = topk_experts.size(1);

    c10::cuda::CUDAGuard guard(packed_X_grad.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    torch::Tensor X_grad = torch::zeros(
        {num_tokens, hidden_dim},
        packed_X_grad.options()
    );
    torch::Tensor topk_weights_grad = torch::zeros(
        {num_tokens, topK},
        packed_topk_weights_grad.options().dtype(torch::kFloat32)
    );

    dim3 threads(256);
    dim3 blocks(total_assignments);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        packed_X_grad.scalar_type(),
        "pack_kernel_backward",
        [&] {
            pack_kernel_backward_cu<scalar_t><<<blocks, threads, 0, stream>>>(
                packed_X_grad.data_ptr<scalar_t>(),
                packed_topk_weights_grad.data_ptr<float>(),
                X_grad.data_ptr<scalar_t>(),
                topk_weights_grad.data_ptr<float>(),
                packed_tokenId.data_ptr<int32_t>(),
                packed_expert.data_ptr<int32_t>(),
                topk_experts.data_ptr<int32_t>(),
                topK,
                hidden_dim
            );
        }
    );

    return {X_grad, topk_weights_grad};
}
