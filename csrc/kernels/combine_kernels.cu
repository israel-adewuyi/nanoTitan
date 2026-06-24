#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>


template <typename T>
__global__ void combine_tokens_kernel_cu(
    const T* __restrict__ expert_outputs,            //[assignments, d_model]
    const int32_t* __restrict__ packed_tokenId,       //[assignments]
    const float* __restrict__ packed_topk_weights,       //[assignments]
    float* combined_residual_stream,   //[num_tokens, d_model]
    size_t d_model,
    int32_t num_assignments
){
    size_t assignment = blockIdx.x;
    size_t thread_idx = threadIdx.x;

    if (assignment >= num_assignments) return;

    __shared__ size_t token_idx;
    
    if(thread_idx == 0){
        token_idx = packed_tokenId[assignment];
    }
    
    __syncthreads();

    for(size_t h = thread_idx; h < d_model; h += blockDim.x){
        float weight = packed_topk_weights[assignment];
        float value = static_cast<float>(expert_outputs[assignment * d_model + h]);
        atomicAdd(&combined_residual_stream[token_idx * d_model + h], weight * value);
    }
}

torch::Tensor combine_tokens_kernel(
    torch::Tensor expert_outputs,            //[assignments, d_model]
    torch::Tensor packed_tokenId,       //[assignments]
    torch::Tensor packed_topk_weights,       //[assignments]
    size_t num_tokens,
    size_t hidden_dim
){
    TORCH_CHECK(expert_outputs.is_contiguous(), "Expert output tensor should be contiguous");
    TORCH_CHECK(packed_tokenId.is_contiguous(), "packed token ids tensor should be contiguous");
    TORCH_CHECK(packed_topk_weights.is_contiguous(), "packed topk weights tensor should be contiguous");

    int32_t num_assignments = expert_outputs.size(0);

    dim3 threads(256);
    dim3 blocks(num_assignments);

    torch::Tensor combined_buffer = torch::zeros(
        {static_cast<long>(num_tokens), static_cast<long>(hidden_dim)},
        expert_outputs.options().dtype(torch::kFloat32)
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        expert_outputs.scalar_type(),
        "combine_tokens_kernel",
        [&] {
            combine_tokens_kernel_cu<scalar_t><<<blocks, threads>>>(
                expert_outputs.data_ptr<scalar_t>(),
                packed_tokenId.data_ptr<int32_t>(),
                packed_topk_weights.data_ptr<float>(),
                combined_buffer.data_ptr<float>(),
                hidden_dim,
                num_assignments
            );
        }
    );

    return combined_buffer;
}