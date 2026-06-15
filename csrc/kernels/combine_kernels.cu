#include <cuda_runtime.h>
#include <torch/extension.h>


template <typename T>
__global__ void combine_tokens_kernel_cu(
    T* expert_outputs,            //[assignments, d_model]
    int32_t* packed_tokenId,       //[assignments]
    T* packed_topk_weights,       //[assignments]
    size_t d_model,
    float* combined_residual_stream   //[num_tokens, d_model]
){
    size_t assignment = blockIdx.x;
    size_t thread_idx = threadIdx.x;

    __shared__ size_t token_idx;
    
    if(thread_idx == 0){
        token_idx = packed_tokenId[assignment];
    }
    
    __syncthreads();

    for(size_t h = thread_idx; h < d_model; h += blockDim.x){
        T weight = packed_topk_weights[assignment];
        atomicAdd(&combined_residual_stream[token_idx * d_model + h], weight * expert_outputs[assignment * d_model + h]);
        // combined_residual_stream[token_idx * d_model + h] = packed_topk_weights[assignment] * expert_outputs[assignment * d_model + h];
    }
}

void combine_tokens_kernel(
    torch::Tensor expert_outputs,            //[assignments, d_model]
    torch::Tensor packed_tokenId,       //[assignments]
    torch::Tensor packed_topk_weights,       //[assignments]
    size_t d_model,
    torch::Tensor combined_residual_stream   //[num_tokens, d_model]
){
    TORCH_CHECK(expert_outputs.is_contiguous(), "Expert output tensor should be contiguous");
    TORCH_CHECK(packed_tokenId.is_contiguous(), "packed token ids tensor should be contiguous");
    TORCH_CHECK(packed_topk_weights.is_contiguous(), "packed topk weights tensor should be contiguous");
    TORCH_CHECK(combined_residual_stream.is_contiguous(), "Resid stream tensor should be contiguous");


    int32_t num_assignments = expert_outputs.shape[0];

    dim3 threads(256);
    dim3 blocks(num_assignments);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        expert_outputs.scalar_type(),
        "pack_tokens_kernel",
        [&] {
            combine_tokens_kernel_cu<scalar_t><<<blocks, threads>>>(
                expert_outputs.data_ptr<scalar_t>(),
                packed_tokenId.data_ptr<int32_t>(),
                packed_topk_weights.data_ptr<scalar_t>(),
                d_model,
                combined_residual_stream.data_ptr<float>()
            );
        }
    );
}