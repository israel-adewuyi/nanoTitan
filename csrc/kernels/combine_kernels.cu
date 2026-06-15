#include <cuda_runtime.h>


template <typename T>
__global__ void combine_tokens(
    T* expert_outputs,
    size_t* packed_tokenId,
    T* packed_topk_weights,
    size_t d_model,
    T* combined_residual_stream
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

/*
expert outputs [num_assignments, d_model]



*/