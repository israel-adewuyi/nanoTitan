#include <cuda_runtime.h>


template <typename T>
__global__ void pack_tokens_kernel(
    T* X,                                // [num_tokens, d_model]
    T* topk_weights,                     // [num_tokens, top_K]
    size_t* topk_experts,                // [num_tokens, top_K]
    size_t topK,
    size_t total_assignments,
    size_t* expert_offset_cpy, 
    T* packed_X, 
    size_t* packed_tokenId, 
    size_t* packed_expert, 
    T* packed_weights,
    size_t d_model
){
    // Map logical threads to num_tokens * topK
    size_t assignment = blockIdx.x;
    size_t thread_idx = threadIdx.x; 

    if(assignment >= total_assignments)return;

    // Get the token id for this thread and the e-th expert token_id wants to route to
    size_t token_id = assignment / topK;
    size_t expert_idx = assignment % topK;

    // Get the expert
    size_t expert = topk_experts[token_id * topK + expert_idx];

    __shared__ size_t slot;

    if(thread_idx == 0){
        slot = atomicAdd(&expert_offset_cpy[expert], 1);
        packed_tokenId[slot] = token_id;
        packed_expert[slot] = expert;
        packed_weights[slot] = topk_weights[token_id * topK + expert_idx];
    }

    __syncthreads();

    for(size_t h = thread_idx; h < d_model; h += blockDim.x){
        packed_X[slot * d_model + h] = X[token_id * d_model + h];
    }
}

//TODO: This should not be void...should be returning something. 
void pack_tokens_kernel(
    torch::Tensor X, 
    torch::Tensor topk_weights,
    torch::Tensor topk_experts,
    size_t topK,
    size_t total_assignments,
    torch::Tensor expert_offset_cpy, 
    torch::Tensor packed_X, 
    torch::Tensor packed_tokenId, 
    torch::Tensor packed_expert, 
    torch::Tensor packed_weight
    size_t hidden_dim){
        size_t threads = 256;
        size_t blocks = total_assignments;

        pack_tokens<<<threads, blocks>>>(
            X,
            topk_weights,
            topk_experts, 
            topK, 
            total_assignments, 
            expert_offset_cpy,
            packed_X, 
            packed_tokenId, 
            packed_expert, 
            packed_weight, 
            hidden_dim
        )
    }
    