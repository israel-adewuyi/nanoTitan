#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_runtime.h>


template <typename T>
__global__ void pack_tokens_kernel_cu(
    const T* __restrict__ X,                                // [num_tokens, d_model]
    const float* __restrict__ topk_weights,                     // [num_tokens, top_K]
    const int32_t* __restrict__ topk_experts,                // [num_tokens, top_K]
    size_t topK,
    size_t total_assignments,
    int32_t* expert_offset_cpy,
    T* packed_X,
    int32_t* packed_tokenId,
    int32_t* packed_expert,
    float* packed_topk_weights,
    size_t hidden_dim
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

    __shared__ int32_t slot;

    if(thread_idx == 0){
        slot = atomicAdd(&expert_offset_cpy[expert], 1);
        packed_tokenId[slot] = token_id;
        packed_expert[slot] = expert;
        packed_topk_weights[slot] = topk_weights[token_id * topK + expert_idx];
    }

    __syncthreads();

    for(size_t h = thread_idx; h < hidden_dim; h += blockDim.x){
        packed_X[slot * hidden_dim + h] = X[token_id * hidden_dim + h];
    }
}


std::tuple<torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor> pack_tokens_kernel(
    torch::Tensor X,                   //[num_tokens, d_model]
    torch::Tensor topk_weights,        //[num_tokens, topK]
    torch::Tensor topk_experts,        //[num_tokens, topK]
    torch::Tensor expert_offset_cpy,   //[num_experts]
){
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(topk_weights.is_contiguous(), "Weights for the topK should be off");

    size_t num_tokens = X.sizes(0);
    size_t topK = topk_weights.sizes(1);
    size_t hidden_dim = X.sizes(1);
    size_t total_assignments = num_tokens * topK;   

    torch.Tensor packed_X = torch::empty(
        {total_assignments, hidden_dim},
        X.options()
    );
    torch.Tensor packed_expert = torch::empty(
        {total_assignments},
        X.options().dtype(torch::Int32)
    );
    torch.Tensor packed_tokenId = torch.empty(
        {total_assignments},
        X.options().dtype(torch::Int32)
    );
    torch.Tensor packed_topk_weights = torch.empty(
        {total_assignments},
        X.options().dtype(torch::kFloat32)
    );

    size_t threads = 256;
    size_t blocks = total_assignments;

    c10::cuda::CUDAGuard guard(X.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        X.scalar_type(),
        "pack_tokens_kernel",
        [&] {
            pack_tokens_kernel_cu<scalar_t><<<blocks, threads, 0, stream>>>(
                X.data_ptr<scalar_t>(),
                topk_weights.data_ptr<float>(),
                topk_experts.data_ptr<int32_t>(), 
                topK, 
                total_assignments, 
                expert_offset_cpy.data_ptr<int32_t>(),
                packed_X.data_ptr<scalar_t>(),
                packed_tokenId.data_ptr<int32_t>(), 
                packed_expert.data_ptr<int32_t>(), 
                packed_topk_weights.data_ptr<float>(), 
                hidden_dim
            );
        }
    );

    return {packed_X, packed_tokenId, packed_expert, packed_topk_weights};
}
    