#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define BLOCK_M 16
#define BLOCK_N 64
#define MAX_TOKEN_BLOCKS 64 // If number of tokens routed to some expert exceeds 

template <typename T>
__global__ void grouped_gemm_cu(
    const T* __restrict__ X, //[#assignments, d_model]
    const int32_t* __restrict__ expert_offset, //[#experts, ]
    const T* __restrict__ Weights, //[#experts, d_model, d_ffn_in]
    T* __restrict__ out, //[#assignments, d_ffn_in]
    size_t hidden_dim_A, //TODO: Bad Notation, sir. I should change this. WTF does hidden_dimA and B mean for someone seeing the repo for the firs time?
    size_t hidden_dim_B
){
    size_t expert_id = blockIdx.z;
    int local_row = threadIdx.y + blockIdx.y * blockDim.y;
    int out_col = threadIdx.x + blockIdx.x * blockDim.x;

    int start = expert_offset[expert_id];
    int end = expert_offset[expert_id + 1];
    int numTokens = end - start;

    if(out_col >= hidden_dim_B || local_row >= numTokens){
        return;
    }

    float acc{0.0f};
    for(int i = 0; i < hidden_dim_A; i++){
        T x = X[(start + local_row) * hidden_dim_A + i];
        T w = Weights[(expert_id * hidden_dim_A * hidden_dim_B) + (i * hidden_dim_B) + out_col];
        acc += static_cast<float>(x) * static_cast<float>(w);
    }

    out[(start + local_row) * hidden_dim_B + out_col] = static_cast<T>(acc);
}


torch::Tensor grouped_gemm_kernel(
    torch::Tensor X, 
    torch::Tensor expert_offset,
    torch::Tensor weights
){
    int assignments = X.size(0);
    int d_model = X.size(1);
    int num_experts = weights.size(0);
    int d_ffn_in = weights.size(2);

    torch::Tensor Out = torch::empty(
        {static_cast<int64_t>(assignments), static_cast<int64_t>(d_ffn_in)},
        X.options()
    );

    int max_tokens_per_expert = 0;
    for(int i = 0; i < num_experts; i++){
        int tokens_for_expert = expert_offset[i + 1].item<int>() - expert_offset[i].item<int>();
        max_tokens_per_expert = std::max(max_tokens_per_expert, tokens_for_expert);
    }

    dim3 threads(BLOCK_M, BLOCK_N);
    dim3 blocks(
        ((d_ffn_in + BLOCK_M - 1) / BLOCK_M),
        ((max_tokens_per_expert + BLOCK_N - 1) / BLOCK_N),
        num_experts
    );

    c10::cuda::CUDAGuard guard(X.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        X.scalar_type(),
        "grouped_gemm_kernel",
        [&] {
            grouped_gemm_cu<scalar_t><<<blocks, threads, 0, stream>>>(
                X.data_ptr<scalar_t>(),
                expert_offset.data_ptr<int32_t>(),
                weights.data_ptr<scalar_t>(),
                Out.data_ptr<scalar_t>(),
                d_model,
                d_ffn_in
            );
        }
    );

    return Out;
}