#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "cuda_check.h"


__global__ void count_experts(
    const int* topk_experts,
    const int* mask, 
    int* expert_count, 
    size_t num_topk_experts,
    size_t N){
    /*
        Kernel is executed for each token i.e each logical thread maps to each token
    */
    // Get the global index for this thread
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N && (mask == nullptr || mask[idx] != 0)){
        for(size_t i = 0; i < num_topk_experts; i++){
            // To get the experts for token idx, offset by the index of the token as well as the number of topK tokens. 
            // idx * num_topK experts gets you to the row of token idx
            // +i gives you the ith expert chosen by token idx 
            int expert = topk_experts[idx * num_topk_experts + i];
            atomicAdd(&expert_count[expert], 1);
        }
    }
}


torch::Tensor count_expert_kernel(torch::Tensor topk_experts_per_token, torch::Tensor mask, size_t num_experts, size_t num_topk_experts){
    int N = topk_experts_per_token.size(0); // Number of tokens
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    c10::cuda::CUDAGuard guard(topk_experts_per_token.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    auto expert_count = torch::empty(
        {static_cast<int64_t>(num_experts)},
        torch::TensorOptions()
            .dtype(torch::kInt32)
            .device(topk_experts_per_token.device())
    );

    cudaMemset(expert_count.data_ptr<int>(), 0, num_experts * sizeof(int));

    count_experts<<<blocks, threads, 0, stream>>>(
        topk_experts_per_token.data_ptr<int>(),
        mask.data_ptr<int>(),
        expert_count.data_ptr<int>(),
        num_topk_experts,
        N
    );
    return expert_count;
}