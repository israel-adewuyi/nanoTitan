#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


#define BLOCK_N 16
#define BLOCK_M 16

template <typename T>
__global__ void bwd_grouped_gemm_up_proj_dW_cu(
    const T* __restrict__ X,                     // [#assignments, d_model]
    const int32_t* __restrict__ expert_offset, // [#experts, ]
    const T* __restrict__ dOut,               // [#assignments, d_in]
    T* __restrict__ dW,                      // [#experts, d_model, d_in]
    int hidden_dim,
    int d_in
){
    size_t hidden_dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t d_in_idx = blockIdx.y * blockDim.y + threadIdx.y;
    size_t expert_id = blockIdx.z;

    if(hidden_dim_idx >= hidden_dim || d_in_idx >= d_in){
        return;
    }

    int start = expert_offset[expert_id];
    int end = expert_offset[expert_id + 1];
    int num_assignments = end - start;

    float acc{0.0f};

    for(int i = 0; i < num_assignments; i++){
        int assignment = start + i;
        T x = X[assignment * hidden_dim + hidden_dim_idx];
        T dout = dOut[d_in_idx + d_in * assignment];
        acc += static_cast<float>(x) * static_cast<float>(dout);
    }
    dW[(expert_id * d_in * hidden_dim) + (hidden_dim_idx * d_in) + d_in_idx] = static_cast<T>(acc);
}


torch::Tensor bwd_grouped_gemm_up_proj_dW_kernel(
    torch::Tensor X,
    torch::Tensor expert_offset,
    torch::Tensor dOut
){
    TORCH_CHECK(X.is_contiguous(), "X must be contiguous");
    TORCH_CHECK(dOut.is_contiguous(), "dOut must be contiguous");

    int hidden_dim = X.size(1);
    int d_in = dOut.size(1);
    int num_experts = expert_offset.size(0) - 1;

    torch::Tensor dW =  torch::empty(
        {static_cast<int64_t>(num_experts), static_cast<int64_t>(hidden_dim), static_cast<int64_t>(d_in)},
        X.options()
    );

    c10::cuda::CUDAGuard guard(X.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 threads(BLOCK_N, BLOCK_M);
    dim3 blocks(
        ((hidden_dim + BLOCK_N - 1) / BLOCK_N),
        ((d_in + BLOCK_M - 1) / BLOCK_M),
        num_experts 
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        X.scalar_type(),
        "bwd_grouped_kernel_up_proj_dW",
        [&] {
            bwd_grouped_gemm_up_proj_dW_cu<scalar_t><<<blocks, threads, 0, stream>>>(
                X.data_ptr<scalar_t>(),
                expert_offset.data_ptr<int32_t>(),
                dOut.data_ptr<scalar_t>(),
                dW.data_ptr<scalar_t>(),
                hidden_dim,
                d_in
            );
        }
    );
    return dW;
}