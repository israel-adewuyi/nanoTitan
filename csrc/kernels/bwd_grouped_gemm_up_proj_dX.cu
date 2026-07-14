#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


#define BLOCK_N 16
#define BLOCK_M 16
#define MAX_TOKENS_PER_BLOCK 256

template <typename T>
__global__ void bwd_grouped_gemm_up_proj_dX_cu(
    const T* __restrict__ W_gate,               // [#expert, hidden_dim, d_in]
    const int32_t* __restrict__ expert_offset, // [#experts, ]
    const T* __restrict__ dOut,               // [#assignments, d_in]
    T* __restrict__ dX,                      // [#assignments, d_model]
    int hidden_dim,
    int d_in
){
    size_t expert_id = blockIdx.z;
    size_t local_assignment = blockIdx.y * blockDim.y + threadIdx.y;
    int ass_idx;//fill later
    
    int start = expert_offset[expert_id];
    int end = expert_offset[expert_id + 1];

    int assignment_idx = start + local_assignment;

    int col_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(local_assignment >= end - start || col_idx >= hidden_dim){
        return;
    }

    float acc{0.0f};

    for(int i = 0; i < d_in; i++){
        T d_out_i = dOut[assignment_idx * d_in + i];
        T wei = W_gate[(expert_id * d_in * hidden_dim) + (col_idx * d_in) + i];
        acc += static_cast<float>(d_out_i) * static_cast<float>(wei);
    }
    dX[(assignment_idx * hidden_dim) + col_idx] = static_cast<T>(acc);
}


torch::Tensor bwd_grouped_gemm_up_proj_dX_kernel(
    torch::Tensor W_gate,
    torch::Tensor expert_offset,
    torch::Tensor dOut,
){
    TORCH_CHECK(W_gate.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(dOut.is_contiguous(), "dOut must be contiguous");

    int assignments = dOut.size(0);
    int hidden_dim = W_gate.size(1);
    int d_in = W_gate.size(2);

    torch::Tensor dX =  torch::empty(
        {static_cast<int64_t>(assignments), static_cast<int64_t>(hidden_dim)},
        W_gate.options()
    );

    c10::cuda::CUDAGuard guard(W_gate.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    dim3 threads(BLOCK_N, BLOCK_M);
    dim3 blocks(
        ((hidden_dim + BLOCK_N - 1) / BLOCK_N),
        ((MAX_TOKENS_PER_BLOCK + BLOCK_M - 1) / BLOCK_M),
        num_experts 
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        W_gate.scalar_type(),
        "bwd_grouped_kernel_up_proj_dX",
        [&] {
            bwd_grouped_gemm_gate_proj_cu<scalar_t><<<blocks, threads, 0, stream>>>(
                W_gate.data_ptr<scalar_t>(),
                expert_offset.data_ptr<int32_t>(),
                dOut.data_ptr<scalar_t>(),
                dX.data_ptr<scalar_t>(),
                hidden_dim,
                d_in
            );
        }
    );
    return dX;
}