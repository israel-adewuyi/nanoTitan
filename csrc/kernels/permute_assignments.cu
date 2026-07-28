#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>


#define MAX_THREADS_PER_BLOCK 128


template <typename T>
__global__ void permute_expert_assignment_cu(
    const T* __restrict__ Input,
    T* __restrict__ Out,
    const int32_t* __restrict__ src_matrix,
    const int32_t* __restrict__ src_matrix_offset,
    const int32_t* __restrict__ dest_matrix_offset,
    int ep_group_size,
    int num_local_experts,
    int d_model
){
    int expert_idx = blockIdx.y % num_local_experts;
    int src_rank_idx = blockIdx.y / num_local_experts;

    int rank_expert_assignment = blockIdx.x * blockDim.x + threadIdx.x;
    int rank_expert_idx = src_rank_idx * num_local_experts  + expert_idx;
    int destrank_expert_idx = expert_idx * ep_group_size + src_rank_idx;

    if(rank_expert_assignment >= src_matrix[src_rank_idx * num_local_experts + expert_idx]){
        return;
    }

    for(int i = 0; i < d_model; i++){
        int src_row = src_matrix_offset[rank_expert_idx] + rank_expert_assignment;
        int dest_row = dest_matrix_offset[destrank_expert_idx] + rank_expert_assignment;
        Out[(dest_row * d_model) + i]  = Input[(src_row * d_model) + i];
    }
}


torch::Tensor permute_expert_assignment_kernel(
    torch::Tensor X,
    torch::Tensor src_matrix
){
    TORCH_CHECK(X.is_contiguous(), "X should be contiguous");
    TORCH_CHECK(src_matrix.is_contiguous(), "src_matrix should be contiguous");

    c10::cuda::CUDAGuard guard(X.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    int d_model = X.size(1);
    int assignments = X.size(0);
    // These two break down during the backward kernel, but they still logically mean the same thing
    int ep_group_size = src_matrix.size(0);
    int num_local_experts = src_matrix.size(1);

    torch::Tensor Out = torch::empty(
        {static_cast<int64_t>(assignments), static_cast<int64_t>(d_model)},
        X.options()
    );

    torch::Tensor src_matrix_1D = src_matrix.view({num_local_experts * ep_group_size});
    auto src_offset = at::cumsum(src_matrix_1D, 0, at::kInt) - src_matrix_1D;
    torch::Tensor dest_matrix_1D = src_matrix.transpose(1, 0).contiguous().view({(num_local_experts * ep_group_size)});
    auto dest_offset = at::cumsum(dest_matrix_1D, 0, at::kInt) - dest_matrix_1D;

    TORCH_CHECK(src_offset.scalar_type() == at::kInt);
    TORCH_CHECK(dest_offset.scalar_type() == at::kInt);

    int max_tokens = src_matrix.max().item<int>();

    dim3 threads(MAX_THREADS_PER_BLOCK);
    dim3 blocks(
        ((max_tokens + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK),
        (ep_group_size * num_local_experts)
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        X.scalar_type(),
        "regroup_kernel",
        [&] {
            permute_expert_assignment_cu<scalar_t><<<blocks, threads, 0, stream>>>(
                X.data_ptr<scalar_t>(),
                Out.data_ptr<scalar_t>(),
                src_matrix.data_ptr<int32_t>(),
                src_offset.data_ptr<int32_t>(),
                dest_offset.data_ptr<int32_t>(),
                ep_group_size,
                num_local_experts,
                d_model
            );
        }
    );
    return Out;
}