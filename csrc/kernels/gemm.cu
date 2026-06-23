#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>


template <typename T>
__global__ void naive_gemm_cu(
    const T* __restrict__ A,
    const T* __restrict__ B, 
    T* __restrict__ C,
    int M,
    int N,
    int K
){
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    
    if (row >= M || col >= N){
        return;
    }

    float acc = 0.0f;

    for (int i = 0; i < K; i++){
        acc += float(A[row * K + i]) * float(B[i * N + col]);
    }
    
    C[row * N + col] = T(acc);
}

torch::Tensor naive_gemm_kernel(
    torch::Tensor A,
    torch::Tensor B
){
    TORCH_CHECK(A.is_contiguous(), "A should be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B should be contiguous");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "Inner dimension should be the same for the matrices in the matmul");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    dim3 threads (16, 16);
    dim3 blocks (
        (M + threads.x - 1) / threads.x,
        (N + threads.y - 1) / threads.y
    );

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        A.scalar_type(),
        "naive_gemm_kernel",
        [&] {
            naive_gemm_cu<scalar_t><<<blocks, threads>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                M, 
                N, 
                K
            );
        }
    );
}