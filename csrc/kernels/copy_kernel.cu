#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include <iostream>
#include <cassert>
#include <cstdint>
#include <vector>

using namespace std;
using Vec = uint4;

template <typename T>
__global__ void copy_kernel_scalar(T* src, T* dest, uint64_t N){
    // get global index
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        dest[i] = src[i];
    }
}

template <typename T>
__global__ void copy_kernel_vector(T* src, T* dest, uint64_t N){
    // get global index
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // We assume total bytes is divisible by int4 (16 bytes)
    uint64_t totalBytes = N * sizeof(T);
    uint64_t vecN = totalBytes / sizeof(Vec);

    if(i < vecN){
        reinterpret_cast<Vec*>(dest)[i] = reinterpret_cast<Vec*>(src)[i];
    }
}


void peer_copy_scalar(torch::Tensor src, size_t src_device, torch::Tensor dest, size_t dest_device, size_t count){
    TORCH_CHECK(src.is_contiguous(), "src must be contiguous");
    TORCH_CHECK(dest.is_contiguous(), "dest must be contiguous");

    c10::cuda::CUDAGuard guard(dest_device);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(dest_device);

    CUDA_CHECK(cudaMemcpyPeerAsync(
        dest.data_ptr(), dest_device,
        src.data_ptr(), src_device,
        static_cast<size_t>(count), stream
    ));
}


void copy_scalar(torch::Tensor src, torch::Tensor dest, uint64_t N){
    int threads = 256;
    int blocks  = (src.numel() + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::kHalf,
        at::kBFloat16,
        src.scalar_type(),
        "copy_scalar",
        [&] {
            using T = scalar_t;
            copy_kernel_scalar<T><<<blocks, threads>>>(
                src.data_ptr<T>(),
                dest.data_ptr<T>(),
                static_cast<uint64_t>(src.numel())
            );
        }
    );
}

void copy_vector(torch::Tensor src, torch::Tensor dest){
    size_t vecN = (src.numel() * src.element_size()) / sizeof(Vec);
    assert (sizeof(Vec) == 16);
    int threads = 256;
    int blocks = (vecN + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, src.scalar_type(), "copy_vector", [&] {
        using T = scalar_t;
        copy_kernel_vector<T><<<blocks, threads>>>(
            src.data_ptr<T>(),
            dest.data_ptr<T>(),
            static_cast<uint64_t>(src.numel())
            );
        }
    );
}
