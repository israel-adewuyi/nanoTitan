#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_check.h>
#include <iostream>
#include <cassert>
#include <vector>

using namespace std;
using Vec = uint4;

template <typename T>
__global__ void copy_kernel_scalar(T* src, T* dest, int N){
    // get global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        dest[i] = src[i];
    }
}

template <typename T>
__global__ void copy_kernel_vector(T* src, T* dest, int N){
    // get global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // We assume total bytes is divisible by int4 (16 bytes)
    int totalBytes = N * sizeof(T);
    int vecN = totalBytes / sizeof(Vec);

    if(i < vecN){
        reinterpret_cast<Vec*>(dest)[i] = reinterpret_cast<Vec*>(src)[i];
    }
}

void copy_scalar(torch::Tensor src, torch::Tensor dest, int N){
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
                src.numel()
            );
        }
    );
}

// int main(){
//     int N;
//     cin >> N;

//     vector<float>src(N, 12345);
//     vector<float>dest(N);

//     float *c_src, *c_dest;

//     CUDA_CHECK(cudaMalloc(&c_src, N * sizeof(float)));
//     CUDA_CHECK(cudaMalloc(&c_dest, N * sizeof(float)));

//     CUDA_CHECK(cudaMemcpy(c_src, src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

//     int threads = 256;
//     int blocks  = (N + threads - 1) / threads;

//     copy_kernel_vector<<<blocks, threads>>>(c_src, c_dest, N);

//     CUDA_KERNEL_CHECK();

//     CUDA_CHECK(cudaMemcpy(dest.data(), c_dest, N * sizeof(float), cudaMemcpyDeviceToHost));

//     CUDA_CHECK(cudaFree(c_src));
//     CUDA_CHECK(cudaFree(c_dest));

//     for(auto &num : dest){
//         assert(num == 12345);
//         cout << num << endl;
//     }

//     return 0;
// }
