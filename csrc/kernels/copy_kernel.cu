#include <cuda_runtime.h>
#include <cuda_check.h>
#include <iostream>
#include <cassert>
#include <vector>

using namespace std;

__global__ void copy_kernel(float* src, float* dest, int N){
    // get index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if(i < N){
        dest[i] = src[i];
    }
}

int main(){
    int N;
    cin >> N;

    vector<float>src(N, 12345);
    vector<float>dest(N);

    float *c_src, *c_dest;

    CUDA_CHECK(cudaMalloc(&c_src, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&c_dest, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(c_src, src.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    copy_kernel<<<blocks, threads>>>(c_src, c_dest, N);

    CUDA_KERNEL_CHECK()

    CUDA_CHECK(cudaMemcpy(dest.data(), c_dest, N * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(c_src));
    CUDA_CHECK(cudaFree(c_dest));

    for(auto &num : dest){
        assert(num == 12345);
    }

    return 0;
}
