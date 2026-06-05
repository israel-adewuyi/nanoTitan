#pragma once
#include <cstdint>

template <typename T>
struct Tensor2D{
    T* ptr;
    int64_t dim0;
    int64_t dim1;
    int64_t stride0;
    int64_t stride1;

    __host__ __device__ T& operator()(int64_t i, int64_t j) const {
        return ptr[i * stride0 + j * stride1]
    }
};