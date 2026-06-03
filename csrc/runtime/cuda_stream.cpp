#include <cuda_runtime.h>
#include "cuda_stream.h"
#include "cuda_check.h"


CudaStream::CudaStream() {
    CUDA_CHECK(cudaStreamCreate(&stream_));
}

CudaStream::~CudaStream() {
    cudaStreamDestroy(stream_);
}

cudaStream_t CudaStream::get() const {
    return stream_;
}

