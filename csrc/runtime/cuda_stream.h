#pragma once
#include <cuda_runtime.h>

class CudaStream {
    public:
        CudaStream();

        ~CudaStream();

        CudaStream(const CudaStream&) = delete;
        CudaStream& operator=(const CudaStream&) = delete;

        cudaStream_t get() const;
    
    private:
        cudaStream_t stream_ = nullptr;

};