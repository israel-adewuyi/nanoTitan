#pragma once
#include <cuda_runtime.h>
#include <cuda_check.h>

template <typename T>
class DeviceBuffer{
    public:
        DeviceBuffer(size_t count){
            CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
        }

        ~DeviceBuffer(){
            cudaFree(ptr_);
        }

        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        T* data(){
            return ptr_;
        }


    private:
        T* ptr_{nullptr};
        size_t count_{0};
}