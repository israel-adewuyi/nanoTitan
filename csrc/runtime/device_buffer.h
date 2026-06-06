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

        // Move constructor
        DeviceBuffer(DeviceBuffer&& other) noexcept: ptr_(other.ptr_), count_(other.count_){
            other.ptr_ = nullptr;
            other.count_ = 0;
        }

        // Disable copy constructor and copy assignment operator
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;

        T* data(){
            return ptr_;
        }

        int64_t size() const {
            return count_;
        }

        int64_t bytes() const {
            return count_ * sizeof(T);
        }

    private:
        T* ptr_{nullptr};
        size_t count_{0};
};