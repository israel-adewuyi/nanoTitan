#pragma once
#include <cuda_runtime.h>
#include <cuda_check.h>

class CudaEvent{
    public:
        CudaEvent(){
            CUDA_CHECK(cudaEventCreate(&event_));
        }

        ~CudaEvent(){
            if(event_) cudaEventDestroy(event_);
        }

        // Move constructor
        CudaEvent(CudaEvent&& other) noexcept : event_(other.event_){
            other.event_ = nullptr;
        }

        // Movement assignment operator
        CudaEvent& operator=(CudaEvent&& other) noexcept {
            if(this != &event){
                if(event_)cudaEventDestroy(event_);
                event_ = other.event_;
                other.event_ = nullptr;
            }
            return *this;
        }

        // Disable copy constructor and copy assignment operator
        CudaEvent(const CudaEvent&) = delete;
        CudaEvent& operator=(const CudaEvent&) = delete;

        void record(cudaStream_t stream){
            CUDA_CHECK(cudaEventRecord(event_, stream));
        }

        void wait(cudaStream_t stream){
            CUDA_CHECK(cudaStreamWaitEvent(stream, event_, 0));
        }

        cudaEvent_t get() const {
            return event_;
        }


    private:
        cudaEvent_t event_ = nullptr;
};