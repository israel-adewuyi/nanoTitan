#pragma once
#include <cuda_runtime.h>
#include <cuda_check.h>

class CudaEvent{
    public:
        CudaEvent(){
            CUDA_CHECK(cudaEventCreate(&event_));
        }

        ~CudaEvent(){
            cudaEventDestroy(event_);
        }

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