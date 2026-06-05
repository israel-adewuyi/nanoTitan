#include <gtest/gtest.h>
#include <cuda_stream.h>
#include <cuda_check.h>
#include <device_buffer.h>

TEST(CudaStreamTest, canLaunchKernel){
    // Initialize stream
    CudaStream stream;
    // Initialize buffer
    DeviceBuffer<float> x(1024);

    // I honestly don't know how this work in-depth. Will revisit when I get the testing module to work first.
    fill_kernel<<<4, 256, 0, stream.get()>>>(x.data(), 1024, 1.0f);

    CUDA_KERNEL_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(stream.get()));
}