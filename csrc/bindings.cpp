#include <torch/extension.h>
#include <cstdint>

// Declare the function implemented in random.cpp
torch::Tensor random_op(torch::Tensor t, int x);
void copy_scalar(torch::Tensor src, torch::Tensor dest, uint64_t N);
void copy_vector(torch::Tensor src, torch::Tensor dest);
void count_expert_kernel(torch::Tensor topk_experts_per_token, torch::Tensor mask, size_t num_experts, size_t num_topk_experts);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("random_op", &random_op, "Random op");
    m.def("copy_scalar", &copy_scalar, "Copy scalar");
    m.def("copy_vector", &copy_vector, "Copy vector");
    m.def("count_expert_kernel", &count_expert_kernel, "Kernel to count the number of experts for incoming resid stream");
}
