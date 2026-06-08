#include <torch/extension.h>
#include <cstdint>

// Declare the function implemented in random.cpp
torch::Tensor random_op(torch::Tensor t, int x);
void copy_scalar(torch::Tensor src, torch::Tensor dest, uint64_t N);
void copy_vector(torch::Tensor src, torch::Tensor dest);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("random_op", &random_op, "Random op");
    m.def("copy_scalar", &copy_scalar, "Copy scalar");
    m.def("copy_vector", &copy_vector, "Copy vector");
}
