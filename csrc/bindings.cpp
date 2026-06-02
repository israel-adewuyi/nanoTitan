#include <torch/extension.h>

// Declare the function implemented in random.cpp
torch::Tensor random_op(torch::Tensor t, int x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("random_op", &random_op, "Random op");
}