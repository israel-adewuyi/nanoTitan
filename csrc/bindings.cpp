#include <torch/extension.h>
#include <cstdint>
#include <string>

// Declare the function implemented in random.cpp
torch::Tensor random_op(torch::Tensor t, int x);
void copy_scalar(torch::Tensor src, torch::Tensor dest, uint64_t N);
void copy_vector(torch::Tensor src, torch::Tensor dest);
torch::Tensor count_expert_kernel(torch::Tensor topk_experts_per_token, torch::Tensor mask, size_t num_experts, size_t num_topk_experts);
torch::Tensor combine_tokens_kernel(
    torch::Tensor expert_outputs,          
    torch::Tensor packed_tokenId,      
    torch::Tensor packed_topk_weights,
    size_t num_tokens,    
    size_t d_model   
);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> pack_tokens_kernel(
    torch::Tensor X,
    torch::Tensor topk_weights,
    torch::Tensor topk_experts,
    torch::Tensor expert_offset_cpy
);

std::tuple<torch::Tensor, torch::Tensor> combine_kernel_backward(
    torch::Tensor expert_outputs,
    torch::Tensor packed_tokenIds,
    torch::Tensor packed_topk_weights,
    torch::Tensor residual_stream_grad,
    size_t hidden_dim
);

std::tuple<torch::Tensor, torch::Tensor> pack_kernel_backward(
    torch::Tensor packed_X_grad,
    torch::Tensor packed_topk_weights_grad,
    torch::Tensor packed_tokenId,
    torch::Tensor packed_expert,
    torch::Tensor topk_experts
);

torch::Tensor gemm_kernel(torch::Tensor A, torch::Tensor B, const std::string& implementation = "tiled");
torch::Tensor tiled_gemm_kernel(torch::Tensor A, torch::Tensor B);
torch::Tensor naive_gemm_kernel(torch::Tensor A, torch::Tensor B);
torch::Tensor grouped_gemm_kernel(
    torch::Tensor X, 
    torch::Tensor expert_offset,
    torch::Tensor weights
);
torch::Tensor bwd_grouped_gemm_up_proj_dW_kernel(
    torch::Tensor X,
    torch::Tensor expert_offset,
    torch::Tensor dOut
);
torch::Tensor bwd_grouped_gemm_up_proj_dX_kernel(
    torch::Tensor W_gate,
    torch::Tensor expert_offset,
    torch::Tensor dOut
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("random_op", &random_op, "Random op");
    m.def("copy_scalar", &copy_scalar, "Copy scalar");
    m.def("copy_vector", &copy_vector, "Copy vector");
    m.def("count_expert_kernel", &count_expert_kernel, "Kernel to count the number of experts for incoming resid stream");
    m.def("pack_tokens_kernel", &pack_tokens_kernel, "[num_tokens d_model] -> [num_tokens * topK d_model] packing");

    m.def("combine_tokens_kernel", &combine_tokens_kernel, "Kernel to weight-average token vectors from K experts into the residual stream");
    m.def("combine_kernel_backward", &combine_kernel_backward, "Kernel to run bwd pass for the expert outputs combination");

    m.def(
        "gemm_kernel",
        &gemm_kernel,
        "GEMM kernel",
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("implementation") = "tiled"
    );
    m.def("tiled_gemm_kernel", &tiled_gemm_kernel, "Tiled GEMM kernel");
    m.def("naive_gemm_kernel", &naive_gemm_kernel, "Naive GEMM kernel");

    m.def("pack_kernel_backward", &pack_kernel_backward, "Kernel to run bwd pass on the pack ops");

    m.def("grouped_gemm_kernel", &grouped_gemm_kernel, "Kernel to run grouped expert fwd pass");

    m.def("bwd_grouped_gemm_up_proj_dW_kernel",
    &bwd_grouped_gemm_up_proj_dW_kernel,
    "Kernel to compute derivatives w.r.t W_gate (up proj) in grouped gemm");

    m.def("bwd_grouped_gemm_up_proj_dX_kernel",
    &bwd_grouped_gemm_up_proj_dX_kernel,
    "Kernel to compute derivatives w.r.t X (up proj) in grouped gemm");

}
