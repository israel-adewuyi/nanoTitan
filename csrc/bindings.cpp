#include <torch/extension.h>
#include <cstdint>

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
void pack_tokens_kernel(
    torch::Tensor X,
    torch::Tensor topk_weights,
    torch::Tensor topk_experts,
    size_t topK,
    size_t total_assignments,
    torch::Tensor expert_offset_cpy,
    torch::Tensor packed_X,
    torch::Tensor packed_tokenId,
    torch::Tensor packed_expert,
    torch::Tensor packed_weight,
    size_t hidden_dim
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

torch::Tensor naive_gemm_kernel(torch::Tensor A, torch::Tensor B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("random_op", &random_op, "Random op");
    m.def("copy_scalar", &copy_scalar, "Copy scalar");
    m.def("copy_vector", &copy_vector, "Copy vector");
    m.def("count_expert_kernel", &count_expert_kernel, "Kernel to count the number of experts for incoming resid stream");
    m.def("pack_tokens_kernel", &pack_tokens_kernel, "[num_tokens d_model] -> [num_tokens * topK d_model] packing");

    m.def("combine_tokens_kernel", &combine_tokens_kernel, "Kernel to weight-average token vectors from K experts into the residual stream");
    m.def("combine_kernel_backward", &combine_kernel_backward, "Kernel to run bwd pass for the expert outputs combination");

    m.def("naive_gemm_kernel", &naive_gemm_kernel, "Naive GEMM kernel");

    m.def("pack_kernel_backward", &pack_kernel_backward, "Kernel to run bwd pass on the pack ops");
}
