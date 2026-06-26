#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <cuda_runtime.h>


template <typename T>
__global__ void backward_combine_kernel_cu(
    const T* __restrict__ expert_outputs,
    const int32_t* __restrict__ packed_tokenIds,
    const float* __restrict__ packed_topk_weights,
    const T* __restrict__ residual_stream_grad,
    float* __restrict__ packed_topk_weights_grad,
    T* __restrict__ expert_output_grad,
    size_t num_assignments,
    size_t hidden_dim
){
    // Assuming the MoEs do topk = 1
    size_t assignment = blockIdx.x;
    size_t thread_idx = threadIdx.x;

    if(assignment >= num_assignments){
        return;
    }

    size_t token_id = packed_tokenIds[assignment];

    float weight_grad = 0.0f;
    float weight = static_cast<float>(packed_topk_weights[assignment]);

    for(size_t h = thread_idx; h < hidden_dim; h += blockDim.x){
        size_t resid_stream_idx = token_id * hidden_dim + h;
        size_t out_idx = assignment * hidden_dim + h;

        float dy = static_cast<float>(residual_stream_grad[resid_stream_idx]);

        weight_grad += dy * static_cast<float>(expert_outputs[out_idx]);
        expert_output_grad[out_idx] = static_cast<T>(dy * weight);
    }
    atomicAdd(&packed_topk_weights_grad[assignment], weight_grad);
}

std::tuple<torch::Tensor, torch::Tensor> backward_combine_kernel(
    torch::Tensor expert_outputs,
    torch::Tensor packed_tokenIds,
    torch::Tensor packed_topk_weights,
    torch::Tensor residual_stream_grad,
    size_t hidden_dim,
    size_t num_assignments
){
    TORCH_CHECK(expert_outputs.is_contiguous(), "Tensor should be laid out in contiguous memory location");
    TORCH_CHECK(packed_tokenIds.is_contiguous(), "Tensor should be laid out in contiguous memory location");
    TORCH_CHECK(packed_topk_weights.is_contiguous(), "Tensor should be laid out in contiguous memory location");
    TORCH_CHECK(residual_stream_grad.is_contiguous(), "Tensor should be laid out in contiguous memory location");

    c10::cuda::CUDAGuard guard(expert_outputs.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    packed_topk_weights_grad = torch::zeros(
        packed_topk_weights.sizes(),
        packed_topk_weights.options().dtype(torch::kFloat32)
    );

    expert_output_grad = torch.zeros(
        expert_outputs.sizes(),
        expert_outputs.options()
    );

    dim3 threads(128);
    dim3 blocks(num_assignments);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        expert_outputs.scalar_type(),
        "combine_backward_kernel",
        [&] {
            backward_combine_kernel_cu<scalar_t><<<blocks, threads, 0, stream>>>(
                expert_outputs.data_ptr<scalar_t>(), 
                packed_tokenIds.data_ptr<int32_t>(),
                packed_topk_weights.data_ptr<float>(),
                residual_stream_grad.data_ptr<scalar_t>(),
                packed_topk_weights_grad.data_ptr<float>(),
                expert_output_grad.data_ptr<scalar_t>(),
                num_assignments,
                hidden_dim
            );
        }
    );

    return {expert_output_grad, packed_topk_weights_grad};
}