#include <torch/torch.h>

using namespace std;

torch::Tensor random_op(torch::Tensor t, int x){
    return torch::zeros({x}, t.options());
}