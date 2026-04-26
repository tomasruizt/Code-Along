#include <torch/extension.h>
#include <cuda_runtime.h>

torch::Tensor block_reduce_sum(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_reduce_sum", &block_reduce_sum, "block_reduce_sum (CUDA)");
}
