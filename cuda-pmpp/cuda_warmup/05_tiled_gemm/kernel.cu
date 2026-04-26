#include <torch/extension.h>
#include <cuda_runtime.h>

torch::Tensor gemm(torch::Tensor a, torch::Tensor b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "tiled gemm (CUDA)");
}
