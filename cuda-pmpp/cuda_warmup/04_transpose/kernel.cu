#include <torch/extension.h>
#include <cuda_runtime.h>

torch::Tensor transpose_naive(torch::Tensor x);
torch::Tensor transpose_tiled(torch::Tensor x);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_naive", &transpose_naive, "transpose_naive (CUDA)");
    m.def("transpose_tiled", &transpose_tiled, "transpose_tiled (CUDA)");
}
