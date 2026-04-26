#include <ATen/ops/empty_like.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void kernel(float* x, float* out, int n) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    float val = 0.0;
    if (i < n) {
        val = x[i];
    }
    for (int offset = 1; offset <= 16; offset *= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    bool isThread0InWarp = threadIdx.x % 32 == 0;
    if (i < n && isThread0InWarp) {
        out[i / 32] = val;
    }
}

torch::Tensor warp_reduce_sum(torch::Tensor x) {
    const int n = x.numel();
    const int nThreads = 256;
    const int nBlocks = (n + nThreads - 1) / nThreads;
    const int outSize = (n + 32 - 1) / 32;
    const auto out = torch::empty(outSize, x.options());
    kernel<<<nBlocks, nThreads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warp_reduce_sum", &warp_reduce_sum, "warp_reduce_sum (CUDA)");
}
