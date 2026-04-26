#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void warp_reduce_sum_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float val = x[tid];

    // Butterfly reduction within the warp.
    val += __shfl_down_sync(0xFFFFFFFF, val, 16);
    val += __shfl_down_sync(0xFFFFFFFF, val, 8);
    val += __shfl_down_sync(0xFFFFFFFF, val, 4);
    val += __shfl_down_sync(0xFFFFFFFF, val, 2);
    val += __shfl_down_sync(0xFFFFFFFF, val, 1);

    int lane = threadIdx.x & 31;
    int warp_id = tid >> 5;
    if (lane == 0) {
        out[warp_id] = val;
    }
}

torch::Tensor warp_reduce_sum(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");
    int n = x.numel();
    TORCH_CHECK(n % 32 == 0, "n must be a multiple of 32");

    auto out = torch::empty({n / 32}, x.options());
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    warp_reduce_sum_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("warp_reduce_sum", &warp_reduce_sum, "warp_reduce_sum (CUDA)");
}
