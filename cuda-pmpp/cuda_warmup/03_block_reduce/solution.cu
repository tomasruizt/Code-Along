#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BLOCK = 256;

__global__ void block_reduce_sum_kernel(const float* __restrict__ x, float* __restrict__ out, int n) {
    __shared__ float sdata[BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK + tid;

    sdata[tid] = (idx < n) ? x[idx] : 0.0f;
    __syncthreads();

    // Tree reduction in shared memory.
    for (int s = BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

torch::Tensor block_reduce_sum(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "input must be CUDA");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "input must be float32");
    TORCH_CHECK(x.is_contiguous(), "input must be contiguous");

    int n = x.numel();
    auto out = torch::zeros({}, x.options());
    const int threads = BLOCK;
    const int blocks = (n + threads - 1) / threads;
    block_reduce_sum_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_reduce_sum", &block_reduce_sum, "block_reduce_sum (CUDA)");
}
