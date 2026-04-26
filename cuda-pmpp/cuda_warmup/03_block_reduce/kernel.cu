#include <ATen/ops/zeros.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void kernel(float* x, float* sum, int n) {
    // Each warp reduces locally
    // Each block reduces using SMEM
    // Then atomicAdd to HBM

    __shared__ float warpSums[8];

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    float val = 0.0;
    if (i < n) {
        val = x[i];
    }
    // warp reduction
    for (int offset = 1; offset <= 16; offset *= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    // block reduction
    const bool isThread0InWarp = threadIdx.x % 32 == 0;
    if (isThread0InWarp) {
        int warpIdx = threadIdx.x / 32;
        warpSums[warpIdx] = val;
    }
    // syncs all threads in block
    // makes sure that warpSums is populated before reading below
    __syncthreads();

    // Comment:
    // Tomas: This could be replaced with a tree reduction, which reduces the
    // number of consecutive cycles from num_warps to log2(num_warps), but introduces
    // potentially extra synchthreads().
    //
    // Claude Code suggests the tree reduction is not that good, but instead that
    // a single warp could load all values from SMEM and do a warp shuffle reduction.
    // This prevents introducing syncthreads() and sequential reads from SMEM. Very smart.
    const bool isThread0InBlock = threadIdx.x == 0;
    if (isThread0InBlock) {
        float blockSum = 0.0;
        for (int sm_i = 0; sm_i < 8; sm_i++) {
            blockSum += warpSums[sm_i];
        }
        atomicAdd(sum, blockSum);
    }
}

torch::Tensor block_reduce_sum(torch::Tensor x) {
    const int n = x.numel();
    const int nThreads = 256;  // 256/32=8 warps
    const int nBlocks = (n + nThreads - 1) / nThreads;
    const auto out = torch::zeros({}, x.options());
    kernel<<<nBlocks, nThreads>>>(x.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("block_reduce_sum", &block_reduce_sum, "block_reduce_sum (CUDA)");
}
