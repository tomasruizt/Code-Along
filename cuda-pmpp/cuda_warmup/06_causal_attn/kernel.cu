#include <ATen/ops/empty.h>
#include <cmath>
#include <limits>
#include <torch/extension.h>
#include <cuda_runtime.h>

// Causal attention scores: S = Q @ K^T with the strict upper triangle masked to -inf.
//
// Q, K: [N, D]   (row-major, contiguous)
// S:    [N, N]   S[i, j] = sum_d Q[i, d] * K[j, d]   for j <= i
//                S[i, j] = -inf                       for j  > i
//
// This is a tiled GEMM where the second operand is K^T (read K[col, d] instead of B[d, col]),
// plus a causal mask applied at the store. Build on the 05_tiled_gemm pattern.

constexpr int BN = 32;  // tile of S along both row and column

int cdiv(int n, int d) {
    return (n + d - 1) / d;
}

__global__ void causal_qk_kernel(const float* q,
                                 const float* k,
                                 float* out,
                                 int N,
                                 int D,
                                 int stride_q0,
                                 int stride_q1,
                                 int stride_k0,
                                 int stride_k1,
                                 int stride_out0,
                                 int stride_out1) {
    __shared__ float warpSums[4];

    const int tx = threadIdx.x;
    const int col = blockIdx.x;
    const int row = blockIdx.y;

    if (tx == 0 && col > row) {  // fill with -inf
        out[row * stride_out0 + col * stride_out1] = -std::numeric_limits<float>::infinity();
        return;
    }

    // read row of Q, we know tx is large enough, no inner loop needed
    float q_ix = 0.0;
    if (tx < D) {
        q_ix = q[row * stride_q0 + tx];
    }

    // read row of K ("column").
    float k_jx = 0.0;
    if (tx < D) {
        k_jx = k[col * stride_k0 + tx];
    }

    // dot product
    float o_ij = q_ix * k_jx;
    // reduce in warps
    for (int offset = 1; offset <= 16; offset *= 2) {
        o_ij += __shfl_down_sync(0xFFFFFFFF, o_ij, offset);
    }
    // reduce in block
    // all 4 warps share their acc in smem
    // warp 0 idx 0..3 load and warp reduce again
    const bool isLane0 = tx % 32 == 0;
    if (isLane0) {
        warpSums[tx / 32] = o_ij;
    }
    __syncthreads();
    if (tx < 4) {
        o_ij = warpSums[tx];
        o_ij += __shfl_down_sync(0x0000000F, o_ij, 1);
        o_ij += __shfl_down_sync(0x0000000F, o_ij, 2);
    }

    // tx0 stores o_ij
    if (tx == 0) {
        out[row * stride_out0 + col * stride_out1] = o_ij;
    }
}

torch::Tensor causal_qk(torch::Tensor q, torch::Tensor k) {
    const int N = q.size(0);
    const int D = q.size(1);
    const dim3 grid = dim3(N, N);
    const int nThreads = 128;
    const auto out = torch::empty({N, N}, q.options());
    causal_qk_kernel<<<grid, nThreads>>>(q.data_ptr<float>(), k.data_ptr<float>(), out.data_ptr<float>(),
                                         N, D, q.stride(0), q.stride(1), k.stride(0), k.stride(1),
                                         out.stride(0), out.stride(1));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("causal_qk", &causal_qk, "causal QK^T (CUDA)");
}
