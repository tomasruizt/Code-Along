#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BN = 32;
constexpr int BD = 32;

__global__ void causal_qk_kernel(const float* __restrict__ Q,
                                 const float* __restrict__ K,
                                 float* __restrict__ S,
                                 int N,
                                 int D) {
    __shared__ float Qs[BN][BD];
    __shared__ float Ks[BN][BD + 1];  // +1 dodges 32-way bank conflict on Ks[tx][k]

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = blockIdx.y * BN + ty;
    const int col = blockIdx.x * BN + tx;

    // Whole tile is strictly above the diagonal → all -inf, skip the dot product.
    const int row_max_in_tile = blockIdx.y * BN + BN - 1;
    const int col_min_in_tile = blockIdx.x * BN;
    if (col_min_in_tile > row_max_in_tile) {
        if (row < N && col < N) {
            S[row * N + col] = -INFINITY;
        }
        return;
    }

    float acc = 0.0f;

    for (int kt = 0; kt < D; kt += BD) {
        // Qs[ty][tx] = Q[row, kt + tx]
        const int q_d = kt + tx;
        Qs[ty][tx] = (row < N && q_d < D) ? Q[row * D + q_d] : 0.0f;

        // Ks[ty][tx] = K[blockIdx.x*BN + ty, kt + tx]   (i.e. Ks[c][d] = K[block_col_base + c, kt + d])
        const int k_row = blockIdx.x * BN + ty;
        const int k_d = kt + tx;
        Ks[ty][tx] = (k_row < N && k_d < D) ? K[k_row * D + k_d] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BD; ++k) {
            acc += Qs[ty][k] * Ks[tx][k];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        S[row * N + col] = (col > row) ? -INFINITY : acc;
    }
}

torch::Tensor causal_qk(torch::Tensor q, torch::Tensor k) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda());
    TORCH_CHECK(q.dtype() == torch::kFloat32 && k.dtype() == torch::kFloat32);
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous());
    TORCH_CHECK(q.dim() == 2 && k.dim() == 2);
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(1) == k.size(1));

    const int N = q.size(0);
    const int D = q.size(1);
    auto s = torch::empty({N, N}, q.options());

    const dim3 threads(BN, BN);
    const dim3 blocks((N + BN - 1) / BN, (N + BN - 1) / BN);
    causal_qk_kernel<<<blocks, threads>>>(q.data_ptr<float>(), k.data_ptr<float>(),
                                          s.data_ptr<float>(), N, D);
    return s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("causal_qk", &causal_qk, "causal QK^T (CUDA)");
}
