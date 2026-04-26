#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int BM = 32;
constexpr int BN = 32;
constexpr int BK = 32;

__global__ void gemm_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M,
                            int N,
                            int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    int tx = threadIdx.x;  // column within the tile (also column within Bs / output col)
    int ty = threadIdx.y;  // row    within the tile (also row    within As / output row)

    int row = blockIdx.y * BM + ty;
    int col = blockIdx.x * BN + tx;

    float acc = 0.0f;

    for (int kt = 0; kt < K; kt += BK) {
        // Load A[row, kt + tx] into As[ty][tx].
        int a_col = kt + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        // Load B[kt + ty, col] into Bs[ty][tx].
        int b_row = kt + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor gemm(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda());
    TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32);
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous());
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2);
    int M = a.size(0), K = a.size(1);
    int Kb = b.size(0), N = b.size(1);
    TORCH_CHECK(K == Kb, "A.cols must equal B.rows");

    auto c = torch::empty({M, N}, a.options());
    dim3 threads(BN, BM);
    dim3 blocks((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N,
                                     K);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "tiled gemm (CUDA)");
}
