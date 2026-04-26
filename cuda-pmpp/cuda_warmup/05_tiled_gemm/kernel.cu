#include <torch/extension.h>
#include <cuda_runtime.h>

const int M_BLOCK = 16;
const int N_BLOCK = 16;
const int K_BLOCK = 16;

int cdiv(int n, int d) {
    return (n + d - 1) / d;
}

__global__ void kernel(const float* a, const float* b, float* c, int M, int N, int K) {
    __shared__ float a_tile[M_BLOCK][K_BLOCK];
    __shared__ float b_tile[K_BLOCK][N_BLOCK];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int a_y = ty + blockDim.y * blockIdx.y;
    int b_x = tx + blockDim.x * blockIdx.x;

    float acc = 0.0f;
    for (int k = 0; k < K; k += K_BLOCK) {
        int a_x = k + tx;
        int b_y = k + ty;

        // collaborate to load a_tile, b_tile
        float a_val = 0.0f;
        if (a_x < K && a_y < M) {
            a_val = a[a_y * K + a_x];
        }
        a_tile[ty][tx] = a_val;

        float b_val = 0.0f;
        if (b_x < N && b_y < K) {
            b_val = b[b_y * N + b_x];
        }
        b_tile[ty][tx] = b_val;
        // before consuming smem, wait on all warps
        __syncthreads();

        // compute dot product (each thread in threadblock)
        // together, all make a matmul
        for (int j = 0; j < K_BLOCK; j++) {
            acc += a_tile[ty][j] * b_tile[j][tx];
        }

        // sync again to prevent overwriting smem on next iter
        __syncthreads();
    }

    // store
    if (a_y < M && b_x < N) {
        c[a_y * N + b_x] = acc;
    }
}

torch::Tensor gemm(torch::Tensor a, torch::Tensor b) {
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);
    const auto c = torch::empty({M, N}, a.options());
    const dim3 grid = dim3(cdiv(N, N_BLOCK), cdiv(M, M_BLOCK));
    const dim3 bSizes = dim3(N_BLOCK, M_BLOCK);
    kernel<<<grid, bSizes>>>(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), M, N, K);
    return c;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm, "tiled gemm (CUDA)");
}
