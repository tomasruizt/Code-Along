#include <torch/extension.h>
#include <cuda_runtime.h>

constexpr int TILE = 32;

__global__ void transpose_naive_kernel(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int M,
                                       int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        out[col * M + row] = in[row * N + col];
    }
}

__global__ void transpose_tiled_kernel(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       int M,
                                       int N) {
    __shared__ float tile[TILE][TILE + 1];

    // Coalesced load: thread (tx, ty) reads in[(by*TILE + ty), (bx*TILE + tx)].
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    }

    __syncthreads();

    // Coalesced store: write the transposed tile to (blockIdx.y * TILE, blockIdx.x * TILE).
    // Output is N x M, so index is y_out * M + x_out.
    int x_out = blockIdx.y * TILE + threadIdx.x;
    int y_out = blockIdx.x * TILE + threadIdx.y;
    if (x_out < M && y_out < N) {
        out[y_out * M + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

torch::Tensor transpose_naive(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous());
    TORCH_CHECK(x.dim() == 2);
    int M = x.size(0), N = x.size(1);
    auto out = torch::empty({N, M}, x.options());
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    transpose_naive_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}

torch::Tensor transpose_tiled(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kFloat32 && x.is_contiguous());
    TORCH_CHECK(x.dim() == 2);
    int M = x.size(0), N = x.size(1);
    auto out = torch::empty({N, M}, x.options());
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    transpose_tiled_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), M, N);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_naive", &transpose_naive, "transpose_naive (CUDA)");
    m.def("transpose_tiled", &transpose_tiled, "transpose_tiled (CUDA)");
}
