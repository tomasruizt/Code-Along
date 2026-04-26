#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void naive_kernel(const float* x, float* out, int nRows, int nCols) {
    const int i = threadIdx.y + blockDim.y * blockIdx.y;
    const int j = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < nRows && j < nCols) {
        // O[j,i] = X[i,j]
        const int x_idx = j + i * nCols;
        const int o_idx = i + j * nRows;
        out[o_idx] = x[x_idx];
    }
}

int cdiv(int n, int d) {
    return (n + d - 1) / d;
}

torch::Tensor transpose_naive(torch::Tensor x) {
    const int nRows = x.size(0);
    const int nCols = x.size(1);
    const auto out = torch::full({nCols, nRows}, -999, x.options());

    const int n = x.numel();
    const int blockSize = 16;
    const int xBlocks = cdiv(nCols, blockSize);
    const int yBlocks = cdiv(nRows, blockSize);
    const dim3 grid = dim3(xBlocks, yBlocks);
    naive_kernel<<<grid, dim3(blockSize, blockSize)>>>(x.data_ptr<float>(), out.data_ptr<float>(), nRows,
                                                       nCols);
    return out;
}

const int blockSizeX = 32;  // 32*32=1024
const int blockSizeY = blockSizeX;

__global__ void smem_kernel(const float* x, float* out, int nRows, int nCols) {
    __shared__ float x_tile[blockSizeY][blockSizeX + 1];

    const int xi = threadIdx.y + blockDim.y * blockIdx.y;
    const int xj = threadIdx.x + blockDim.x * blockIdx.x;
    if (xi < nRows && xj < nCols) {
        // O[j,i] = X[i,j]
        const int x_idx = xj + xi * nCols;
        x_tile[threadIdx.x][threadIdx.y] = x[x_idx];
    }
    __syncthreads();

    const int oi_base = xj - threadIdx.x;
    const int oj_base = xi - threadIdx.y;
    const int oj = oj_base + threadIdx.x;
    const int oi = oi_base + threadIdx.y;

    if (oi < nCols && oj < nRows) {
        const int o_idx = oj + oi * nRows;
        out[o_idx] = x_tile[threadIdx.y][threadIdx.x];
    }
}

torch::Tensor transpose_tiled(torch::Tensor x) {
    const int nRows = x.size(0);
    const int nCols = x.size(1);
    const auto out = torch::full({nCols, nRows}, -999, x.options());

    const int n = x.numel();
    const int xBlocks = cdiv(nCols, blockSizeX);
    const int yBlocks = cdiv(nRows, blockSizeY);
    const dim3 grid = dim3(xBlocks, yBlocks);
    smem_kernel<<<grid, dim3(blockSizeX, blockSizeY)>>>(x.data_ptr<float>(), out.data_ptr<float>(),
                                                        nRows, nCols);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transpose_naive", &transpose_naive, "transpose_naive (CUDA)");
    m.def("transpose_tiled", &transpose_tiled, "transpose_tiled (CUDA)");
}
