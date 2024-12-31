#include "matmul.cuh"

// Function implementations
__host__ __device__ int idx(int i, int j, int rowSize) {
    return i * rowSize + j;
}

__host__ __device__ int idx_col(int i, int j, int colSize) {
    return i + j * colSize;
}

__global__ void matrix_mult(float *A, float *B, float *C, int m, int n, int p) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < p) {
        float sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[idx(i, k, n)] * B[idx(k, j, p)];
        }
        C[idx(i, j, p)] = sum;
    }
}

__global__ void matrix_mult_tiled(float *A, float *B, float *C, int m, int n, int p) {
    __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        if (row < m && (tile * TILE_SIZE + tx) < n) {
            A_tile[ty][tx] = A[idx(row, tile * TILE_SIZE + tx, n)];
        } else {
            A_tile[ty][tx] = 0.0f;
        }
        
        if ((tile * TILE_SIZE + ty) < n && col < p) {
            B_tile[ty][tx] = B[idx(tile * TILE_SIZE + ty, col, p)];
        } else {
            B_tile[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < m && col < p) {
        C[idx(row, col, p)] = sum;
    }
}

__global__ void tiledMatMulWithoutBoundsCheck(float *A, float *B, float *C, int n) {
    __shared__ float A_block[TILE_SIZE][TILE_SIZE];
    __shared__ float B_block[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float C_ij = 0.0f;
    
    for (int t = 0; t < n / TILE_SIZE; t++) {
        A_block[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        B_block[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            C_ij += A_block[threadIdx.y][k] * B_block[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    C[row * n + col] = C_ij;
}

void cpu_matrix_mult(float *A, float *B, float *C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[idx_col(i, k, n)] * B[idx_col(k, j, p)];
            }
            C[idx_col(i, j, p)] = sum;
        }
    }
}

bool compare_matrices(float *A, float *B, int m, int n, float tolerance) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (std::abs(A[idx(i, j, n)] - B[idx(i, j, n)]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

void transpose_and_copy(float* dest, const float* src, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            dest[idx(i, j, cols)] = src[idx_col(i, j, rows)];
        }
    }
} 