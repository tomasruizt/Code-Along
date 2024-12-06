#define ARMA_ALLOW_FAKE_GCC
#include <stdio.h>
#include <armadillo>
#include <cmath>

__host__ __device__ int idx(int i, int j, int rowSize) {
    return i * rowSize + j;
}

__host__ __device__ int idx_col(int i, int j, int colSize) {
    return i + j * colSize;
}

void print_matrix(int m, int n, double *A, bool rowMajor = true) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (rowMajor) {
                printf("%.6f ", A[idx(i, j, n)]);
            } else {
                printf("%.6f ", A[idx_col(i, j, m)]);
            }
        }
        printf("\n");
    }
}

__global__ void matrix_mult(double *A, double *B, double *C, int m, int n, int p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < p) {
        double sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[idx_col(i, k, n)] * B[idx_col(k, j, p)];
        }
        C[idx_col(i, j, p)] = sum;
    }
}

// Add this constant for tile size
#define TILE_SIZE 16

__global__ void matrix_mult_tiled(double *A, double *B, double *C, int m, int n, int p) {
    __shared__ double A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ double B_tile[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    double sum = 0.0;
    
    // Loop over tiles
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tiles into shared memory
        if (row < m && (tile * TILE_SIZE + tx) < n) {
            A_tile[ty][tx] = A[idx_col(row, tile * TILE_SIZE + tx, n)];
        } else {
            A_tile[ty][tx] = 0.0;
        }
        
        if ((tile * TILE_SIZE + ty) < n && col < p) {
            B_tile[ty][tx] = B[idx_col(tile * TILE_SIZE + ty, col, p)];
        } else {
            B_tile[ty][tx] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[ty][k] * B_tile[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < m && col < p) {
        C[idx_col(row, col, p)] = sum;
    }
}

void cpu_matrix_mult(double *A, double *B, double *C, int m, int n, int p) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[idx_col(i, k, n)] * B[idx_col(k, j, p)];
            }
            C[idx_col(i, j, p)] = sum;
        }
    }
}

bool compare_matrices(double *A, double *B, int m, int n, double tolerance) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (std::abs(A[idx(i, j, n)] - B[idx(i, j, n)]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    int m = 2;
    int n = 2;
    int p = 2;
    // Attention! Armadillo stores matrices in column-major order
    arma::mat A(m, n, arma::fill::randu);
    arma::mat B = {{1, 2}, {3, 4}}; //(n, p, arma::fill::randu);

    A.print("Matrix A:");
    printf("\n");

    B.print("Matrix B:");
    printf("\n");
    
    // Armadillo multiplication
    arma::mat C_arma = A * B;
    C_arma.print("Armadillo result C:");
    printf("\n");

    // CPU multiplication
    arma::mat C_cpu(m, p);
    cpu_matrix_mult(A.memptr(), B.memptr(), C_cpu.memptr(), m, n, p);
    printf("CPU result:\n");
    print_matrix(m, p, C_cpu.memptr(), false);
    printf("\n");

    // CUDA multiplication
    double *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, m * n * sizeof(double));
    cudaMalloc((void**)&B_d, n * p * sizeof(double));
    cudaMalloc((void**)&C_d, m * p * sizeof(double));

    cudaMemcpy(A_d, A.memptr(), m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B.memptr(), n * p * sizeof(double), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, (p + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_mult<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, m, n, p);
    
    arma::mat C_cuda(m, p);
    cudaMemcpy(C_cuda.memptr(), C_d, m * p * sizeof(double), cudaMemcpyDeviceToHost);

    printf("CUDA result:\n");
    print_matrix(m, p, C_cuda.memptr(), false);
    printf("\n");

    // Tiled CUDA multiplication
    arma::mat C_cuda_tiled(m, p);
    matrix_mult_tiled<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, m, n, p);
    
    cudaMemcpy(C_cuda_tiled.memptr(), C_d, m * p * sizeof(double), cudaMemcpyDeviceToHost);

    printf("CUDA Tiled result:\n");
    print_matrix(m, p, C_cuda_tiled.memptr(), false);
    printf("\n");

    // Compare results
    double tolerance = 1e-6;
    bool cpu_match = compare_matrices(C_arma.memptr(), C_cpu.memptr(), m, p, tolerance);
    bool cuda_match = compare_matrices(C_arma.memptr(), C_cuda.memptr(), m, p, tolerance);
    bool cuda_tiled_match = compare_matrices(C_arma.memptr(), C_cuda_tiled.memptr(), m, p, tolerance);

    printf("CPU result matches Armadillo: %s\n", cpu_match ? "Yes" : "No");
    printf("CUDA result matches Armadillo: %s\n", cuda_match ? "Yes" : "No");
    printf("CUDA Tiled result matches Armadillo: %s\n", cuda_tiled_match ? "Yes" : "No");

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    return 0;
}