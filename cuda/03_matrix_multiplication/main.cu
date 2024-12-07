#define ARMA_ALLOW_FAKE_GCC
#include <stdio.h>
#include <armadillo>
#include <cmath>
#include <chrono>

__host__ __device__ int idx(int i, int j, int rowSize) {
    return i * rowSize + j;
}

__host__ __device__ int idx_col(int i, int j, int colSize) {
    return i + j * colSize;
}

__global__ void matrix_mult(double *A, double *B, double *C, int m, int n, int p) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < m && j < p) {
        double sum = 0;
        for (int k = 0; k < n; k++) {
            sum += A[idx(i, k, n)] * B[idx(k, j, p)];
        }
        C[idx(i, j, p)] = sum;
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
            A_tile[ty][tx] = A[idx(row, tile * TILE_SIZE + tx, n)];
        } else {
            A_tile[ty][tx] = 0.0;
        }
        
        if ((tile * TILE_SIZE + ty) < n && col < p) {
            B_tile[ty][tx] = B[idx(tile * TILE_SIZE + ty, col, p)];
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
        C[idx(row, col, p)] = sum;
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

// Helper function to transpose matrix from column-major to row-major
void transpose_and_copy(double* dest, const double* src, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            dest[idx(i, j, cols)] = src[idx_col(i, j, rows)];
        }
    }
}

int main() {
    const int NUM_RUNS = 5;  // Number of times to run each method
    int m = 1024;
    int n = 1024;
    int p = 1024;
    
    printf("Matrix dimensions: %dx%d * %dx%d = %dx%d\n", m, n, n, p, m, p);
    printf("Running each method %d times...\n\n", NUM_RUNS);
    
    // Timing variables
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_milliseconds = 0;
    
    // Initialize matrices
    arma::mat A(m, n, arma::fill::randu);
    arma::mat B(n, p, arma::fill::randu);

    // Armadillo multiplication timing
    double arma_total = 0;
    arma::mat C_arma(m, p);
    for (int run = 0; run < NUM_RUNS; run++) {
        auto cpu_start = std::chrono::high_resolution_clock::now();
        C_arma = A * B;
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        arma_total += duration.count() / 1000.0;
    }
    printf("Armadillo average time: %.3f ms\n\n", arma_total / NUM_RUNS);

    // CPU multiplication timing
    double cpu_total = 0;
    arma::mat C_cpu(m, p);
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_matrix_mult(A.memptr(), B.memptr(), C_cpu.memptr(), m, n, p);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
    cpu_total += duration.count() / 1000.0;
    printf("CPU average time: %.3f ms\n\n", cpu_total);

    // CUDA setup
    double *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, m * n * sizeof(double));
    cudaMalloc((void**)&B_d, n * p * sizeof(double));
    cudaMalloc((void**)&C_d, m * p * sizeof(double));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (p + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Allocate host memory for row-major matrices
    double *A_row = new double[m * n];
    double *B_row = new double[n * p];
    double *C_row = new double[m * p];
    
    // Convert matrices from column-major (Armadillo) to row-major
    transpose_and_copy(A_row, A.memptr(), m, n);
    transpose_and_copy(B_row, B.memptr(), n, p);

    // Memory transfer timing
    cudaEventRecord(start);
    cudaMemcpy(A_d, A_row, m * n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_row, n * p * sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    float h2d_transfer_time = gpu_milliseconds;

    // Warmup run for GPU
    matrix_mult<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, m, n, p);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error calling matrix_mult(): %s\n", cudaGetErrorString(error));
        exit(1);
    }

    // Regular CUDA multiplication timing
    float cuda_total = 0;
    arma::mat C_cuda(m, p);
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);
        matrix_mult<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, m, n, p);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_milliseconds, start, stop);
        cuda_total += gpu_milliseconds;
    }
    
    // Time device to host transfer
    cudaEventRecord(start);
    cudaMemcpy(C_cuda.memptr(), C_d, m * p * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    float d2h_transfer_time = gpu_milliseconds;

    // Warmup run for tiled GPU
    matrix_mult_tiled<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, m, n, p);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error calling matrix_mult_tiled(): %s\n", cudaGetErrorString(error));
        exit(1);
    }

    // Tiled CUDA multiplication timing
    float cuda_tiled_total = 0;
    arma::mat C_cuda_tiled(m, p);
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);
        matrix_mult_tiled<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, m, n, p);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_milliseconds, start, stop);
        cuda_tiled_total += gpu_milliseconds;
    }
    
    // Time device to host transfer for tiled version
    cudaEventRecord(start);
    cudaMemcpy(C_cuda_tiled.memptr(), C_d, m * p * sizeof(double), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    float d2h_transfer_time_tiled = gpu_milliseconds;

    // Print timing results in a table
    printf("\nPerformance Results:\n");
    printf("----------------------------------------------------------------------\n");
    printf("| Method      | Computation | H2D Transfer | D2H Transfer | Total    |\n");
    printf("|-------------|-------------|--------------|--------------|----------|\n");
    printf("| CUDA        | %11.3f | %12.3f | %12.3f | %8.3f |\n",
           cuda_total / NUM_RUNS,
           h2d_transfer_time,
           d2h_transfer_time,
           (cuda_total / NUM_RUNS) + h2d_transfer_time + d2h_transfer_time);
    printf("| CUDA Tiled  | %11.3f | %12.3f | %12.3f | %8.3f |\n",
           cuda_tiled_total / NUM_RUNS,
           h2d_transfer_time,
           d2h_transfer_time_tiled,
           (cuda_tiled_total / NUM_RUNS) + h2d_transfer_time + d2h_transfer_time_tiled);
    printf("----------------------------------------------------------------------\n");
    printf("All times in milliseconds (ms)\n\n");

    // CPU results for comparison
    printf("CPU Results:\n");
    printf("Armadillo: %.3f ms\n", arma_total / NUM_RUNS);
    printf("Basic CPU: %.3f ms\n\n", cpu_total);

    // Compare results
    double tolerance = 1e-6;
    bool cpu_match = compare_matrices(C_arma.memptr(), C_cpu.memptr(), m, p, tolerance);

    transpose_and_copy(C_row, C_cuda.memptr(), m, p);
    bool cuda_match = compare_matrices(C_arma.memptr(), C_row, m, p, tolerance);

    transpose_and_copy(C_row, C_cuda_tiled.memptr(), m, p);
    bool cuda_tiled_match = compare_matrices(C_arma.memptr(), C_row, m, p, tolerance);

    printf("Verification Results:\n");
    printf("CPU result matches Armadillo: %s\n", cpu_match ? "Yes" : "No");
    printf("CUDA result matches Armadillo: %s\n", cuda_match ? "Yes" : "No");
    printf("CUDA Tiled result matches Armadillo: %s\n", cuda_tiled_match ? "Yes" : "No");

    // Cleanup
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Additional cleanup
    delete[] A_row;
    delete[] B_row;

    return 0;
}
