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

#define TILE_SIZE 16

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

// Without bounds checking for blog post
__global__ void tiledMatMulWithoutBoundsCheck(float *A, float *B, float *C, int n) {
    // Define shared memory arrays (cache)
    __shared__ float A_block[TILE_SIZE][TILE_SIZE];
    __shared__ float B_block[TILE_SIZE][TILE_SIZE];
    
    // CUDA thread variables
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float C_ij = 0.0f;
    
    // Loop over the blocks
    for (int t = 0; t < n / TILE_SIZE; t++) {
        // Transfer data from main memory into the cache
        A_block[threadIdx.y][threadIdx.x] = A[row * n + t * TILE_SIZE + threadIdx.x];
        B_block[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * n + col];
        
        // Ensure data transfer is complete before proceeding
        __syncthreads();
        
        // Matrix multiply both blocks
        for (int k = 0; k < TILE_SIZE; k++) {
            C_ij += A_block[threadIdx.y][k] * B_block[k][threadIdx.x];
        }
        
        // Finish multiplying the blocks before overwriting the cache next iteration
        __syncthreads();
    }
    
    // Transfer the result back to global memory
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

int main() {
    const int NUM_RUNS = 5;
    const int m = 1024*1;
    const int n = m;
    const int p = m;
    
    printf("Matrix dimensions: %dx%d * %dx%d = %dx%d\n", m, n, n, p, m, p);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_milliseconds = 0;
    
    // Initialize matrices with single precision
    arma::fmat A(m, n, arma::fill::randu);
    arma::fmat B(n, p, arma::fill::randu);
    arma::fmat C_arma(m, p);

    const bool run_cpu_baseline = true;
    if (run_cpu_baseline) {
        printf("Armadillo multiplication timing (single run)...\n");
        auto cpu_start = std::chrono::high_resolution_clock::now();
        C_arma = A * B;
        auto cpu_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start);
        printf("Armadillo time: %.2f seconds\n\n", duration.count() / 1e6);
    }

    // CUDA setup
    printf("Running each CUDA method %d times...", NUM_RUNS);
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, m * n * sizeof(float));
    cudaMalloc((void**)&B_d, n * p * sizeof(float));
    cudaMalloc((void**)&C_d, m * p * sizeof(float));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (p + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Allocate host memory for row-major matrices
    float *A_row = new float[m * n];
    float *B_row = new float[n * p];
    float *C_row = new float[m * p];
    
    transpose_and_copy(A_row, A.memptr(), m, n);
    transpose_and_copy(B_row, B.memptr(), n, p);

    cudaEventRecord(start);
    cudaMemcpy(A_d, A_row, m * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_row, n * p * sizeof(float), cudaMemcpyHostToDevice);
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
    arma::fmat C_cuda(m, p);
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);
        matrix_mult<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, m, n, p);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_milliseconds, start, stop);
        cuda_total += gpu_milliseconds;
    }
    
    cudaEventRecord(start);
    cudaMemcpy(C_cuda.memptr(), C_d, m * p * sizeof(float), cudaMemcpyDeviceToHost);
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
    arma::fmat C_cuda_tiled(m, p);
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);
        matrix_mult_tiled<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, m, n, p);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_milliseconds, start, stop);
        cuda_tiled_total += gpu_milliseconds;
    }
    
    cudaEventRecord(start);
    cudaMemcpy(C_cuda_tiled.memptr(), C_d, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    float d2h_transfer_time_tiled = gpu_milliseconds;

    // Tiled CUDA multiplication without bounds checking
    float tiled_wo_bc_total = 0;
    arma::fmat C_cuda_tiled_wo_bc(m, p);
    for (int run = 0; run < NUM_RUNS; run++) {
        cudaEventRecord(start);
        tiledMatMulWithoutBoundsCheck<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpu_milliseconds, start, stop);
        tiled_wo_bc_total += gpu_milliseconds;
    }
    
    cudaEventRecord(start);
    cudaMemcpy(C_cuda_tiled_wo_bc.memptr(), C_d, m * p * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    float d2h_transfer_time_simple = gpu_milliseconds;

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
    printf("| Tiled wo BC | %11.3f | %12.3f | %12.3f | %8.3f |\n",
           tiled_wo_bc_total / NUM_RUNS,
           h2d_transfer_time,
           d2h_transfer_time_simple,
           (tiled_wo_bc_total / NUM_RUNS) + h2d_transfer_time + d2h_transfer_time_simple);
    printf("----------------------------------------------------------------------\n");
    printf("All times in milliseconds (ms)\n\n");

    float tolerance = 1e-3f;
    if (run_cpu_baseline) {
        transpose_and_copy(C_row, C_cuda_tiled_wo_bc.memptr(), m, p);
        bool cuda_simple_match = compare_matrices(C_arma.memptr(), C_row, m, p, tolerance);

        transpose_and_copy(C_row, C_cuda.memptr(), m, p);
        bool cuda_match = compare_matrices(C_arma.memptr(), C_row, m, p, tolerance);

        transpose_and_copy(C_row, C_cuda_tiled.memptr(), m, p);
        bool cuda_tiled_match = compare_matrices(C_arma.memptr(), C_row, m, p, tolerance);

        printf("Verification Results:\n");
        printf("Tiled wo BC result matches Armadillo: %s\n", cuda_simple_match ? "Yes" : "No");
        printf("CUDA result matches Armadillo: %s\n", cuda_match ? "Yes" : "No");
        printf("CUDA Tiled result matches Armadillo: %s\n", cuda_tiled_match ? "Yes" : "No");
    }

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] A_row;
    delete[] B_row;
    delete[] C_row;

    return 0;
}
