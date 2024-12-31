#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#include "matmul.cuh"

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
