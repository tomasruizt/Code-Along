#include <stdio.h>
#include <stdlib.h>
#include "matmul.cuh"

void init_random_matrix(float* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    const int n = 1024;  // Single dimension for square matrices
    
    printf("Matrix dimensions: %d\n", n);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float gpu_milliseconds = 0;
    
    // Allocate and initialize host matrices
    float *A = new float[n * n];
    float *B = new float[n * n];
    float *C = new float[n * n];
    
    init_random_matrix(A, n);
    init_random_matrix(B, n);

    // CUDA setup
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, n * n * sizeof(float));
    cudaMalloc((void**)&B_d, n * n * sizeof(float));
    cudaMalloc((void**)&C_d, n * n * sizeof(float));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Copy data to device
    cudaMemcpy(A_d, A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Run basic matrix multiplication
    printf("\nRunning basic matrix multiplication...\n");
    cudaEventRecord(start);
    matrix_mult<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, n, n, n);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Basic matrix multiplication kernel failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    printf("Basic matrix multiplication time: %.2f ms\n", gpu_milliseconds);

    // Run tiled matrix multiplication
    printf("\nRunning tiled matrix multiplication...\n");
    cudaEventRecord(start);
    matrix_mult_tiled<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, n, n, n);
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Tiled matrix multiplication kernel failed: %s\n", cudaGetErrorString(error));
        return 1;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_milliseconds, start, stop);
    printf("Tiled matrix multiplication time: %.2f ms\n", gpu_milliseconds);

    // Cleanup
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
} 