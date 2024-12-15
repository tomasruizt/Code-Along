#include <stdio.h>
#include "main.h"

int main() {
    int n = 1024 * 2;
    float* vec = arange(n);
    float sum = cpu_sum(vec, n);
    printf("CPU sum: %.2f\n", sum);

    float* sum_h = cuda_sum(vec, n);
    printf("CUDA sum: %.2f\n", sum_h[0]);
    return 0;
}

float* cuda_sum(float *vec, int len)
{
    float *sum_h = new float[1];
    float *sum_d, *vec_d;
    cudaMalloc((void **)&sum_d, sizeof(float));
    cudaMalloc((void **)&vec_d, len * sizeof(float));
    cudaMemcpy(vec_d, vec, len * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(len / 2);
    dim3 numBlocks(1);
    cudaNaiveSum<<<numBlocks, threadsPerBlock>>>(vec_d, len, sum_d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaMemcpy(sum_h, sum_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(sum_d);
    cudaFree(vec_d);
    return sum_h;
}

float* arange(int n)
{
    float* vec = new float[n];
    for (int i = 0; i < n; i++)
    {
        vec[i] = i + 1;
    }
    return vec;
}

float cpu_sum(float* vec, int len) {
    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        sum += vec[i];
    }
    return sum;
}

__global__ void cudaNaiveSum(float* vec, int len, float* result) {
    unsigned int tx = threadIdx.x;  // Apparently the unsigned int is required
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (tx % stride == 0)
            vec[2 * tx] += vec[2 * tx + stride];
    }
    if (tx == 0)
        result[0] = vec[0];
}
