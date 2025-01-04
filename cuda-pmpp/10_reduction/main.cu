#include <stdio.h>

// Define a type for kernel sum functions
typedef void (*SumFn)(float* vec_d, int len, float* sum_d);

__global__ void cudaNaiveSumKernel(float* vec, int len, float* result) {
    unsigned int tx = threadIdx.x;  // Apparently the unsigned int is required
    // The particular stride creates uncoalesced memory access
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (tx % stride == 0)
            vec[2 * tx] += vec[2 * tx + stride];
        __syncthreads();
    }
    if (tx == 0)
        result[0] = vec[0];
}

void cudaNaiveSum(float* vec_d, int len, float* sum_d) {
    dim3 threadsPerBlock(len / 2);
    dim3 numBlocks(1);
    cudaNaiveSumKernel<<<numBlocks, threadsPerBlock>>>(vec_d, len, sum_d);
}

__global__ void cudaContinguousSumKernel(float* vec, int len, float* result) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        if (i < stride) {
            vec[i] += vec[i + stride];
        }
        __syncthreads();
    }
    if (i == 0)
        result[0] = vec[0];
}

void cudaContinguousSum(float* vec_d, int len, float* sum_d) {
    dim3 threadsPerBlock(len / 2);
    dim3 numBlocks(1);
    cudaContinguousSumKernel<<<numBlocks, threadsPerBlock>>>(vec_d, len, sum_d);
}

float cudaSum(float *vec, int len, SumFn sum)
{
    float *sum_h = new float[1];
    float *sum_d, *vec_d;
    cudaMalloc((void **)&sum_d, sizeof(float));
    cudaMalloc((void **)&vec_d, len * sizeof(float));
    cudaMemcpy(vec_d, vec, len * sizeof(float), cudaMemcpyHostToDevice);
    
    sum(vec_d, len, sum_d);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    cudaMemcpy(sum_h, sum_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(sum_d);
    cudaFree(vec_d);
    return sum_h[0];
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

int main() {
    int n = 1024 * 2;  // cudaNaiveSum and cudaContiguous only work on a single block atm
    float* vec = arange(n);
    float sum = cpu_sum(vec, n);
    printf("CPU sum: %.2f\n", sum);

    // Use the naive kernel
    float sum_h = cudaSum(vec, n, cudaNaiveSum);
    printf("CUDA naive sum: %.2f\n", sum_h);

    float sum_h2 = cudaSum(vec, n, cudaContinguousSum);
    printf("CUDA contiguous sum: %.2f\n", sum_h2);
    
    return 0;
}