#include <stdio.h>

// Define a type for kernel sum functions
typedef void (*SumFn)(float* vec_d, int len, float* sum_d);

__global__ void cudaNaiveSumKernel(float* vec, int len, float* result) {
    unsigned int tx = threadIdx.x;  // Apparently the unsigned int is required
    unsigned int i = 2 * tx + (2 * blockIdx.x * blockDim.x);
    // The particular stride creates uncoalesced memory access
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (tx % stride == 0 && i + stride < len)
            vec[i] += vec[i + stride];
        __syncthreads();
    }
    if (tx == 0)
        atomicAdd(result, vec[i]);
}

__global__ void cudaContinguousSumKernel(float* vec, int len, float* result) {
    unsigned int tx = threadIdx.x;
    unsigned int i = tx + (2 * blockIdx.x * blockDim.x);
    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        if (tx < stride && i + stride < len) {
            vec[i] += vec[i + stride];
        }
        __syncthreads();
    }
    if (tx == 0)
        atomicAdd(result, vec[i]);
}


float cudaSum(float *vec, int len, SumFn sum)
{
    float *sum_h = new float[1];
    float *sum_d, *vec_d;
    cudaMalloc((void **)&sum_d, sizeof(float));
    cudaMalloc((void **)&vec_d, len * sizeof(float));
    cudaMemcpy(vec_d, vec, len * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(1024);
    dim3 numBlocks(ceil(len / (2.0 * threadsPerBlock.x)));
    // printf("numBlocks: %d\n", numBlocks.x);
    // printf("threadsPerBlock: %d\n", threadsPerBlock.x);
    sum<<<numBlocks, threadsPerBlock>>>(vec_d, len, sum_d);
    
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

float* ones(int n)
{
    float* vec = new float[n];
    for (int i = 0; i < n; i++)
    {
        vec[i] = 1.0f;
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
    int n = 1024 * 10 + 1;
    printf("n: %d\n", n);
    float* vec = ones(n);
    float sum = cpu_sum(vec, n);
    printf("CPU sum: %.2f\n", sum);

    // Use the naive kernel
    float sum_h = cudaSum(vec, n, cudaNaiveSumKernel);
    printf("CUDA naive sum: %.2f\n", sum_h);

    float sum_h2 = cudaSum(vec, n, cudaContinguousSumKernel);
    printf("CUDA contiguous sum: %.2f\n", sum_h2);
    
    return 0;
}