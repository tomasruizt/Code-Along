#include <stdio.h>

struct Sum {
    __device__ float operator()(float a, float b) { 
        return a + b; 
    }

    static __device__ void atomic_reduce(float* addr, float val) { 
        atomicAdd(addr, val); 
    }
};

struct Max {
    __device__ float operator()(float a, float b) { 
        return max(a, b); 
    }

    static __device__ void atomic_reduce(float* addr, float val) {
        // Like in https://stackoverflow.com/a/17401122/5730291
        unsigned int* address_as_uint = (unsigned int*)addr;
        unsigned int old = *address_as_uint;
        unsigned int assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_uint, assumed,
                __float_as_int(max(__int_as_float(assumed), val)));
        } while (assumed != old);
    }
};

template<typename ReduceOp>
using ReduceKernelFn = void (*)(float*, int, float*);

template<typename ReduceOp>
__global__ void cudaNaiveKernel(float* vec, int len, float* result) {
    unsigned int tx = threadIdx.x;  // Apparently the unsigned int is required
    unsigned int i = 2 * tx + (2 * blockIdx.x * blockDim.x);
    ReduceOp reduce;
    
    // The particular stride creates uncoalesced memory access
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (tx % stride == 0 && i + stride < len)
            vec[i] = reduce(vec[i], vec[i + stride]);
        __syncthreads();
    }
    if (tx == 0)
        atomicAdd(result, vec[i]);
}

template<typename ReduceOp>
__global__ void cudaContinguousKernel(float* vec, int len, float* result) {
    unsigned int tx = threadIdx.x;
    unsigned int i = tx + (2 * blockIdx.x * blockDim.x);
    ReduceOp reduce;

    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        if (tx < stride && i + stride < len) {
            vec[i] = reduce(vec[i], vec[i + stride]);
        }
        __syncthreads();
    }
    if (tx == 0)
        atomicAdd(result, vec[i]);
}


const int BLOCK_SIZE = 256;  // to fit 3 per SM

struct rtx3090 {
    int threads_per_sm = 1536;
    int blocks_per_sm = threads_per_sm / BLOCK_SIZE;
    int numSM = 82;
};

template<typename ReduceOp>
__global__ void cudaSharedMemKernel(float* vec, int len, float* result) {
    __shared__ float vec_s[BLOCK_SIZE];
    unsigned int tx = threadIdx.x;
    unsigned int i = tx + (2 * blockIdx.x * blockDim.x);
    ReduceOp reduce;

    // populate shared memory
    vec_s[tx] = reduce(vec[i], vec[i + blockDim.x]);
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        __syncthreads();
        if (tx < stride && i + stride < len) {
            vec_s[tx] = reduce(vec_s[tx], vec_s[tx + stride]);
        }
    }
    if (tx == 0)
        ReduceOp::atomic_reduce(result, vec_s[tx]);
}

template<typename ReduceOp>
float cudaReduce(float *vec, int len, ReduceKernelFn<ReduceOp> reduce)
{
    float *sum_h = new float[1];
    float *sum_d, *vec_d;
    cudaMalloc((void **)&sum_d, sizeof(float));
    cudaMalloc((void **)&vec_d, len * sizeof(float));
    cudaMemcpy(vec_d, vec, len * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks(ceil(len / (2.0 * threadsPerBlock.x)));
    // printf("numBlocks: %d\n", numBlocks.x);
    // printf("threadsPerBlock: %d\n", threadsPerBlock.x);
    reduce<<<numBlocks, threadsPerBlock>>>(vec_d, len, sum_d);
    
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

float cpu_max(float* vec, int len) {
    float max = -INFINITY;
    for (int i = 0; i < len; i++) {
        max = std::max(max, vec[i]);
    }
    return max;
}

float* randu(int n) {
    srand(0);
    float* vec = new float[n];
    for (int i = 0; i < n; i++) {
        vec[i] = rand() / (float)RAND_MAX;
    }
    return vec;
}

int main() {
    auto gpu = rtx3090();
    int nBlocks = gpu.blocks_per_sm * gpu.numSM;
    int nThreads = gpu.threads_per_sm * gpu.numSM;
    printf("nBlocks: %d\n", nBlocks);
    printf("nThreads: %d\n", nThreads);
    int n = nThreads * 2;
    
    printf("vec size (n): %d\n", n);
    float* vec = randu(n);
    float sum = cpu_sum(vec, n);
    printf("CPU sum: %.2f\n", sum);

    // Use the naive kernel
    float sum_h = cudaReduce<Sum>(vec, n, cudaNaiveKernel<Sum>);
    printf("CUDA naive sum: %.2f\n", sum_h);

    float sum_h2 = cudaReduce<Sum>(vec, n, cudaContinguousKernel<Sum>);
    printf("CUDA contiguous sum: %.2f\n", sum_h2);

    float sum_h3 = cudaReduce<Sum>(vec, n, cudaSharedMemKernel<Sum>);
    printf("CUDA sharedmem sum: %.2f\n", sum_h3);

    float max = cpu_max(vec, n);
    printf("CPU max: %.6f\n", max);

    float max_h = cudaReduce<Max>(vec, n, cudaSharedMemKernel<Max>);
    printf("CUDA sharedmem max: %.6f\n", max_h);

    return 0;
}