/// WORK IN PROGRESS

#include <stdio.h>
#include <math.h>

__global__ void naiveSoftmax(float* x, float* s, int n) {
    int i = threadIdx.x;
    s[i] = exp(x[i]);
}

int main() {
    float x[] = {1, 2, 3, 4, 5};
    int n = sizeof(x) / sizeof(x[0]);
    printf("n = %d\n", n);
    float s[n];
    float* d_x;
    float* d_s;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_s, n * sizeof(float));
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    int numBlocks = 1;
    int numThreads = n;
    naiveSoftmax<<<numBlocks, numThreads>>>(d_x, d_s, n);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    cudaMemcpy(s, d_s, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_s);
    for (int i = 0; i < n; i++) {
        printf("%f ", s[i]);
    }
    printf("\n");
    return 0;
}
