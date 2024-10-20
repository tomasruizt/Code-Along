#include <stdio.h>

__global__ void plus(int* a, int* b, int* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}

void print_vector(int *a, int n) {
    printf("[ ");
    for (int i = 0; i < n; i++) {
        printf("%d ", a[i]);
    }
    printf("]");
    printf("\n");
}