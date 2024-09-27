#include <stdio.h>
#include <cuda.h>
#include "operations.h"

int main() {
    int n = 5;
    int a[] = {1, 2, 3, 4, 5};
    int b[] = {1, 2, 3, 4, 5};
    int c[n];
    mult(a, b, c, n);

    // Device pointers
    int *d_a, *d_b, *d_c;

    // Allocate memory on the device
    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, n * sizeof(int));
    cudaMalloc((void**)&d_c, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_a, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(int), cudaMemcpyHostToDevice);

    printf("a = ");
    print_vector(a, n);

    printf("b = ");
    print_vector(b, n);

    printf("c = ");
    print_vector(c, n);
    printf("\n");
    return 0;
}