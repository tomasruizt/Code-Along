#include <stdio.h>
#include "operations.h"

int main() {
    int n = 5;
    int a[] = {1, 2, 3, 4, 5};
    int b[] = {1, 2, 3, 4, 5};
    int c[n];
    mult(a, b, c, n);

    // Device pointers
    int *a_d, *b_d, *c_d;

    // Allocate memory on the device
    cudaMalloc((void**)&a_d, n * sizeof(int));
    cudaMalloc((void**)&b_d, n * sizeof(int));
    cudaMalloc((void**)&c_d, n * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(a_d, a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, n * sizeof(int), cudaMemcpyHostToDevice);

    printf("a = ");
    print_vector(a, n);

    printf("b = ");
    print_vector(b, n);

    printf("c = ");
    print_vector(c, n);
    printf("\n");

    // Free device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    return 0;
}