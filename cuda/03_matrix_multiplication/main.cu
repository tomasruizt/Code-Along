#define ARMA_ALLOW_FAKE_GCC
#include <stdio.h>
#include <armadillo>

int idx(int i, int j, int rowSize) {
    return i * rowSize + j;
}

void print_matrix(int m, int n, double *A) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", A[idx(i, j, n)]);
        }
        printf("\n");
    }
}

int main() {
    int m = 10;
    int n = 10;
    int p = 10;
    arma::mat A(m, n, arma::fill::randu);  // 1000x1000 random matrix
    arma::mat B(n, p, arma::fill::randu);  // 1000x1000 random matrix
    arma::mat C = A * B;  // Matrix multiplication
    print_matrix(m, p, C.memptr());

    int *a_d;
    cudaMalloc((void**)&a_d, n * sizeof(int));
    cudaFree(a_d);
    return 0;
}
