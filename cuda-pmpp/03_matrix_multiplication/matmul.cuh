#pragma once

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

#define TILE_SIZE 16

// Function declarations
__host__ __device__ int idx(int i, int j, int rowSize);
__host__ __device__ int idx_col(int i, int j, int colSize);
__global__ void matrix_mult(float *A, float *B, float *C, int m, int n, int p);
__global__ void matrix_mult_tiled(float *A, float *B, float *C, int m, int n, int p);
__global__ void tiledMatMulWithoutBoundsCheck(float *A, float *B, float *C, int n);
void cpu_matrix_mult(float *A, float *B, float *C, int m, int n, int p);
bool compare_matrices(float *A, float *B, int m, int n, float tolerance);
void transpose_and_copy(float* dest, const float* src, int rows, int cols); 