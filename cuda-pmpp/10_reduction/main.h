#pragma once
float *arange(int n);
float *cuda_sum(float *vec, int len);
float cpu_sum(float* vec, int n);
__global__ void cudaNaiveSum(float* vec, int len, float* result);
