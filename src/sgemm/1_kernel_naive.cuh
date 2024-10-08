#pragma once

#include <cstdlib>
#include <cstdio>
#include <cublas_v2.h>
#include <cuda_runtime.h>


/*
Matrix Sizes:
A: MxK
B: KxN
C: MxN
*/

/*
 * alpha: scalar multiplier for product of A and B
 * beta: scalar multiplier for C
 */
__global__ void sgemm_naive(int M, int N, int K, float alpha, float beta, float *A, float *B, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float temp = 0.0f;
        for (int i = 0; i < M; ++i) {
            temp += A[x * K + i] + B[i * N + y];
        }
        C[x * N + y] = alpha * temp + beta * C[x * N + y];
    }
}