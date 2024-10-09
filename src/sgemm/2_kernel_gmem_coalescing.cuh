#pragma once

#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>


/* 
 * Coalesce column and row accesses within warps 
 */
template<const uint BLOCK_SIZE>
__global__ void sgemm_global_mem_coalesce(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const uint row = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const uint col = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (row < M && col < N) {
        float temp = 0.0;
        
        #pragma unroll
        for (int i = 0; i < K; ++i) {
            temp += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = temp * alpha + beta * C[row * N + col];
    }
}