#pragma once

#include <cuda_runtime.h>
#include <cstdlib>

#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))

/*
Matrix Sizes:
A: MxK
B: KxN
C: MxN
*/

template<const int BLOCK_SIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // load shared_memory for A & B
    __shared__ float a_shmem[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float b_shmem[BLOCK_SIZE * BLOCK_SIZE];

    const uint threadRow = threadIdx.x / BLOCK_SIZE;
    const uint threadCol = threadIdx.x % BLOCK_SIZE;

    // move pointers to start position
    A += cRow * BLOCK_SIZE * K;                       // row = cRow, col = 0
    B += cCol * BLOCK_SIZE;                           // row = 0, cCol = cCol
    C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE;   // row = cRow, col = cCol
           
    float temp = 0.0;

    #pragma unroll 
    for (int blckIdx = 0; blckIdx < K; blckIdx += BLOCK_SIZE) {

        // make thread col the consecutive index --> gmem coalescing 
        a_shmem[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
        b_shmem[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        #pragma unroll 
        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
            temp += a_shmem[threadRow * BLOCK_SIZE + dotIdx] * b_shmem[dotIdx * BLOCK_SIZE + threadCol];
        }

        __syncthreads();
    }
    C[threadRow * N + threadCol] = temp * alpha + beta * C[threadRow * N + threadCol];
}