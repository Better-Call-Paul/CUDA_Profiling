#pragma once

#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))

template<const int block_size_M, const int block_size_N, const int block_size_K, int tile_size_M>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    const uint threadRow = threadIdx.x / block_size_N;
    const uint threadCol = threadIdx.x % block_size_N;

    __shared__ float A_shmem[block_size_M * block_size_K];
    __shared__ float B_shmem[block_size_K * block_size_N];

    A += block_size_M * cRow * K;
    B += block_size_N * cCol;
    C += block_size_N * cCol + cRow * block_size_M * N;

    assert(block_size_M * block_size_K == blockDim.x);
    assert(block_size_K * block_size_N == blockDim.y);

    // warp level GMEM coalescing
    const uint innerRowA = threadIdx.x / block_size_K;
    const uint innerColA = threadIdx.x % block_size_K;
    const uint innerRowB = threadIdx.x / block_size_N;
    const uint innerColB = threadIdx.x % block_size_K;

    float threadResults[tile_size_M] = {0.0};

    for (int bkIdx = 0; bkIdx < K; bkIdx += block_size_K) {
        A_shmem[innerRowA * block_size_K + innerColA] = A[innerRowA * K + innerColA];
        B_shmem[innerRowB * block_size_N + innerColB] = B[innerRowB * N + innerColB];

        __syncthreads();

        A += block_size_K;
        B += block_size_K * N;

        for (uint dotIdx = 0; dotIdx < block_size_K; ++dotIdx) {
            // re-use B entries, cache temp var
            float tempB = B_shmem[dotIdx * block_size_N + threadCol];
            for (uint resIdx = 0; resIdx < tile_size_M; ++resIdx) {
                threadResults[resIdx] += A_shmem[(threadRow * tile_size_M + resIdx) * block_size_K + dotIdx] * tempB;
            }
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < tile_size_M; ++resIdx) {
        C[(threadRow * tile_size_M + resIdx) * N + threadCol] = alpha * threadResults[resIdx] + beta * C[(threadRow * tile_size_M + resIdx) * N + threadCol];
    }
}