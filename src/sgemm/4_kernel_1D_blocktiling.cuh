#pragma once

#include <cstdlib>
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>

#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))

template<const int block_size_M, const int block_size_N, const int block_size_K, const int tile_size_M>
__global__ void sgemm1DBlocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadRow = threadIdx.x / block_size_N;
    const uint threadCol = threadIdx.x % block_size_N;

    __shared__ float A_shmem[block_size_M * block_size_K];
    __shared__ float B_shmem[block_size_K * block_size_N];

    A += cRow * block_size_M * K;
    B += cCol * block_size_N;
    C += cRow * block_size_M * K + cCol * block_size_N;

    assert(block_size_M * block_size_K == blockDim.x);
    assert(block_size_K * block_size_N == blockDim.x);
    
    // warp level gmem coalescing
    const uint innerRowA = threadIdx.x / block_size_K;
    const uint innerColA = threadIdx.x % block_size_K;
    const uint innerRowB = threadIdx.x / block_size_N;
    const uint innerColB = threadIdx.x % block_size_N;

    float thread_results[tile_size_M] = {0.0};

    #pragma unroll
    for (uint bckIdx = 0; bckIdx < K; bckIdx += block_size_K) {
        
        // load all for gmem coalescing
        A_shmem[innerRowA * block_size_K + innerColA] = A[innerRowA * K + innerColA];
        B_shmem[innerRowB * block_size_N + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        A += block_size_K;
        B += block_size_K * N;

        #pragma unroll
        for (uint dotIdx = 0; dotIdx < block_size_K; ++dotIdx) {
            
            float temp_b = B_shmem[dotIdx * block_size_N + threadCol];

            #pragma unroll
            for (uint resIdx = 0; resIdx < tile_size_M; ++resIdx) {
                thread_results[resIdx] += A_shmem[(threadRow * tile_size_M + resIdx) * block_size_K + dotIdx] * temp_b;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (uint resIdx = 0; resIdx < tile_size_M; ++resIdx) {
        C[(threadRow * tile_size_M + resIdx) * N + threadCol] = thread_results[resIdx] * alpha + beta *  C[(threadRow * tile_size_M + resIdx) * N + threadCol];
    }
}