#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))

template<const int block_size_M, const int block_size_N, const int block_size_K, const int tile_size_M, const int tile_size_N>
__global__ void sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadRow = threadIdx.x / block_size_N;
    const uint threadCol = threadIdx.x % block_size_N;

    const uint innerRowA = threadIdx.x / block_size_K;
    const uint innerColA = threadIdx.x % block_size_K;
    const uint innerRowA = threadIdx.x / block_size_N;
    const uint innerColA = threadIdx.x % block_size_N;

    __shared__ float A_shmem[block_size_M * block_size_K];
    __shared__ float B_shmem[block_size_K * block_size_N];

    float thread_results[tile_size_M * tile_size_N] = {0.0};

    #pragma unroll
}


// populate smmem
    // load multiple elemtns

// loud a and b into registers 
// nested dot prod