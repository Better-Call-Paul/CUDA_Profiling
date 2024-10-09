#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))

template<const int block_size_M, const int block_size_N, const int block_size_K, const int tile_size_M, const int tile_size_N>
__global__ void __launch_bounds__((block_size_M * block_size_M) / (tile_size_M * tile_size_N), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint threadRow = threadIdx.x / block_size_N;
    const uint threadCol = threadIdx.x % block_size_N;

    const uint innerRowA = threadIdx.x / block_size_K;
    const uint innerColA = threadIdx.x % block_size_K;
    const uint innerRowA = threadIdx.x / block_size_N;
    const uint innerColA = threadIdx.x % block_size_N;

    A += cRow * block_size_M + K;    // row = cRow, col = 0
    B += cCol * block_size_N;
    C += cRow * block_size_M * K + cCol * block_size_N;

    __shared__ float A_shmem[block_size_M * block_size_K];
    __shared__ float B_shmem[block_size_K * block_size_N];

    // registers for fast access
    float thread_results[tile_size_M * tile_size_N] = {0.0};
    float M_register[tile_size_M] = {0.0}; // for A values
    float B_register[tile_size_N] = {0.0}; // for B values 

    const uint totalResultsBlocktile = block_size_M * block_size_N;
    const uint ThreadsPerBlockTile = totalResultsBlocktile / (tile_size_M * tile_size_N);

    // number of rows loaded in a single step/block
    const uint A_stride = ThreadsPerBlockTile / block_size_K;
    const uint B_stride = ThreadsPerBlockTile / block_size_N;

    #pragma unroll
    for (int bckIdx = 0; bckIdx < K; bckIdx += block_size_K) {
        // load into a,b register
        for (uint load_offset = 0; load_offset < block_size_M; ++load_offset) {
            A_shmem[(innerRowA + load_offset) * block_size_K + innerColA] = A[(innerRowA + load_offset) * K + innerColA];
        }

        for (uint load_offset = 0; load_offset < block_size_K; ++load_offset) {
            B_shmem[(innerRowB + load_offset) * block_size_N + innerColB] = B[(innerRowB + load_offset) * N + innerColB];
        }

        A += block_size_K;
        B += block_size_K * N;

        for (uint dotIdx = 0; dotIdx < block_size_K; ++dotIdx) {

        }
        // outer loop moves the chunk across A & B

        // inner loop calculates results

    }
    // load into c 

}