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
    const uint innerRowB = threadIdx.x / block_size_N;
    const uint innerColB = threadIdx.x % block_size_N;

    A += cRow * block_size_M + K;    // row = cRow, col = 0
    B += cCol * block_size_N;
    C += cRow * block_size_M * K + cCol * block_size_N;

    __shared__ float A_shmem[block_size_M * block_size_K];
    __shared__ float B_shmem[block_size_K * block_size_N];

    // registers for fast access
    float thread_results[tile_size_M * tile_size_N] = {0.0};
    float M_register[tile_size_M] = {0.0}; // for A values
    float N_register[tile_size_N] = {0.0}; // for B values 

    const uint totalResultsBlocktile = block_size_M * block_size_N;
    const uint ThreadsPerBlockTile = totalResultsBlocktile / (tile_size_M * tile_size_N);

    // number of rows loaded in a single step/block
    const uint A_stride = ThreadsPerBlockTile / block_size_K;
    const uint B_stride = ThreadsPerBlockTile / block_size_N;

    #pragma unroll
    for (int bckIdx = 0; bckIdx < K; bckIdx += block_size_K) {
        // load into a,b register
        for (uint load_offset = 0; load_offset < block_size_M; load_offset += A_stride) {
            A_shmem[(innerRowA + load_offset) * block_size_K + innerColA] = A[(innerRowA + load_offset) * K + innerColA];
        }

        for (uint load_offset = 0; load_offset < block_size_K; load_offset += B_stride) {
            B_shmem[(innerRowB + load_offset) * block_size_N + innerColB] = B[(innerRowB + load_offset) * N + innerColB];
        }

        __syncthreads();

        A += block_size_K;
        B += block_size_K * N;

        for (uint dotIdx = 0; dotIdx < block_size_K; ++dotIdx) {
            //load a and b vals into registers
            for (uint i = 0; i < tile_size_M; ++i) {
                M_register[i] = A_shmem[(threadRow * tile_size_M + i) * block_size_K + dotIdx];
            }

            for (uint i = 0; i < tile_size_N; ++i) {
                N_register[i] = B_shmem[dotIdx * block_size_N + i + threadCol * tile_size_N];
            }
        }
        // through each reg 
        for (uint m_i = 0; m_i < tile_size_M; ++m_i) {
            for (uint n_i = 0; n_i < tile_size_N; ++n_i) {
                thread_results[m_i * tile_size_N + n_i] += M_register[m_i] * N_register[n_i];
            }
        }
        __syncthreads();
    }
    // load into c 
    #pragma unroll
    for (int resIdxM = 0; resIdxM < tile_size_M; ++resIdxM) {
        for (int resIdxN = 0; resIdxN < tile_size_N; ++resIdxN) {
            C[(threadRow * tile_size_M + resIdxM) * N + threadCol * tile_size_N + resIdxN] = alpha * thread_results[resIdxM * tile_size_N + resIdxN]
             + beta * C[(threadRow * tile_size_M + resIdxM) * N + threadCol * tile_size_N + resIdxN];
        }
    }

}