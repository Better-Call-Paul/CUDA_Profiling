#pragma once

#include <cassert>
#include <cstdlib>
#include <cuda_runtime.h>

#define CEIL_DIV(M, N) (((M) + (N - 1)) / (N))

template<const int block_size_M, const int block_size_N, const int block_size_K, const int tile_size_M, const int tile_size_N>
__global__ void __launch_bounds__((block_size_M * block_size_N) / (tile_size_M * tile_size_N), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {

    const uint total_block_tile = block_size_M * block_size_N;
    const uint threads_per_blocktile = total_block_tile / (tile_size_M * tile_size_N);
    assert(threads_per_blocktile == blockDim.x);

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    A += cRow * block_size_M * K;
    B += cCol * block_size_N;
    C += cRow * block_size_M * N + cCol * block_size_N;

    __shared__ float AShmem[block_size_M * block_size_K];
    __shared__ float BShmem[block_size_K * block_size_N];

    float thread_results[tile_size_M * tile_size_N] = {0.0f};

    // fast operations by caching values 
    float register_M[tile_size_M] = {0.0f};
    float register_N[tile_size_N] = {0.0f};

    const uint threadRow = threadIdx.x / (block_size_N / tile_size_N);
    const uint threadCol = threadIdx.x % (block_size_N / tile_size_N);

    const uint innerRowA = threadIdx.x / block_size_K;
    const uint innerColA = threadIdx.x % block_size_K;
    const uint innerRowB = threadIdx.x / block_size_N;
    const uint innerColB = threadIdx.x % block_size_N;
    
    static_assert(block_size_K != 0);
    static_assert(block_size_N != 0);

    const uint stride_A = threads_per_blocktile / block_size_K;
    const uint stride_B = threads_per_blocktile / block_size_N;

    for (int bckIdx = 0; bckIdx < K; bckIdx += block_size_K) {
        // load shmem
        for (int load_offset = 0; load_offset < block_size_M; load_offset += stride_A) {
            AShmem[(innerRowA + load_offset) * block_size_K + innerColA] = A[(innerRowA + load_offset) * K + innerColA];
        }

        for (int load_offset = 0; load_offset < block_size_K; load_offset += stride_B) {
            BShmem[(innerRowB + load_offset) * block_size_N + innerColB] = B[(innerRowB + load_offset) * N + innerColB];
        }

        __syncthreads();

        A += block_size_K;
        B += block_size_K * N;

        for (int dotIdx = 0; dotIdx < block_size_K; ++dotIdx) {
            // load a
            for (int i = 0; i < tile_size_M; ++i) {
                register_M[i] = AShmem[(threadRow * tile_size_M + i) * block_size_K + dotIdx];
            }

            for (int i = 0; i < tile_size_N; ++i) {
                register_N[i] = BShmem[(dotIdx * block_size_N + threadCol * tile_size_N + i)];
            }

            for (int index_m = 0; index_m < tile_size_M; ++index_m) {
                for (int index_n = 0; index_n < tile_size_N; ++index_n) {
                    thread_results[index_m * tile_size_N + index_n] += register_M[index_m] * register_N[index_n];
                }
            }
    
        }
        __syncthreads();
    } 
    for (int index_m = 0; index_m < tile_size_M; ++index_m) {
        for (int index_n = 0; index_n < tile_size_N; ++index_n) {
            // row: threadRow * tile_size_M + index_m
            // col: threadCol * tile_size_N + index_n
            C[(threadRow * tile_size_M + index_m) * N + threadCol * tile_size_N + index_n] = alpha * thread_results[index_m * tile_size_N + index_n] + beta * C[(threadRow * tile_size_M + index_m) * N + threadCol * tile_size_N + index_n];
        }
    }
}