#pragma once

#include <cuda_runtime.h>
#include <cstdlib>

#define CEIL_DIV(M, N) ((M + N - 1) / N)

// reinterpret_cast can't cast away modifies like const
template<const int block_size_M, const int block_size_N, const int block_size_K, const int tile_size_M, const int tile_size_N>
__global__ void sgemmVectorize(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    A += cRow * block_size_M * K;
    B += cCol * block_size_N;
    C += cRow * block_size_M * N + cCol * block_size_N;

    __shared__ float AShmem[block_size_M * block_size_K];
    __shared__ float BShmem[block_size_K * block_size_N];

    float thread_results[tile_size_M * tile_size_N] = {0.0f};
    float register_M[tile_size_M] = {0.0};
    float register_N[tile_size_N] = {0.0};

    const uint threadRow = threadIdx.x / (block_size_N / tile_size_N);
    const uint threadCol = threadIdx.x % (block_size_N / tile_size_N);

    const uint innerRowA = threadIdx.x / (block_size_K / 4);
    const uint innerColA = threadIdx.x % (block_size_K / 4);
    const uint innerRowB = threadIdx.x / (block_size_N / 4);
    const uint innerColB = threadIdx.x % (block_size_N / 4);


    for (int bckIdx = 0; bckIdx < K; bckIdx += block_size_K) {
        // float4 contains 4 float values in a vector
        // treating the address as if it points to float4 instead of a single flaot
        // [0] derefences the pointer, reading the first elment 
        // access 4 at once reduces memory transactions 
        // float4 is 16 byte, so memory aligned
        float4 temp = reinterpret_cast<float4 *>(&A[innerRowA * K + innerColA * 4])[0];
        
        // transpose A during GMEM to SHMEM transfer
        // for coaslecing memory access, wehn threads access it in col major order, the accesses are coalesced
        // also arranged to minimize bank conflicts
        AShmem[(innerColA * 4 + 0) * block_size_M + innerRowA] = temp.x;
        AShmem[(innerColA * 4 + 1) * block_size_M + innerRowA] = temp.y;
        AShmem[(innerColA * 4 + 2) * block_size_M + innerRowA] = temp.z;
        AShmem[(innerColA * 4 + 3) * block_size_M + innerRowA] = temp.w;

        reinterpret_cast<float4 *>(&BShmem[innerRowB * block_size_N + innerColB * 4])[0] = 
            reinterpret_cast<float4 *>(&B[innerRowB * N + innerColB * 4])[0];

        __syncthreads();

        A += block_size_K;
        B += block_size_K * N;

        for (int dotIdx = 0; dotIdx < block_size_K; ++dotIdx) {
            // load a
            for (int i = 0; i < tile_size_M; ++i) {
                register_M[i] = AShmem[dotIdx * block_size_M + threadRow * tile_size_M + i];
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

    for (uint resIdxM = 0; resIdxM < tile_size_M; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < tile_size_N; resIdxN += 4) {
            // load C into registers 
            float4 temp = reinterpret_cast<float4 *>(&C[(threadRow * tile_size_M + resIdxM) * N + threadCol * tile_size_N + resIdxN])[0];

            // perform GMEM in register
            temp.x = alpha * thread_results[resIdxM * tile_size_N + resIdxN + 0] + beta * temp.x;
            temp.y = alpha * thread_results[resIdxM * tile_size_N + resIdxN + 1] + beta * temp.y;
            temp.z = alpha * thread_results[resIdxM * tile_size_N + resIdxN + 2] + beta * temp.z;
            temp.w = alpha * thread_results[resIdxM * tile_size_N + resIdxN + 3] + beta * temp.w;

            // write back
            reinterpret_cast<float4 *>(&C[(threadRow * tile_size_M + resIdxM) * N + threadCol * tile_size_N + resIdxN])[0] = temp;
        }
    }
}


