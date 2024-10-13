#pragma once

#include <cuda_runtime.h>
#include <cstdlib>

template<const int block_size_M, const int block_size_N, const int block_size_K, const int tile_size_M, const int tile_size_N>
__global__ void sgemmAutoTuned(int M, int N, int K, float alpha, const float *A, const float *B, float beta, const float *C) {

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    

}


















