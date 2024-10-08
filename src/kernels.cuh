#include "sgemm/1_kernel_naive.cuh"
#include "sgemm/2_kernel_gmem_coalescing.cuh"
#include "sgemm/3_shared_mem_blocking.cuh"
#include "sgemm/4_kernel_1D_blocktiling.cuh"
#include "sgemm/5_kernel_2d_blocktiling.cuh"
#include "sgemm/6_kernel_vectorize.cuh"
#include "sgemm/7_kernel_resolve_bank_conflict.cuh"
#include "sgemm/8_kernel_bank_extra_col.cuh"
#include "sgemm/9_kernel_autotuned.cuh"
#include "sgemm/10_kernel_warptiling.cuh"