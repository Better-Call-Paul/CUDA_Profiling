#!/usr/bin/env bash

set -u

#def ranges
BLOCK_SIZE_K_VALUES=(8 16 32 64)
BLOCK_SIZE_M_VALUES=(64 128 256)
BLOCK_SIZE_N_VALUES=(64 128 256)

TILE_SIZE_N=(4 8 16 32)
TILE_SIZE_M=(4 8 16 32)

NUM_THREADS=(256)

cd "$(dirname "$0")"
cd "../build"

RUNNER="../src/runner.cu"
KERNEL="../src/sgemm/9_kernel_autotuned.cuh"
OUTPUT="../benchmark_results/kernel_9_autotuned_results.txt"

echo "" > $OUTPUT

export DEVICE="2"

TOTAL_CONFIGS="$(( ${#BLOCK_SIZE_K_VALUES[@]} * ${#BLOCK_SIZE_M_VALUES[@]} * ${#BLOCK_SIZE_N_VALUES[@]} * ${#TILE_SIZE_M[@]} * ${#TILE_SIZE_N[@]} * ${#NUM_THREADS[@]} ))"
CONFIG_NUM=0

# check all possible conbinations of parameters
for bk in ${#BLOCK_SIZE_K_VALUES[@]}; do
    for tm in ${#TILE_SIZE_M[@]}; do
        for tn in ${#BLOCK_SIZE_N_VALUES[@]}; do
            for bm in ${#BLOCK_SIZE_M_VALUES[@]}; do
                for bn in ${#BLOCK_SIZE_N_VALUES[@]}; do
                    for nt in ${#NUM_THREADS[@]}; do 

                        # skip configurations that don't fullfil preconditions
                        config="BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NT=$nt"
                        if [[ $(( ($nt * 4) % bk )) -ne 0 ]]; then
                        echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BK = $(( ($nt * 4) % bk )) != 0))"
                        continue
                        fi
                        if [[ $(( ($nt * 4) % bn )) -ne 0 ]]; then
                        echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BN = $(( ($nt * 4) % bn )) != 0))"
                        continue
                        fi
                        if [[ $(( $bn % (16 * $tn ) )) -ne 0 ]]; then
                        echo "QUANTIZATION: Skipping $config because BN % (16 * TN) = $(( $bn % (16 * $tn ) )) != 0))"
                        continue
                        fi
                        if [[ $(( $bm % (16 * $tm ) )) -ne 0 ]]; then
                        echo "QUANTIZATION: Skipping $config because BM % (16 * TM) = $(( $bm % (16 * $tm ) )) != 0))"
                        continue
                        fi
                        if [[ $(( ($bm * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
                        echo "VECTORIZE: Skipping $config because (BM * BK) % (4 * NUM_THREADS) = $(( ($bm * $bk) % ( 4 * 256 ) )) != 0))"
                        continue
                        fi
                        if [[ $(( ($bn * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
                        echo "VECTORIZE: Skipping $config because (BN * BK) % (4 * NUM_THREADS) = $(( ($bn * $bk) % ( 4 * 256 ) )) != 0))"
                        continue
                        fi

                        # Update the parameters in the source code
                        sed -i "s/const uint K9_BK = .*/const uint K9_BK = $bk;/" $RUNNER
                        sed -i "s/const uint K9_TM = .*/const uint K9_TM = $tm;/" $RUNNER
                        sed -i "s/const uint K9_TN = .*/const uint K9_TN = $tn;/" $RUNNER
                        sed -i "s/const uint K9_BM = .*/const uint K9_BM = $bm;/" $RUNNER
                        sed -i "s/const uint K9_BN = .*/const uint K9_BN = $bn;/" $RUNNER
                        sed -i "s/const int K9_NUM_THREADS = .*/const int K9_NUM_THREADS = $nt;/" $KERNEL
                        
                        # Rebuild the program
                        make 

                        echo "($CONFIG_NUM/$TOTAL_CONFIGS): BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NUM_THREADS=$nt" |& tee -a $OUTPUT
                        # Run the benchmark and get the result
                        # Kill the program after 4 seconds if it doesn't finish
                        timeout -v 4 ./sgemm 9 | tee -a $OUTPUT

                    done
                done
            done
        done
    done
done





