#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Script: run_sgemm_benchmarks.sh
# Description: Benchmarks SGEMM CUDA kernels by running the 'sgemm' executable
#              with various kernel numbers, logs the outputs, and invokes
#              the Python plotting script.
# =============================================================================

# Create benchmark_results directory if it doesn't exist
mkdir -p benchmark_results

# Define the SGEMM executable name
SGEMM_EXEC=./build/sgemm

# Check if the SGEMM executable exists and is executable
if [[ ! -x "${SGEMM_EXEC}" ]]; then
    echo "Error: Executable '${SGEMM_EXEC}' not found or not executable."
    exit 1
fi

# Iterate through SGEMM kernel numbers (adjust the range as needed)
for kernel in {0..10}; do
    echo "Running SGEMM kernel ${kernel}..."
    # Define the log file name
    LOG_FILE="benchmark_results/sgemm_${kernel}_output.txt"
    # Execute the benchmark and log the output
    ${SGEMM_EXEC} ${kernel} | tee "${LOG_FILE}"
    # Optional: Wait for 2 seconds between runs to ensure system stability
    sleep 2
done

# Invoke the Python plotting script
python3 scripts/plot_benchmark_results.py

echo "SGEMM benchmarking completed. Results are saved in 'benchmark_results/'."
