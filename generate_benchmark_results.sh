#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status,
# Treat unset variables as an error, and
# Prevents errors in a pipeline from being masked.
set -euo pipefail

# ---------------------------
# Configuration Variables
# ---------------------------

# Directories
PROJECT_ROOT="$(pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
BENCHMARK_DIR="${PROJECT_ROOT}/benchmark_results"
SCRIPTS_DIR="${PROJECT_ROOT}/scripts"

# Executables
SGEMM_EXEC="${BUILD_DIR}/sgemm"
CUBLAS_SGEMM_EXEC="${BUILD_DIR}/cuBLAS_sgemm"

# Python Script
PLOT_SCRIPT="${SCRIPTS_DIR}/plot_benchmark_results.py"

# Benchmark Parameters
REPEAT_TIMES=50

KERNEL_RANGE=(0 1 2 3 4 5 6 7 8 9 10 11 12)  # Kernel numbers to run

# ---------------------------
# Helper Functions
# ---------------------------

# Function to check if a command exists
command_exists () {
    command -v "$1" >/dev/null 2>&1
}

# Function to print error messages
error_exit () {
    echo "Error: $1" >&2
    exit 1
}

# ---------------------------
# Pre-Execution Checks
# ---------------------------

# Check for required commands
REQUIRED_COMMANDS=("cmake" "make" "python3" "nvcc")
for cmd in "${REQUIRED_COMMANDS[@]}"; do
    if ! command_exists "${cmd}"; then
        error_exit "Required command '${cmd}' is not installed or not in PATH."
    fi
done

# Check for required Python packages
REQUIRED_PYTHON_PACKAGES=("matplotlib" "seaborn" "pandas")
for pkg in "${REQUIRED_PYTHON_PACKAGES[@]}"; do
    if ! python3 -c "import ${pkg}" &>/dev/null; then
        error_exit "Python package '${pkg}' is not installed. Install it using pip."
    fi
done

# Check if Python plot script exists
if [ ! -f "${PLOT_SCRIPT}" ]; then
    error_exit "Plotting script '${PLOT_SCRIPT}' not found."
fi

# ---------------------------
# Build the Project
# ---------------------------

echo "=============================="
echo "1. Building the Project with CMake"
echo "=============================="

# Create build directory if it doesn't exist
mkdir -p "${BUILD_DIR}"

# Navigate to build directory
cd "${BUILD_DIR}"

# Configure the project
echo "Configuring the project..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Compile the project
echo "Compiling the project..."
make -j"$(nproc)"

# Verify executables exist
if [ ! -f "${SGEMM_EXEC}" ]; then
    error_exit "Executable '${SGEMM_EXEC}' not found after build."
fi

if [ ! -f "${CUBLAS_SGEMM_EXEC}" ]; then
    echo "Warning: Executable '${CUBLAS_SGEMM_EXEC}' not found. Proceeding without it."
fi

# Navigate back to project root
cd "${PROJECT_ROOT}"

# ---------------------------
# Run Benchmarks
# ---------------------------

echo "=============================="
echo "2. Running Benchmarks"
echo "=============================="

# Create benchmark_results directory
mkdir -p "${BENCHMARK_DIR}"

# Iterate over each kernel number and run benchmarks
for kernel_num in "${KERNEL_RANGE[@]}"; do
    echo "-----------------------------------"
    echo "Running Kernel Number: ${kernel_num}"
    echo "-----------------------------------"

    # Define output log file
    LOG_FILE="${BENCHMARK_DIR}/${kernel_num}_output.txt"

    # Execute the benchmark
    if [ "${kernel_num}" -eq 0 ] && [ -f "${CUBLAS_SGEMM_EXEC}" ]; then
        # Run the separate cuBLAS executable if kernel_num is 0
        echo "Running cuBLAS SGEMM..."
        "${CUBLAS_SGEMM_EXEC}" | tee "${LOG_FILE}"
    else
        # Run the main sgemm executable with kernel_num argument
        echo "Running custom SGEMM kernel ${kernel_num}..."
        "${SGEMM_EXEC}" "${kernel_num}" | tee "${LOG_FILE}"
    fi

    # Optional: Introduce a short pause between runs
    sleep 1
done

# ---------------------------
# Generate Plots
# ---------------------------

echo "=============================="
echo "3. Generating Performance Plots"
echo "=============================="

# Run the Python plotting script
python3 "${PLOT_SCRIPT}"

echo "=============================="
echo "Benchmarking and Profiling Complete!"
echo "Results are stored in '${BENCHMARK_DIR}/' and the plot is generated."
echo "=============================="
