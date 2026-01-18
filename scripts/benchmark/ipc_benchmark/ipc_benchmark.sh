#!/usr/bin/env bash

# =============================================================================
# IPC Ordering Benchmark Script
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
OUTPUT_DIR="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/IPC"
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/multiple_factorization/gpu_ordering_multi_ipc_benchmark"
# CHECK_POINT="/media/behrooz/FarazHard/IPC_matrices/MatOnBoard/mat225x225t40-mat225x225t40_fall_NH_BE_interiorPoint_20260115020027"
# CHECK_POINT="/media/behrooz/FarazHard/IPC_matrices/test"
CHECK_POINT="/media/behrooz/FarazHard/IPC_matrices/MatOnBoard/mat225x225t40_null_NH_BE_interiorPoint_20260117123833"

# # -----------------------------------------------------------------------------
# # Section A: DEFAULT ordering
# # -----------------------------------------------------------------------------
# echo "=== Running DEFAULT ordering ==="
# "$BENCHMARK_BIN" \
#     -a DEFAULT \
#     -k "$CHECK_POINT" \
#     -o "${OUTPUT_DIR}/ipc_default"

# -----------------------------------------------------------------------------
# Section B: PATCH_ORDERING
# -----------------------------------------------------------------------------
echo "=== Running PATCH_ORDERING ==="
"$BENCHMARK_BIN" \
    -a PATCH_ORDERING \
    -k "$CHECK_POINT" \
    -o "${OUTPUT_DIR}/ipc_patch_ordering"

echo "=== Benchmark complete ==="
