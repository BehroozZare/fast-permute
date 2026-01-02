#!/usr/bin/env bash

# =============================================================================
# IPC Benchmark Script
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_ROOT="/media/behrooz/FarazHard/IPC_matrices/MatOnBoard/sphere19K_DCOBallHitWall_NH_NM_interiorPoint_20251226022300"
OUTPUT_CSV="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/IPC"
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/gpu_ordering_ipc_benchmark"

# PATCH_ORDERING parameters
PATCH_SIZE=256
BINARY_LEVEL=8
PATCH_TYPE="rxmesh"

# -----------------------------------------------------------------------------
# Section A: DEFAULT ordering
# -----------------------------------------------------------------------------
echo "=== Running DEFAULT ordering ==="
"$BENCHMARK_BIN" \
    -a DEFAULT \
    -s CUDSS \
    -k "$INPUT_ROOT" \
    -o "$OUTPUT_CSV"

# -----------------------------------------------------------------------------
# Section B: PATCH_ORDERING
# -----------------------------------------------------------------------------
echo "=== Running PATCH_ORDERING ==="
"$BENCHMARK_BIN" \
    -a PATCH_ORDERING \
    -s CUDSS \
    -k "$INPUT_ROOT" \
    -o "$OUTPUT_CSV" \
    -p "$PATCH_TYPE" \
    -z "$PATCH_SIZE" \
    -b "$BINARY_LEVEL"

echo "=== IPC Benchmark complete ==="

