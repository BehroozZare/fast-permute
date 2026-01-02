#!/usr/bin/env bash

# =============================================================================
# SLIM Benchmark Script
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SOLVER="CUDSS"  # Options: CUDSS, MKL
INPUT_DIR="/media/behrooz/FarazHard/Last_Project/slim_benchmark/hand_high_res"
OUTPUT_CSV=""/home/behrooz/Desktop/Last_Project/gpu_ordering/output/SLIM/slim""
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/multiple_factorization/gpu_ordering_multi_slim_benchmark"
DIM=2

# -----------------------------------------------------------------------------
# Section A: DEFAULT ordering
# -----------------------------------------------------------------------------
echo "=== Running DEFAULT ordering ==="
"$BENCHMARK_BIN" \
    -k "$INPUT_DIR" \
    -s "$SOLVER" \
    -a DEFAULT \
    -d "$DIM" \
    -o "$OUTPUT_CSV"

# -----------------------------------------------------------------------------
# Section B: PATCH_ORDERING (patch_type × binary_level)
# -----------------------------------------------------------------------------
echo "=== Running PATCH_ORDERING ==="
PATCH_TYPES=("rxmesh")
BINARY_LEVELS=(8 10)

for patch_type in "${PATCH_TYPES[@]}"; do
    for binary_level in "${BINARY_LEVELS[@]}"; do
        echo "Processing: patch_type=$patch_type | binary_level=$binary_level"
        "$BENCHMARK_BIN" \
            -k "$INPUT_DIR" \
            -s "$SOLVER" \
            -a PATCH_ORDERING \
            -p "$patch_type" \
            -b "$binary_level" \
            -d "$DIM" \
            -o "$OUTPUT_CSV"
    done
done

# -----------------------------------------------------------------------------
# Section C: PARTH ordering (binary_level)
# -----------------------------------------------------------------------------
echo "=== Running PARTH ordering ==="
for binary_level in "${BINARY_LEVELS[@]}"; do
    echo "Processing: binary_level=$binary_level"
    "$BENCHMARK_BIN" \
        -k "$INPUT_DIR" \
        -s "$SOLVER" \
        -a PARTH \
        -b "$binary_level" \
        -d "$DIM" \
        -o "$OUTPUT_CSV"
done

echo "=== SLIM Benchmark complete ==="

