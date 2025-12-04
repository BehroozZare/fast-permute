#!/usr/bin/env bash

# =============================================================================
# CUDSS Ordering Benchmark Script
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_ROOT="/media/behrooz/FarazHard/Last_Project/MIT_meshes_small"
OUTPUT_CSV="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/small_benchmark"
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/gpu_ordering_cudss_benchmark"

# -----------------------------------------------------------------------------
# Mesh Discovery
# -----------------------------------------------------------------------------
mapfile -t MESHES < <(find "$INPUT_ROOT" -type f -name "*.obj")
echo "Found ${#MESHES[@]} meshes"

# -----------------------------------------------------------------------------
# Section A: DEFAULT ordering
# -----------------------------------------------------------------------------
echo "=== Running DEFAULT ordering ==="
for mesh in "${MESHES[@]}"; do
    echo "Processing: $mesh"
    "$BENCHMARK_BIN" \
        -i "$mesh" \
        -s CUDSS \
        -a DEFAULT \
        -g 0 \
        -o "$OUTPUT_CSV"
done

# -----------------------------------------------------------------------------
# Section B: PARTH ordering (binary_level: 2, 4, 6, 8, 10)
# -----------------------------------------------------------------------------
echo "=== Running PARTH ordering ==="
for mesh in "${MESHES[@]}"; do
    for binary_level in 2 4 6 8 10; do
        echo "Processing: $mesh | binary_level=$binary_level"
        "$BENCHMARK_BIN" \
            -i "$mesh" \
            -s CUDSS \
            -a PARTH \
            -g 0 \
            -b "$binary_level" \
            -o "$OUTPUT_CSV"
    done
done

# -----------------------------------------------------------------------------
# Section C: PATCH_ORDERING (patch_type × patch_size × binary_level)
# -----------------------------------------------------------------------------
echo "=== Running PATCH_ORDERING ==="
PATCH_TYPES=("metis_kway" "rxmesh")
PATCH_SIZES=(64 256 512)
BINARY_LEVELS=(2 4 6 8 10)

for mesh in "${MESHES[@]}"; do
    for patch_type in "${PATCH_TYPES[@]}"; do
        for patch_size in "${PATCH_SIZES[@]}"; do
            for binary_level in "${BINARY_LEVELS[@]}"; do
                echo "Processing: $mesh | patch_type=$patch_type | patch_size=$patch_size | binary_level=$binary_level"
                "$BENCHMARK_BIN" \
                    -i "$mesh" \
                    -s CUDSS \
                    -a PATCH_ORDERING \
                    -g 0 \
                    -p "$patch_type" \
                    -z "$patch_size" \
                    -b "$binary_level" \
                    -o "$OUTPUT_CSV"
            done
        done
    done
done

echo "=== Benchmark complete ==="
