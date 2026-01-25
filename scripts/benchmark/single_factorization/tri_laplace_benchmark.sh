l#!/usr/bin/env bash

# =============================================================================
# CUDSS Ordering Benchmark Script
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SOLVER="CUDSS"  # Options: CUDSS, MKL
INPUT_ROOT="/media/behrooz/FarazHard/Last_Project/BenchmarkMesh/tri-mesh/final"
OUTPUT_CSV="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/single_factorization/tri_CUDSS_benchmark"
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/single_factorization/gpu_ordering_tri_mesh_laplace_benchmark"

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
        -s "$SOLVER" \
        -a DEFAULT \
        -g 0 \
        -o "$OUTPUT_CSV"
done

# # -----------------------------------------------------------------------------
# # Section B: PARTH ordering (binary_level: ()8, 10)
# # -----------------------------------------------------------------------------
#  echo "=== Running PARTH ordering ==="
#  for mesh in "${MESHES[@]}"; do
#      for binary_level in 8 10; do
#          echo "Processing: $mesh | binary_level=$binary_level"
#          "$BENCHMARK_BIN" \
#              -i "$mesh" \
#              -s "$SOLVER" \
#              -a PARTH \
#              -g 0 \
#              -b "$binary_level" \
#              -o "$OUTPUT_CSV"
#      done
#  done

# -----------------------------------------------------------------------------
# Section C: PATCH_ORDERING (patch_type × patch_size × binary_level)
# -----------------------------------------------------------------------------
#  echo "=== Running PATCH_ORDERING ==="
#  PATCH_TYPES=("rxmesh" "metis_kway")
#  PATCH_SIZES=(512)
#  BINARY_LEVELS=(8 10)

#  for mesh in "${MESHES[@]}"; do
#      for patch_type in "${PATCH_TYPES[@]}"; do
#          for patch_size in "${PATCH_SIZES[@]}"; do
#              for binary_level in "${BINARY_LEVELS[@]}"; do
#                  echo "Processing: $mesh | patch_type=$patch_type | patch_size=$patch_size | binary_level=$binary_level"
#                  "$BENCHMARK_BIN" \
#                      -i "$mesh" \
#                      -s "$SOLVER" \
#                      -a PATCH_ORDERING \
#                      -g 0 \
#                      -p "$patch_type" \
#                      -z "$patch_size" \
#                      -b "$binary_level" \
#                      -o "$OUTPUT_CSV"
#              done
#          done
#      done
#  done

#  echo "=== Benchmark complete ==="
