l#!/usr/bin/env bash

# =============================================================================
# CUDSS Ordering Benchmark Script
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
#NOTE: Change the iparm[1] = 3 for MKL for Parallel METIS ordering
#NOTE: Change the iparm[1] = 2 for MKL for METIS ordering
#NOTE: Change the iparm[1] = 0 for MKL for AMD ordering

SOLVER="MKL"  # Options: CUDSS, MKL
INPUT_ROOT="/media/behrooz/FarazHard/Last_Project/BenchmarkMesh/final"
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/single_factorization/gpu_ordering_tri_mesh_laplace_benchmark"

# -----------------------------------------------------------------------------
# Mesh Discovery
# -----------------------------------------------------------------------------
mapfile -t MESHES < <(find "$INPUT_ROOT" -type f -name "*.obj")
echo "Found ${#MESHES[@]} meshes"

# # -----------------------------------------------------------------------------
# # Section A: DEFAULT ordering + METIS
# # -----------------------------------------------------------------------------
# echo "=== Running DEFAULT ordering + $DEFAULT_ORDERING_TYPE ==="
# DEFAULT_ORDERING_TYPE="METIS"
# OUTPUT_CSV="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/single_factorization/metis_benchmark"
# for mesh in "${MESHES[@]}"; do
#     echo "Processing: $mesh"
#     "$BENCHMARK_BIN" \
#         -i "$mesh" \
#         -s "$SOLVER" \
#         -a DEFAULT \
#         -d "$DEFAULT_ORDERING_TYPE" \
#         -g 0 \
#         -o "$OUTPUT_CSV"
# done


# # -----------------------------------------------------------------------------
# # Section B: DEFAULT ordering + AMD
# # -----------------------------------------------------------------------------
# DEFAULT_ORDERING_TYPE="AMD"
# OUTPUT_CSV="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/single_factorization/amd_benchmark"
# echo "=== Running DEFAULT ordering + $DEFAULT_ORDERING_TYPE ==="
# for mesh in "${MESHES[@]}"; do
#     echo "Processing: $mesh"
#     "$BENCHMARK_BIN" \
#         -i "$mesh" \
#         -s "$SOLVER" \
#         -a DEFAULT \
#         -d "$DEFAULT_ORDERING_TYPE" \
#         -g 0 \
#         -o "$OUTPUT_CSV"
# done


# # -----------------------------------------------------------------------------
# # Section C: DEFAULT ordering + ParMETIS
# # -----------------------------------------------------------------------------
# DEFAULT_ORDERING_TYPE="ParMETIS"
# OUTPUT_CSV="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/single_factorization/parmetis_benchmark"
# echo "=== Running DEFAULT ordering + $DEFAULT_ORDERING_TYPE ==="
# for mesh in "${MESHES[@]}"; do
#     echo "Processing: $mesh"
#     "$BENCHMARK_BIN" \
#         -i "$mesh" \
#         -s "$SOLVER" \
#         -a DEFAULT \
#         -d "$DEFAULT_ORDERING_TYPE" \
#         -g 0 \
#         -o "$OUTPUT_CSV"
# done

# -----------------------------------------------------------------------------
# Section B: PATCH_ORDERING (patch_type × patch_size × binary_level)
# -----------------------------------------------------------------------------
OUTPUT_CSV="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/single_factorization/patch_METIS_benchmark"
echo "=== Running PATCH_ORDERING ==="
PATCH_TYPES=("rxmesh" "metis_kway")
PATCH_SIZES=(512)
BINARY_LEVELS=(8 10)

for mesh in "${MESHES[@]}"; do
    for patch_type in "${PATCH_TYPES[@]}"; do
        for patch_size in "${PATCH_SIZES[@]}"; do
            for binary_level in "${BINARY_LEVELS[@]}"; do
                echo "Processing: $mesh | patch_type=$patch_type | patch_size=$patch_size | binary_level=$binary_level"
                "$BENCHMARK_BIN" \
                    -i "$mesh" \
                    -s "$SOLVER" \
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
