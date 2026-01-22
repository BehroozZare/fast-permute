l#!/usr/bin/env bash

# =============================================================================
# CUDSS Ordering Benchmark Script
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SOLVERS=(CUDSS MKL)
INPUT_ROOT="/media/behrooz/FarazHard/Last_Project/BenchmarkMesh/tetMeshes/mesh"
OUTPUT_CSV="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/single_factorization/tet_mesh_benchmark"
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/single_factorization/gpu_ordering_tet_mesh_laplace_benchmark"

# -----------------------------------------------------------------------------
# Mesh Discovery
# -----------------------------------------------------------------------------
mapfile -t MESHES < <(find "$INPUT_ROOT" -type f -name "*.mesh")
echo "Found ${#MESHES[@]} meshes"

# -----------------------------------------------------------------------------
# Iterate over solvers
# -----------------------------------------------------------------------------
for SOLVER in "${SOLVERS[@]}"; do
    echo ""
    echo "############################################################"
    echo "# Running benchmarks with solver: $SOLVER"
    echo "############################################################"

    # -------------------------------------------------------------------------
    # Section A: DEFAULT ordering
    # -------------------------------------------------------------------------
    echo "=== Running DEFAULT ordering ($SOLVER) ==="
    for mesh in "${MESHES[@]}"; do
        echo "Processing: $mesh"
        "$BENCHMARK_BIN" \
            -i "$mesh" \
            -s "$SOLVER" \
            -a DEFAULT \
            -g 0 \
            -o "$OUTPUT_CSV"
    done

    # # -------------------------------------------------------------------------
    # # Section B: PARTH ordering (binary_level: 8, 10)
    # # -------------------------------------------------------------------------
    # echo "=== Running PARTH ordering ($SOLVER) ==="
    # for mesh in "${MESHES[@]}"; do
    #     for binary_level in 8 10; do
    #         echo "Processing: $mesh | binary_level=$binary_level"
    #         "$BENCHMARK_BIN" \
    #             -i "$mesh" \
    #             -s "$SOLVER" \
    #             -a PARTH \
    #             -g 0 \
    #             -b "$binary_level" \
    #             -o "$OUTPUT_CSV"
    #     done
    # done

    # -------------------------------------------------------------------------
    # Section C: PATCH_ORDERING (patch_size × binary_level)
    # -------------------------------------------------------------------------
    echo "=== Running PATCH_ORDERING ($SOLVER) ==="
    PATCH_SIZES=(128 256)
    BINARY_LEVELS=(8 10)

    for mesh in "${MESHES[@]}"; do
        for patch_size in "${PATCH_SIZES[@]}"; do
            for binary_level in "${BINARY_LEVELS[@]}"; do
                echo "Processing: $mesh | patch_size=$patch_size | binary_level=$binary_level"
                "$BENCHMARK_BIN" \
                    -i "$mesh" \
                    -s "$SOLVER" \
                    -a PATCH_ORDERING \
                    -g 0 \
                    -z "$patch_size" \
                    -b "$binary_level" \
                    -o "$OUTPUT_CSV"
            done
        done
    done

done

echo "=== Benchmark complete ==="
