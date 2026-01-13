#!/usr/bin/env bash

# =============================================================================
# Patch Ordering Ablation Study Script
# =============================================================================
# This script runs a baseline and three ablation studies:
# 0. METIS baseline (baseline.csv)
# 1. Patch size effect (ablation.csv)
# 2. Level effect (level_effect.csv)
# 3. Module selection (module_select.csv)
# =============================================================================

SOLVER="MKL"
INPUT_ROOT="/media/behrooz/FarazHard/Last_Project/BenchmarkMesh/tri-mesh/PatchAnalysis"
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/single_factorization/gpu_ordering_tri_mesh_laplace_benchmark"
OUTPUT_BASE="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/patch_ordering_ablation"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_BASE"

# -----------------------------------------------------------------------------
# Mesh Discovery
# -----------------------------------------------------------------------------
mapfile -t MESHES < <(find "$INPUT_ROOT" -type f -name "*.obj")
echo "Found ${#MESHES[@]} meshes"

if [ ${#MESHES[@]} -eq 0 ]; then
    echo "Error: No meshes found in $INPUT_ROOT"
    exit 1
fi

# =============================================================================
# Baseline: METIS Ordering (baseline.csv)
# =============================================================================
# - Standard METIS ordering as baseline for comparison
# =============================================================================
echo "=== Running Baseline: METIS Ordering ==="
OUTPUT_CSV_BASELINE="${OUTPUT_BASE}/baseline"

for mesh in "${MESHES[@]}"; do
    echo "Processing: $mesh | ordering=METIS"
    "$BENCHMARK_BIN" \
        -i "$mesh" \
        -s "$SOLVER" \
        -a METIS \
        -g 0 \
        -o "$OUTPUT_CSV_BASELINE"
done

echo "=== Baseline complete ==="

# =============================================================================
# Ablation Study 1: Patch Size Effect (ablation.csv)
# =============================================================================
# - Patch sizes: 32, 64, 128, 256, 512
# - Fixed: use_patch_separator=1, patch_type=rxmesh, local_permute_method=amd
# =============================================================================
echo "=== Running Ablation Study 1: Patch Size Effect ==="
OUTPUT_CSV_ABLATION="${OUTPUT_BASE}/patch_size_effect"
PATCH_SIZES=(64 128 256 512)
BINARY_LEVEL=8

for mesh in "${MESHES[@]}"; do
    for patch_size in "${PATCH_SIZES[@]}"; do
        echo "Processing: $mesh | patch_size=$patch_size"
        "$BENCHMARK_BIN" \
            -i "$mesh" \
            -s "$SOLVER" \
            -a PATCH_ORDERING \
            -g 0 \
            -p "rxmesh" \
            -z "$patch_size" \
            -b "$BINARY_LEVEL" \
            -u 1 \
            -m "amd" \
            -o "$OUTPUT_CSV_ABLATION"
    done
done

echo "=== Ablation Study 1 complete ==="

# =============================================================================
# Ablation Study 2: Level Effect (level_effect.csv)
# =============================================================================
# - Binary levels: 2, 4, 8, 10, 12
# - Fixed: patch_size=256, use_patch_separator=1, patch_type=rxmesh, local_permute_method=amd
# =============================================================================
echo "=== Running Ablation Study 2: Level Effect ==="
OUTPUT_CSV_LEVEL="${OUTPUT_BASE}/level_effect"
BINARY_LEVELS=(2 4 8 10 12)
PATCH_SIZE_FIXED=256

for mesh in "${MESHES[@]}"; do
    for binary_level in "${BINARY_LEVELS[@]}"; do
        echo "Processing: $mesh | binary_level=$binary_level"
        "$BENCHMARK_BIN" \
            -i "$mesh" \
            -s "$SOLVER" \
            -a PATCH_ORDERING \
            -g 0 \
            -p "rxmesh" \
            -z "$PATCH_SIZE_FIXED" \
            -b "$binary_level" \
            -u 1 \
            -m "amd" \
            -o "$OUTPUT_CSV_LEVEL"
    done
done

echo "=== Ablation Study 2 complete ==="

# =============================================================================
# Ablation Study 3: Module Selection (module_select.csv)
# =============================================================================
# - Fixed: patch_size=256, binary_level=8
# - Iterate: patch_type=(rxmesh, metis_kway), 
#            use_patch_separator=(0=metis, 1=patch-based),
#            local_permute_method=(amd, metis)
# =============================================================================
echo "=== Running Ablation Study 3: Module Selection ==="
OUTPUT_CSV_MODULE="${OUTPUT_BASE}/module_select"
PATCH_TYPES=("rxmesh" "metis_kway")
USE_PATCH_SEPARATORS=(0 1)
LOCAL_PERMUTE_METHODS=("amd" "metis")
PATCH_SIZE_FIXED=512
BINARY_LEVEL=8
for mesh in "${MESHES[@]}"; do
    for patch_type in "${PATCH_TYPES[@]}"; do
        for use_patch_separator in "${USE_PATCH_SEPARATORS[@]}"; do
            for local_permute_method in "${LOCAL_PERMUTE_METHODS[@]}"; do
                echo "Processing: $mesh | patch_type=$patch_type | use_patch_separator=$use_patch_separator | local_permute=$local_permute_method"
                "$BENCHMARK_BIN" \
                    -i "$mesh" \
                    -s "$SOLVER" \
                    -a PATCH_ORDERING \
                    -g 0 \
                    -p "$patch_type" \
                    -z "$PATCH_SIZE_FIXED" \
                    -b "$BINARY_LEVEL" \
                    -u "$use_patch_separator" \
                    -m "$local_permute_method" \
                    -o "$OUTPUT_CSV_MODULE"
            done
        done
    done
done

echo "=== Ablation Study 3 complete ==="
echo "=== All benchmark ablation studies complete ==="
