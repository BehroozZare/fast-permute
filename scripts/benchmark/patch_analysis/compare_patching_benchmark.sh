#!/usr/bin/env bash

# =============================================================================
# Patch Metrics / Clustering Quality Benchmark Script
# =============================================================================
# Drives the compare_patching binary across a small grid of options so we can
# correlate graph clustering quality (edge cut, conductance, boundary ratio,
# patch balance, per-level separator sizes) with separator size, fill-in, and
# runtime. Output lives in a single CSV per study under $OUTPUT_BASE.
# =============================================================================

set -u

SOLVER="CHOLMOD"
INPUT_ROOT="/media/behrooz/FarazHard/Last_Project/BenchmarkMesh/tri-mesh/PatchAnalysis"
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/patch_metrics/gpu_ordering_compare_patching"
OUTPUT_BASE="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/patch_metrics"

mkdir -p "$OUTPUT_BASE"

# -----------------------------------------------------------------------------
# Mesh Discovery
# -----------------------------------------------------------------------------
mapfile -t MESHES < <(find "$INPUT_ROOT" -type f \( -iname "*.obj" -o -iname "*.off" \))
echo "Found ${#MESHES[@]} meshes"

if [ ${#MESHES[@]} -eq 0 ]; then
    echo "Error: No meshes found in $INPUT_ROOT"
    exit 1
fi

# =============================================================================
# Study 1: Patch Size Sweep (primary)
# =============================================================================
# Sweeps patch_size in {64, 128, 256, 512, 1024} for PATCH_ORDERING so we can
# plot clustering-quality metrics vs separator size / fill-in / runtime on the
# same x-axis (directly addresses the reviewer comment on Figs 9/10/11).
# Fixed: patch_type=rxmesh, binary_level=8, use_patch_separator=1, local_permute=amd.
# =============================================================================
echo "=== Study 1: Patch Size Sweep (PATCH_ORDERING, rxmesh) ==="
OUT_PATCH_SIZE="${OUTPUT_BASE}/patch_size_sweep"
PATCH_SIZES=(64 128 256 512 1024)
BINARY_LEVEL_FIXED=8

for mesh in "${MESHES[@]}"; do
    for patch_size in "${PATCH_SIZES[@]}"; do
        echo "[Study 1] $mesh | patch_size=$patch_size"
        "$BENCHMARK_BIN" \
            -i "$mesh" \
            -s "$SOLVER" \
            -a PATCH_ORDERING \
            -g 0 \
            -p "rxmesh" \
            -z "$patch_size" \
            -b "$BINARY_LEVEL_FIXED" \
            -u 1 \
            -m "amd" \
            -o "$OUT_PATCH_SIZE"
    done
done

# =============================================================================
# Study 2: Patch Type Comparison (secondary)
# =============================================================================
# Compares the two patch producers at a fixed patch_size / binary_level. Useful
# for showing whether clustering-quality differences explain fill-in/runtime
# differences between rxmesh patches and METIS k-way patches.
# =============================================================================
echo "=== Study 2: Patch Type Comparison ==="
OUT_PATCH_TYPE="${OUTPUT_BASE}/patch_type_compare"
PATCH_TYPES=("rxmesh" "metis_kway")
PATCH_SIZE_FIXED=256

for mesh in "${MESHES[@]}"; do
    for patch_type in "${PATCH_TYPES[@]}"; do
        echo "[Study 2] $mesh | patch_type=$patch_type"
        "$BENCHMARK_BIN" \
            -i "$mesh" \
            -s "$SOLVER" \
            -a PATCH_ORDERING \
            -g 0 \
            -p "$patch_type" \
            -z "$PATCH_SIZE_FIXED" \
            -b "$BINARY_LEVEL_FIXED" \
            -u 1 \
            -m "amd" \
            -o "$OUT_PATCH_TYPE"
    done
done

# =============================================================================
# Study 3: Baselines (METIS and PARTH)
# =============================================================================
# Gives non-patch reference points so the PATCH_ORDERING sweep numbers can be
# placed on the same figure / table. METIS does not expose getPatch, so its
# clustering-quality columns will be empty -- that's fine, it only contributes
# separator-size + fill-in + runtime.
# =============================================================================
echo "=== Study 3: Baselines (METIS, PARTH) ==="
OUT_BASELINE="${OUTPUT_BASE}/baselines"

for mesh in "${MESHES[@]}"; do
    echo "[Study 3] $mesh | ordering=METIS"
    "$BENCHMARK_BIN" \
        -i "$mesh" \
        -s "$SOLVER" \
        -a METIS \
        -g 0 \
        -o "$OUT_BASELINE"

    echo "[Study 3] $mesh | ordering=PARTH | binary_level=$BINARY_LEVEL_FIXED"
    "$BENCHMARK_BIN" \
        -i "$mesh" \
        -s "$SOLVER" \
        -a PARTH \
        -g 0 \
        -b "$BINARY_LEVEL_FIXED" \
        -o "$OUT_BASELINE"
done

echo "=== All patch-metrics studies complete ==="
echo "CSVs written under: $OUTPUT_BASE"
