#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Mesh Smoothing Benchmark Script
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APPS_ROOT="/media/behrooz/FarazHard/Last_Project/Apps"
INPUT_ROOT="${APPS_ROOT}/mesh_smoothing"
OUTPUT_ROOT="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/Apps"
OUTPUT_CSV_BASE="${OUTPUT_ROOT}/smoothing"
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/multiple_factorization/gpu_ordering_multi_mesh_smoothing_benchmark"

NUM_ITERATIONS=10
VISUALIZE=0
USE_GPU=0
PATCH_TYPE="rxmesh"
PATCH_SIZE=512
BINARY_LEVEL=10
SOLVERS=("CUDSS" "MKL")
ORDERINGS=("DEFAULT" "PATCH_ORDERING")

mkdir -p "${OUTPUT_ROOT}"
rm -f "${OUTPUT_CSV_BASE}.csv"

if [[ ! -x "${BENCHMARK_BIN}" ]]; then
    echo "Benchmark binary not found or not executable: ${BENCHMARK_BIN}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Dataset discovery
# -----------------------------------------------------------------------------
shopt -s nullglob
MESHES=("${INPUT_ROOT}"/*.obj)
shopt -u nullglob

if [[ ${#MESHES[@]} -eq 0 ]]; then
    echo "No mesh_smoothing .obj files found under ${INPUT_ROOT}"
    exit 1
fi

echo "Found ${#MESHES[@]} mesh_smoothing datasets."

run_case() {
    local mesh="$1"
    local solver="$2"
    local ordering="$3"

    local mesh_name
    mesh_name="$(basename "${mesh}" .obj)"
    echo "Running: mesh=${mesh_name} solver=${solver} ordering=${ordering}"
    if [[ "${ordering}" == "PATCH_ORDERING" ]]; then
        "${BENCHMARK_BIN}" \
            -i "${mesh}" \
            -n "${NUM_ITERATIONS}" \
            -v "${VISUALIZE}" \
            -s "${solver}" \
            -a "${ordering}" \
            -g "${USE_GPU}" \
            -p "${PATCH_TYPE}" \
            -z "${PATCH_SIZE}" \
            -b "${BINARY_LEVEL}" \
            -o "${OUTPUT_CSV_BASE}"
    else
        "${BENCHMARK_BIN}" \
            -i "${mesh}" \
            -n "${NUM_ITERATIONS}" \
            -v "${VISUALIZE}" \
            -s "${solver}" \
            -a "${ordering}" \
            -g "${USE_GPU}" \
            -o "${OUTPUT_CSV_BASE}"
    fi
}

# -----------------------------------------------------------------------------
# Run matrix: solver x ordering x dataset
# -----------------------------------------------------------------------------
for mesh in "${MESHES[@]}"; do
    for solver in "${SOLVERS[@]}"; do
        for ordering in "${ORDERINGS[@]}"; do
            run_case "${mesh}" "${solver}" "${ordering}"
        done
    done
done

echo "Mesh smoothing benchmark complete. CSVs are in ${OUTPUT_ROOT}"
