#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Inverse Rendering Benchmark - Smoke Test (dragon_small only)
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APPS_ROOT="/media/behrooz/FarazHard/Last_Project/Apps"
DATASET="${APPS_ROOT}/inverse_rendering/dragon_small"
OUTPUT_ROOT="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/Apps"
OUTPUT_CSV_BASE="${OUTPUT_ROOT}/inverse_rendering_smoke"
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/multiple_factorization/gpu_ordering_inverse_rendering_benchmark"

PATCH_TYPE="rxmesh"
BINARY_LEVEL=10
MIN_VERTICES=50000
SOLVERS=("CUDSS")
ORDERINGS=("DEFAULT" "PATCH_ORDERING" "PARTH")

mkdir -p "${OUTPUT_ROOT}"
rm -f "${OUTPUT_CSV_BASE}.csv"

if [[ ! -x "${BENCHMARK_BIN}" ]]; then
    echo "Benchmark binary not found or not executable: ${BENCHMARK_BIN}"
    exit 1
fi

if [[ ! -f "${DATASET}/counts.csv" ]]; then
    echo "dragon_small dataset not found (missing counts.csv): ${DATASET}"
    exit 1
fi

echo "Smoke-test dataset: ${DATASET}"

run_case() {
    local dataset="$1"
    local solver="$2"
    local ordering="$3"

    local dataset_name
    dataset_name="$(basename "${dataset}")"
    echo "Running: dataset=${dataset_name} solver=${solver} ordering=${ordering}"
    if [[ "${ordering}" == "PATCH_ORDERING" ]]; then
        "${BENCHMARK_BIN}" \
            -k "${dataset}" \
            -s "${solver}" \
            -a "${ordering}" \
            -p "${PATCH_TYPE}" \
            -b "${BINARY_LEVEL}" \
            -m "${MIN_VERTICES}" \
            -o "${OUTPUT_CSV_BASE}"
    else
        "${BENCHMARK_BIN}" \
            -k "${dataset}" \
            -s "${solver}" \
            -a "${ordering}" \
            -m "${MIN_VERTICES}" \
            -o "${OUTPUT_CSV_BASE}"
    fi
}

# -----------------------------------------------------------------------------
# Run matrix: single dataset x solver x ordering
# -----------------------------------------------------------------------------
for solver in "${SOLVERS[@]}"; do
    for ordering in "${ORDERINGS[@]}"; do
        run_case "${DATASET}" "${solver}" "${ordering}"
    done
done

echo "Inverse rendering smoke benchmark complete. CSV: ${OUTPUT_CSV_BASE}.csv"
