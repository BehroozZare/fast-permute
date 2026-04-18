#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# SLIM Benchmark Script
# =============================================================================

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
APPS_ROOT="/media/behrooz/FarazHard/Last_Project/Apps"
INPUT_ROOT="${APPS_ROOT}/SLIM"
OUTPUT_ROOT="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/Apps"
OUTPUT_CSV_BASE="${OUTPUT_ROOT}/slim"
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/multiple_factorization/gpu_ordering_multi_slim_benchmark"

DIM=2
PATCH_TYPE="rxmesh"
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
mapfile -t DATASETS < <(
    for candidate in "${INPUT_ROOT}"/*; do
        [[ -d "${candidate}" ]] || continue
        if compgen -G "${candidate}/tinyAD_*_H_*" >/dev/null && \
           compgen -G "${candidate}/tinyAD_*_g_*" >/dev/null && \
           compgen -G "${candidate}/*.obj" >/dev/null; then
            printf '%s\n' "${candidate}"
        fi
    done
)

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    echo "No SLIM datasets found under ${INPUT_ROOT}"
    exit 1
fi

echo "Found ${#DATASETS[@]} SLIM datasets."

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
            -d "${DIM}" \
            -o "${OUTPUT_CSV_BASE}"
    else
        "${BENCHMARK_BIN}" \
            -k "${dataset}" \
            -s "${solver}" \
            -a "${ordering}" \
            -d "${DIM}" \
            -o "${OUTPUT_CSV_BASE}"
    fi
}

# -----------------------------------------------------------------------------
# Run matrix: solver x ordering x dataset
# -----------------------------------------------------------------------------
for dataset in "${DATASETS[@]}"; do
    for solver in "${SOLVERS[@]}"; do
        for ordering in "${ORDERINGS[@]}"; do
            run_case "${dataset}" "${solver}" "${ordering}"
        done
    done
done

echo "SLIM benchmark complete. CSVs are in ${OUTPUT_ROOT}"

