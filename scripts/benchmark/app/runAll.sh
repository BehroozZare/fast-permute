#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SCRIPTS=(
    "data_smoothing_benchmark.sh"
    "mesh_smoothing_benchmark.sh"
    "scp_benchmark.sh"
    "slim_benchmark.sh"
)

for script in "${SCRIPTS[@]}"; do
    echo "============================================================"
    echo "Running ${script}"
    echo "============================================================"
    "${SCRIPT_DIR}/${script}"
done

echo "All app benchmarks finished."
