#!/usr/bin/env bash
# ------------------------------------------------------------------
# Loop matrix_example over every .mtx file in input/ and write a
# per-matrix JSON result to results/<solver>/<name>.json.
#
# Usage:
#     scripts/examples/run_spd_benchmark.sh                (default solver: cudss)
#     scripts/examples/run_spd_benchmark.sh mkl
#     scripts/examples/run_spd_benchmark.sh cholmod
#     scripts/examples/run_spd_benchmark.sh cudss 1024     (override patch size)
# ------------------------------------------------------------------

set -u

SOLVER="${1:-cudss}"
PATCH_SIZE="${2:-512}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR/../.."
INPUT_DIR="$REPO_ROOT/input"
BINARY="$REPO_ROOT/build/bin/matrix_example"
OUT_DIR="results/$SOLVER"

if [ ! -x "$BINARY" ]; then
    echo "ERROR: matrix_example not found at \"$BINARY\""
    echo "Build it first with:"
    echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \\"
    echo "        -DHOMA_BUILD_EXAMPLE=ON -DHOMA_WITH_CHOLMOD=ON \\"
    echo "        -DHOMA_WITH_MKL=ON -DHOMA_WITH_CUDSS=ON"
    echo "  cmake --build build --config Release"
    exit 1
fi

shopt -s nullglob
mtx_files=("$INPUT_DIR"/*.mtx)
shopt -u nullglob

if [ "${#mtx_files[@]}" -eq 0 ]; then
    echo "ERROR: no .mtx files found in \"$INPUT_DIR\""
    echo "Run scripts/download_suitesparse_spd.py first to populate it."
    exit 1
fi

mkdir -p "$OUT_DIR"

COUNT=0

for f in "${mtx_files[@]}"; do
    COUNT=$((COUNT + 1))
    name="$(basename "$f" .mtx)"
    echo
    echo "============================================================"
    echo "  Matrix: $(basename "$f")  |  solver: $SOLVER  |  patch: $PATCH_SIZE"
    echo "============================================================"
    "$BINARY" -i "$f" -s "$SOLVER" -p "$PATCH_SIZE" --out "$OUT_DIR/${name}.json"
done

echo
echo "============================================================"
echo "Done. Processed $COUNT matrices"
echo "Results written to \"$OUT_DIR\""
echo "============================================================"
