#!/usr/bin/env bash
# ------------------------------------------------------------------
# Loop matrix_example over every .mtx file recursively under input/
# and write a per-matrix JSON result to
# results/<solver>/<relpath>/<name>_<precision>.json, mirroring the input
# directory layout (so groups like input/HB/foo.mtx end up in
# results/<solver>/HB/foo_double.json and foo_float.json).
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
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: input directory not found: \"$INPUT_DIR\""
    exit 1
fi

mtx_files=()
while IFS= read -r -d '' f; do
    mtx_files+=("$f")
done < <(find "$INPUT_DIR" -type f -name "*.mtx" -print0 | LC_ALL=C sort -z)

if [ "${#mtx_files[@]}" -eq 0 ]; then
    echo "ERROR: no .mtx files found under \"$INPUT_DIR\""    
    exit 1
fi

mkdir -p "$OUT_DIR"

COUNT=0

for f in "${mtx_files[@]}"; do
    COUNT=$((COUNT + 1))
    rel="${f#"$INPUT_DIR"/}"
    rel_dir="$(dirname "$rel")"
    name="$(basename "$f" .mtx)"

    if [ "$rel_dir" = "." ]; then
        out_subdir="$OUT_DIR"
    else
        out_subdir="$OUT_DIR/$rel_dir"
        mkdir -p "$out_subdir"
    fi

    echo
    echo "============================================================"
    echo "  Matrix: $rel  |  solver: $SOLVER  |  patch: $PATCH_SIZE"
    echo "============================================================"
    "$BINARY" -i "$f" -s "$SOLVER" -p "$PATCH_SIZE" --precision double --make-spd-from-pattern --out "$out_subdir/${name}_double.json"
    "$BINARY" -i "$f" -s "$SOLVER" -p "$PATCH_SIZE" --precision float --make-spd-from-pattern --out "$out_subdir/${name}_float.json"
done

echo
echo "============================================================"
echo "Done. Processed $COUNT matrices"
echo "Results written to \"$OUT_DIR\""
echo "============================================================"
