#!/usr/bin/env bash
# Usage:
#     scripts/examples/run_cudss_example.sh
#     scripts/examples/run_cudss_example.sh /path/to/thingi10k/meshes
#     scripts/examples/run_cudss_example.sh /path/to/thingi10k/meshes 1024
#     scripts/examples/run_cudss_example.sh /path/to/thingi10k/meshes 1024 results/cudss

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
INPUT_DIR="${1:-$REPO_ROOT/input/meshes}"
PATCH_SIZE="${2:-512}"
OUT_DIR="${3:-results/cudss}"
BINARY="$REPO_ROOT/build/bin/cudss_example"

if [ ! -x "$BINARY" ]; then
    echo "ERROR: cudss_example not found at \"$BINARY\""
    exit 1
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: mesh directory not found: \"$INPUT_DIR\""
    exit 1
fi

INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"

mesh_files=()
while IFS= read -r -d '' f; do
    mesh_files+=("$f")
done < <(find "$INPUT_DIR" -type f \( -iname "*.obj" -o -iname "*.off" -o -iname "*.ply" -o -iname "*.stl" \) -print0 | LC_ALL=C sort -z)

if [ "${#mesh_files[@]}" -eq 0 ]; then
    echo "ERROR: no mesh files found under \"$INPUT_DIR\""
    exit 1
fi

mkdir -p "$OUT_DIR"

COUNT=0

for f in "${mesh_files[@]}"; do
    COUNT=$((COUNT + 1))
    rel="${f#"$INPUT_DIR"/}"
    rel_dir="$(dirname "$rel")"
    base="$(basename "$f")"
    name="${base%.*}"

    if [ "$rel_dir" = "." ]; then
        out_subdir="$OUT_DIR"
    else
        out_subdir="$OUT_DIR/$rel_dir"
        mkdir -p "$out_subdir"
    fi

    echo
    echo "============================================================"
    echo "  Mesh: $rel  |  solver: cudss  |  patch: $PATCH_SIZE"
    echo "============================================================"
    "$BINARY" -i "$f" -r 5 -p "$PATCH_SIZE" --precision double --out "$out_subdir/${name}_double.json"
    "$BINARY" -i "$f" -r 5 -p "$PATCH_SIZE" --precision float --out "$out_subdir/${name}_float.json"
done

echo
echo "============================================================"
echo "Done. Processed $COUNT meshes"
echo "Results written to \"$OUT_DIR\""
echo "============================================================"
