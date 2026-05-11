#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MESHES_DIR="$REPO_ROOT/examples/meshes"

BINARY="$(find "$REPO_ROOT/build" -type f -name "mkl_example" 2>/dev/null | head -1)"
if [ -z "$BINARY" ]; then
    echo "ERROR: mkl_example binary not found under $REPO_ROOT/build/"
    echo "Build it first with:"
    echo "  cmake --preset release -DHOMA_BUILD_BENCHMARKS=ON -DHOMA_EXAMPLE_WITH_MKL=ON"
    echo "  cmake --build --preset release"
    exit 1
fi

for mesh in "$MESHES_DIR"/*.obj; do
    name="$(basename "$mesh")"
    echo ""
    echo "============================================================"
    echo "  Mesh: $name"
    echo "============================================================"
    "$BINARY" -i "$mesh"
done
