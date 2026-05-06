#!/usr/bin/env bash

# =============================================================================
# CUDSS DEFAULT-Ordering Benchmark Script (MT vs Single-Threaded A/B)
# =============================================================================

set -u

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SOLVER="CUDSS"  # Options: CUDSS, MKL
INPUT_ROOT="/media/behrooz/FarazHard/Last_Project/Apps/tri-meshes"
OUTPUT_CSV="/home/behrooz/Desktop/Last_Project/gpu_ordering/output/single_factorization/tri_CUDSS_benchmark"
BENCHMARK_BIN="/home/behrooz/Desktop/Last_Project/gpu_ordering/cmake-build-release/benchmark/single_factorization/gpu_ordering_tri_mesh_laplace_benchmark"

# cuDSS threading-layer shared library (provided by the cuDSS distribution).
# Override CUDSS_MT_LAYER from the environment if your install location differs.
CUDSS_MT_LAYER="${CUDSS_MT_LAYER:-/usr/lib/x86_64-linux-gnu/libcudss/12/libcudss_mtlayer_gomp.so}"

# Modes to A/B benchmark. "off" disables MT; "mt<N>" enables MT with N threads.
# Edit this list to try other thread counts (e.g. add "mt16").
MT_MODES=("mt10")

# -----------------------------------------------------------------------------
# Mesh Discovery
# -----------------------------------------------------------------------------
mapfile -t MESHES < <(find "$INPUT_ROOT" -type f -name "*.obj")
echo "Found ${#MESHES[@]} meshes"

if [[ ${#MESHES[@]} -eq 0 ]]; then
    echo "No meshes found under $INPUT_ROOT - aborting." >&2
    exit 1
fi

if [[ ! -x "$BENCHMARK_BIN" ]]; then
    echo "Benchmark binary not found or not executable: $BENCHMARK_BIN" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Section A: DEFAULT ordering (MT mode A/B)
# -----------------------------------------------------------------------------
for MODE in "${MT_MODES[@]}"; do
    case "$MODE" in
        off)
            ENV_PREFIX=(env \
                -u CUDSS_THREADING_LIB \
                -u RX_CUDSS_THREADING_LIB \
                -u RX_CUDSS_HOST_NTHREADS \
                -u OMP_NUM_THREADS)
            ;;
        mt*)
            N="${MODE#mt}"
            if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -le 0 ]]; then
                echo "Skipping invalid mode '$MODE' (expected mt<positive int>)" >&2
                continue
            fi
            if [[ ! -f "$CUDSS_MT_LAYER" ]]; then
                echo "Skipping mode '$MODE': threading layer not found at $CUDSS_MT_LAYER" >&2
                continue
            fi
            ENV_PREFIX=(env \
                "CUDSS_THREADING_LIB=$CUDSS_MT_LAYER" \
                "OMP_NUM_THREADS=$N" \
                "RX_CUDSS_HOST_NTHREADS=$N")
            ;;
        *)
            echo "Skipping unknown mode '$MODE'" >&2
            continue
            ;;
    esac

    MODE_CSV="${OUTPUT_CSV}_${MODE}"
    echo ""
    echo "############################################################"
    echo "# DEFAULT ordering | solver=$SOLVER | mode=$MODE"
    echo "# csv=${MODE_CSV}.csv"
    echo "############################################################"

    for mesh in "${MESHES[@]}"; do
        echo "Processing: $mesh"
        "${ENV_PREFIX[@]}" "$BENCHMARK_BIN" \
            -i "$mesh" \
            -s "$SOLVER" \
            -a DEFAULT \
            -g 0 \
            -o "$MODE_CSV"
    done
done

echo ""
echo "=== Benchmark complete ==="
echo "Per-mode CSVs:"
for MODE in "${MT_MODES[@]}"; do
    echo "  ${OUTPUT_CSV}_${MODE}.csv"
done

# # -----------------------------------------------------------------------------
# # Section B: PARTH ordering (binary_level: 8, 10) - DISABLED
# # -----------------------------------------------------------------------------
#  echo "=== Running PARTH ordering ==="
#  for mesh in "${MESHES[@]}"; do
#      for binary_level in 8 10; do
#          echo "Processing: $mesh | binary_level=$binary_level"
#          "$BENCHMARK_BIN" \
#              -i "$mesh" \
#              -s "$SOLVER" \
#              -a PARTH \
#              -g 0 \
#              -b "$binary_level" \
#              -o "$OUTPUT_CSV"
#      done
#  done

# # -----------------------------------------------------------------------------
# # Section C: PATCH_ORDERING (patch_type x patch_size x binary_level) - DISABLED
# # -----------------------------------------------------------------------------
#  echo "=== Running PATCH_ORDERING ==="
#  PATCH_TYPES=("rxmesh" "metis_kway")
#  PATCH_SIZES=(512)
#  BINARY_LEVELS=(8 10)
#
#  for mesh in "${MESHES[@]}"; do
#      for patch_type in "${PATCH_TYPES[@]}"; do
#          for patch_size in "${PATCH_SIZES[@]}"; do
#              for binary_level in "${BINARY_LEVELS[@]}"; do
#                  echo "Processing: $mesh | patch_type=$patch_type | patch_size=$patch_size | binary_level=$binary_level"
#                  "$BENCHMARK_BIN" \
#                      -i "$mesh" \
#                      -s "$SOLVER" \
#                      -a PATCH_ORDERING \
#                      -g 0 \
#                      -p "$patch_type" \
#                      -z "$patch_size" \
#                      -b "$binary_level" \
#                      -o "$OUTPUT_CSV"
#              done
#          done
#      done
#  done
