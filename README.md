# GPU Ordering

A GPU-accelerated fill-reducing ordering library for sparse linear systems. This project implements and benchmarks various fill-reducing ordering algorithms, with a focus on patch-based nested dissection methods that leverage GPU parallelism.

## Quick Start

### Prerequisites

- CMake 3.18+
- CUDA Toolkit (with cuSPARSE, cuSOLVER)
- GCC with C++17 support
- OpenMP

### Build

```bash
# Create build directory
mkdir cmake-build-release && cd cmake-build-release

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)
```

Or use the provided build script:

```bash
./scripts/build.sh
```

### Run Benchmark

```bash
./cmake-build-release/benchmark/gpu_ordering_benchmark \
  -i /path/to/mesh.obj \
  -s CUDSS \
  -a PATCH_ORDERING \
  -g 0 \
  -p metis_kway
```

This example:
- Loads a triangle mesh from an OBJ file
- Constructs a cotangent Laplacian matrix
- Applies the `PATCH_ORDERING` fill-reducing ordering using METIS k-way partitioning for patches
- Solves the system using the NVIDIA cuDSS GPU solver

## Command Line Options

| Option | Description | Values |
|--------|-------------|--------|
| `-i, --input` | Input mesh file | Path to OBJ file |
| `-s, --solver` | Linear solver | `CHOLMOD`, `CUDSS`, `PARTH_SOLVER`, `STRUMPACK` |
| `-a, --ordering` | Ordering algorithm | `DEFAULT`, `METIS`, `RXMESH_ND`, `PATCH_ORDERING`, `PARTH`, `NEUTRAL` |
| `-g, --use_gpu` | Use GPU for patch ordering | `0` (CPU), `1` (GPU) |
| `-p, --patch_type` | Patch generation method | `rxmesh`, `metis_kway` |
| `-o, --output` | Output folder | Path to output directory |

### Solver Types

- **CHOLMOD**: CPU-based Cholesky solver from SuiteSparse
- **CUDSS**: NVIDIA cuDSS GPU sparse direct solver
- **PARTH_SOLVER**: Parth direct solver
- **STRUMPACK**: Sparse direct solver with GPU support (requires MPI)

### Ordering Types

- **DEFAULT**: Use the solver's default ordering
- **METIS**: METIS nested dissection ordering
- **RXMESH_ND**: RXMesh-based nested dissection
- **PATCH_ORDERING**: Patch-based fill-reducing ordering (main algorithm under development)
- **PARTH**: Parth ordering algorithm
- **NEUTRAL**: Identity permutation (no reordering)

## Project Structure

```
gpu_ordering/
├── benchmark/           # Benchmark executables
├── cmake/               # CMake configuration and recipes
├── input/               # Sample mesh files
├── output/              # Output directory
├── scripts/             # Build and utility scripts
└── src/                 # Source code
    ├── AnalysisScripts/ # Python analysis and plotting scripts
    ├── LinSysSolvers/   # Linear solver wrappers
    ├── Ordering/        # Fill-reducing ordering algorithms
    ├── PatchOrdering/   # Patch-based ordering implementation
    └── utils/           # Helper utilities
```

### src/AnalysisScripts/

Python scripts for analysis and visualization:

- **FactorVisualization/see_factor.py**: Visualizes sparse matrix sparsity patterns using spy plots. Compares factor structures between different orderings.
- **OrderingAnalysis/see_ratio.py**: Compares fill-ratios between ordering methods. Generates normalized comparison plots against METIS baseline.

### src/LinSysSolvers/

Wrappers for sparse direct linear solvers. Entry point: `LinSysSolver.hpp`

```cpp
// Create a solver instance
LinSysSolver* solver = LinSysSolver::create(LinSysSolverType::GPU_CUDSS);

// Set the sparse matrix (CSC format)
solver->setMatrix(outerIndexPtr, innerIndexPtr, values, numRows, numNonZeros);

// Symbolic analysis with user-defined permutation
solver->analyze_pattern(permutation, etree);

// Numerical factorization
solver->factorize();

// Solve Ax = b
solver->solve(rhs, result);
```

**Files:**
| File | Description |
|------|-------------|
| `LinSysSolver.hpp` | Abstract base class and factory (`create()` function) |
| `LinSysSolver.cpp` | Factory implementation |
| `CHOLMODSolver.hpp/cpp` | SuiteSparse CHOLMOD wrapper |
| `CUDSSSolver.hpp/cu` | NVIDIA cuDSS GPU solver wrapper |
| `STRUMPACKSolver.hpp/cpp` | STRUMPACK solver wrapper |

### src/Ordering/

Fill-reducing ordering algorithms. Entry point: `ordering.h`

```cpp
// Create an ordering instance
Ordering* ordering = Ordering::create(DEMO_ORDERING_TYPE::PATCH_ORDERING);

// Configure options
ordering->setOptions({{"use_gpu", "1"}, {"patch_type", "metis_kway"}});

// Set the adjacency graph (CSR format, without diagonal)
ordering->setGraph(Gp, Gi, numNodes, numEdges);

// Optionally provide mesh data for mesh-aware orderings
if (ordering->needsMesh()) {
    ordering->setMesh(V_data, V_rows, V_cols, F_data, F_rows, F_cols);
}

// Initialize and compute permutation
ordering->init();
ordering->compute_permutation(perm, etree, compute_etree);
```

**Files:**
| File | Description |
|------|-------------|
| `ordering.h` | Abstract base class and factory (`create()` function) |
| `ordering.cu` | Factory implementation |
| `metis_ordering.cpp/h` | METIS nested dissection wrapper |
| `parth_ordering.cpp/h` | Parth ordering wrapper |
| `rxmesh_ordering.cu/h` | RXMesh-based nested dissection |
| `patch_ordering.cu/h` | Patch-based ordering (main algorithm) |
| `neutral_ordering.cpp/h` | Identity permutation |

### src/PatchOrdering/

The core patch-based fill-reducing ordering algorithm. This is the main algorithm under development.

**Algorithm Overview:**
1. Partition the mesh/graph into patches (using RXMesh or METIS k-way)
2. Build a quotient graph where each patch is a supernode
3. Recursively bisect the quotient graph to create a decomposition tree
4. Find separators using bipartite matching refinement
5. Apply local fill-reducing ordering (AMD) within each tree node
6. Assemble the final permutation in post-order

**Files:**
| File | Description |
|------|-------------|
| `cpu_ordering_with_patch.cpp/h` | CPU implementation of patch-based ordering |
| `gpu_ordering_with_patch.cu/h` | GPU-accelerated implementation |

**Key Classes:**
- `GPUOrdering_PATCH`: Main GPU ordering class with decomposition tree management
- `DecompositionTree`: Binary tree representing the nested dissection structure
- `QuotientGraph`: Compressed graph where patches are supernodes

### src/utils/

Helper utilities for benchmarking and development:

| File | Description |
|------|-------------|
| `check_valid_permutation.cpp/h` | Validates that a permutation is valid (bijection) |
| `compute_inverse_perm.cpp/h` | Computes inverse permutation |
| `get_factor_nnz.cpp/h` | Estimates Cholesky factor non-zeros for a given ordering |
| `remove_diagonal.cpp/h` | Removes diagonal entries from sparse matrix (for graph representation) |
| `metis_helper.cpp/h` | METIS utility functions |
| `min_vertex_cover_bipartite.cpp/h` | Hopcroft-Karp algorithm for bipartite matching |
| `cuda_error_handler.h` | CUDA error checking macros |
| `nvtx_helper.h` | NVIDIA Tools Extension for profiling |

### benchmark/

Benchmark executables:

- **cudss_benchmark/cudss_benchmark.cpp**: Main benchmark that:
  1. Loads a triangle mesh
  2. Constructs a cotangent Laplacian matrix
  3. Applies a fill-reducing ordering
  4. Solves the linear system
  5. Reports timing and fill-ratio metrics

- **ordering_benchmark.cpp**: Ordering-focused benchmark for comparing fill-ratios across different ordering methods

- **etree_analysis/etree_analysis.cpp**: Elimination tree analysis benchmark for studying tree structure properties

## Running the Benchmark Script

The project includes a comprehensive benchmark script that automates testing across multiple ordering algorithms and configurations.

### Script Location

```bash
scripts/benchmark/cudss_ordering_benchmark.sh
```

### Configuration

Before running the script, you **must** configure three variables at the top of the script:

| Variable | Description | Example |
|----------|-------------|---------|
| `INPUT_ROOT` | Directory containing your `.obj` mesh files | `/path/to/meshes` |
| `OUTPUT_CSV` | Path prefix for the output CSV file (without `.csv` extension) | `/path/to/output/results` |
| `BENCHMARK_BIN` | Path to the compiled benchmark binary | `/path/to/cmake-build-release/benchmark/gpu_ordering_cudss_benchmark` |

Edit the script to set these paths:

```bash
INPUT_ROOT="/path/to/your/mesh/directory"
OUTPUT_CSV="/path/to/output/benchmark_results"
BENCHMARK_BIN="/path/to/gpu_ordering/cmake-build-release/benchmark/gpu_ordering_cudss_benchmark"
```

### What the Script Benchmarks

The script automatically discovers all `.obj` files in `INPUT_ROOT` and runs three categories of benchmarks:

1. **DEFAULT Ordering**: Uses the solver's built-in ordering for each mesh

2. **PARTH Ordering**: Tests the Parth algorithm with varying `binary_level` values:
   - `binary_level`: 2, 4, 6, 8, 10

3. **PATCH_ORDERING**: Tests patch-based ordering with all combinations of:
   - `patch_type`: `metis_kway`, `rxmesh`
   - `patch_size`: 64, 256, 512

### Running the Benchmark

```bash
# Make the script executable (if needed)
chmod +x scripts/benchmark/cudss_ordering_benchmark.sh

# Run the benchmark
./scripts/benchmark/cudss_ordering_benchmark.sh
```

### Output

Results are appended to `${OUTPUT_CSV}.csv`, containing timing and fill-ratio metrics for each configuration tested

## Build Options

CMake options for enabling/disabling features:

```cmake
-Dgpu_ordering_WITH_PARTH=ON       # Enable Parth ordering (default: ON)
-Dgpu_ordering_WITH_SUITESPARSE=ON # Enable SuiteSparse/CHOLMOD (default: ON)
-Dgpu_ordering_WITH_STRUMPACK=OFF  # Enable STRUMPACK (default: OFF, requires MPI)
-Dgpu_ordering_WITH_PROFILE=ON     # Enable profiling support (default: ON)
```

## Cross-Platform Build Instructions

The project uses CMake recipes that automatically detect and configure BLAS/LAPACK for SuiteSparse. The detection order is:

1. Pre-set `BLAS_LIBRARIES`/`LAPACK_LIBRARIES` (command line override)
2. Intel OneAPI MKL
3. OpenBLAS (system-installed)
4. Any system BLAS/LAPACK (e.g., Apple Accelerate on macOS)
5. Build OpenBLAS from source (fallback)

### Linux

On most Linux distributions, the build should work out of the box if you have a system BLAS/LAPACK installed:

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# Fedora/RHEL
sudo dnf install openblas-devel
```

### macOS

macOS includes the Accelerate framework which provides BLAS/LAPACK. No additional setup is required.

### Windows

On Windows, the library naming conventions differ from Linux. Pre-built OpenBLAS binaries typically use 32-bit integers and have different file names.

#### Option 1: Using Pre-built OpenBLAS (Recommended)

1. Download pre-built OpenBLAS from [GitHub Releases](https://github.com/OpenMathLib/OpenBLAS/releases)
2. Extract to a known location (e.g., `C:\OpenBLAS`)
3. Configure CMake with the following variables:

```powershell
# PowerShell
cmake .. -DCMAKE_BUILD_TYPE=Release `
    -DBLAS_LIBRARIES="C:/OpenBLAS/libopenblas.lib" `
    -DLAPACK_LIBRARIES="C:/OpenBLAS/libopenblas.lib" `
    -DOPENBLAS_INCLUDE_DIR="C:/OpenBLAS/include"
```

```cmd
:: Command Prompt
cmake .. -DCMAKE_BUILD_TYPE=Release ^
    -DBLAS_LIBRARIES="C:/OpenBLAS/libopenblas.lib" ^
    -DLAPACK_LIBRARIES="C:/OpenBLAS/libopenblas.lib" ^
    -DOPENBLAS_INCLUDE_DIR="C:/OpenBLAS/include"
```

#### Option 2: Using Environment Variables

Set environment variables before running CMake:

```powershell
# PowerShell
$env:BLAS_LIBRARIES = "C:/OpenBLAS/libopenblas.lib"
$env:LAPACK_LIBRARIES = "C:/OpenBLAS/libopenblas.lib"
$env:OPENBLAS_INCLUDE_DIR = "C:/OpenBLAS/include"

cmake .. -DCMAKE_BUILD_TYPE=Release
```

#### Option 3: Using vcpkg

```powershell
vcpkg install openblas:x64-windows
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=[vcpkg-root]/scripts/buildsystems/vcpkg.cmake
```

#### Windows Library Naming Notes

| Platform | BLAS Library Name | Notes |
|----------|-------------------|-------|
| Linux | `libopenblas.a` or `libopenblas.so` | 64-bit integers supported |
| macOS | `libopenblas.a` or `libopenblas.dylib` | 64-bit integers supported |
| Windows | `libopenblas.lib` | Typically uses 32-bit integers |

The CMake recipes automatically handle these differences:
- On Windows, SuiteSparse is configured with `SUITESPARSE_USE_64BIT_BLAS=OFF`
- On Linux/macOS, SuiteSparse uses `SUITESPARSE_USE_64BIT_BLAS=ON`

## License

See [LICENSE](LICENSE) file.
