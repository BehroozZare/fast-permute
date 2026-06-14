# Homa

Homa is a C++20 library that computes fast fill-reducing orderings for sparse
symmetric matrices, targeting sparse Cholesky factorization and related sparse
direct solvers. It uses a mesh-induced patch decomposition to build a quotient
graph, applies recursive nested dissection on that graph, and assembles a
low-fill permutation bottom-up. The result plugs directly into CHOLMOD, Intel
MKL PARDISO, or NVIDIA cuDSS with no solver-side changes.

---

## Using Homa in your CMake project (FetchContent)

```cmake
include(FetchContent)
FetchContent_Declare(homa
    GIT_REPOSITORY https://github.com/BehroozZare/fast-permute.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(homa)

target_link_libraries(my_target PRIVATE Homa::homa)
homa_configure_runtime(my_target)
```

Call `homa_configure_runtime()` for executables that link Homa so Windows DLLs
and Linux build RPATHs are set for enabled solver backends.

---

## Building the examples

When examples are enabled, solver examples are built for the enabled solver
backends. Disable the backends you do not have installed.

```bash
# CHOLMOD (requires SuiteSparse)
cmake --preset release \
  -DHOMA_BUILD_EXAMPLE=ON \
  -DHOMA_WITH_CHOLMOD=ON
cmake --build --preset release

# MKL PARDISO (uses installed MKL or the MKL wheel fallback)
cmake --preset release \
  -DHOMA_BUILD_EXAMPLE=ON \
  -DHOMA_WITH_MKL=ON
cmake --build --preset release

# cuDSS (requires CUDA + cuDSS)
cmake --preset release \
  -DHOMA_BUILD_EXAMPLE=ON \
  -DHOMA_WITH_CUDSS=ON
cmake --build --preset release
```

Binaries land in `build/release/examples/`.

---

## Running an example with the bundled dragon mesh

```bash
# CHOLMOD
./build/release/examples/cholmod_example -i input/meshes/dragon.obj

# MKL PARDISO
./build/release/examples/mkl_example -i input/meshes/dragon.obj

# cuDSS (GPU ordering enabled)
./build/release/examples/cudss_example -i input/meshes/dragon.obj -g 1
```

Optional flag: `-p <patch_size>` (default 512).

Each binary prints:

```
Ordering  : <ms>
Analysis  : <ms>
Factorize : <ms>
Solve     : <ms>
Residual  : <value>
```

---

## Runner scripts

Convenience scripts iterate over all five bundled meshes automatically:

```bash
scripts/examples/run_cholmod_example.sh
scripts/examples/run_mkl_example.sh
scripts/examples/run_cudss_example.sh
```

Each script locates its binary under `build/` and exits with a clear message
if the binary has not been built yet.

---

## Solver integration - ergonomic API

```cpp
#include <homa/solvers/LinSysSolver.h>

std::unique_ptr<homa::LinSysSolver> solver(
    homa::LinSysSolver::create(homa::LinSysSolverType::CPU_CHOLMOD));

homa::Options opts;
opts.patch_size = 512;

solver->setMatrix(A);        // Eigen::SparseMatrix<double>
solver->ordering(opts);      // stores solver->ordering_result
solver->analyze_pattern();   // consumes solver-owned ordering state
solver->factorize();
solver->solve(rhs, result);
```

`setMatrix(A)` borrows Eigen's compressed sparse arrays; `A` must be square,
compressed, and outlive the solver analysis/factorization/solve using it.

Raw pointer and device-aware view APIs are also available:

```cpp
solver->setMatrix(p, i, x, n, nnz);

homa::SparseMatrixView Adev{n, n, nnz, d_rowptr, d_colind, d_values,
                            homa::SparseFormat::CSR,
                            homa::MemoryLocation::Device};
solver->setMatrix(Adev); // cuDSS only
```

MKL expects lower-triangular storage:

```cpp
Eigen::SparseMatrix<double> A_lower = A.triangularView<Eigen::Lower>();
A_lower.makeCompressed();
solver->setMatrix(A_lower);
```

If you only want HOMA's permutation/tree for another solver stack, use the
standalone ordering API. This works even when CHOLMOD, MKL, and cuDSS are not
built:

```cpp
homa::Options opts;
opts.compute_etree = true;
homa::OrderingResult ord = homa::compute_ordering(A, opts);
```

---
