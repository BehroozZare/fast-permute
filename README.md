# Homa

Homa is a C++20 library that computes fast fill-reducing orderings for sparse
symmetric matrices, targeting sparse Cholesky factorization and related sparse
direct solvers. It uses a mesh-induced patch decomposition to build a quotient
graph, applies recursive nested dissection on that graph, and assembles a
low-fill permutation bottom-up. The result plugs directly into CHOLMOD, Intel
MKL PARDISO, or NVIDIA cuDSS with no solver-side changes.

---
## Solver API

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

## Building the examples

When examples are enabled, solver examples are built for the enabled solver
backends. Disable the backends you do not have installed.

```bash
# CHOLMOD
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHOMA_BUILD_EXAMPLE=ON -DHOMA_WITH_CHOLMOD=ON
cmake --build build

# MKL PARDISO
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHOMA_BUILD_EXAMPLE=ON -DHOMA_WITH_MKL=ON
cmake --build build

# cuDSS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DHOMA_BUILD_EXAMPLE=ON -DHOMA_WITH_CUDSS=ON
cmake --build build
```
---

### Running the examples

```bash
# CHOLMOD
./build/bin/cholmod_example -i input/meshes/dragon.obj

# MKL PARDISO
./build/bin/mkl_example -i input/meshes/dragon.obj

# cuDSS 
./build/bin/cudss_example -i input/meshes/dragon.obj
```

Optional flag: `-p <patch_size>` (default 512).


## Using Homa in your CMake project with FetchContent

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