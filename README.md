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
```

---

## Building the examples

Each solver is opt-in. Enable the ones you have installed.

```bash
# CHOLMOD (requires SuiteSparse)
cmake --preset release \
  -DHOMA_BUILD_BENCHMARKS=ON \
  -DHOMA_EXAMPLE_WITH_CHOLMOD=ON \
  -DHOMA_WITH_SUITESPARSE=ON
cmake --build --preset release

# MKL PARDISO (requires Intel oneAPI MKL)
cmake --preset release \
  -DHOMA_BUILD_BENCHMARKS=ON \
  -DHOMA_EXAMPLE_WITH_MKL=ON \
  -DHOMA_WITH_MKL=ON
cmake --build --preset release

# cuDSS (requires CUDA + cuDSS)
cmake --preset release \
  -DHOMA_BUILD_BENCHMARKS=ON \
  -DHOMA_EXAMPLE_WITH_CUDSS=ON \
  -DHOMA_WITH_CUDA=ON
cmake --build --preset release
```

Binaries land in `build/release/examples/`.

---

## Running an example with the bundled dragon mesh

```bash
# CHOLMOD
./build/release/examples/cholmod_example -i examples/meshes/dragon.obj

# MKL PARDISO
./build/release/examples/mkl_example -i examples/meshes/dragon.obj

# cuDSS (GPU ordering enabled)
./build/release/examples/cudss_example -i examples/meshes/dragon.obj -g 1
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

## Solver integration — 4-step API

```cpp
#include <homa/ordering.h>          // homa::Ordering, homa::Options
#include "LinSysSolver.hpp"         // homa::LinSysSolver (examples only)

// 1. Build ordering
homa::Ordering* ord = homa::Ordering::create(homa::DEMO_ORDERING_TYPE::PATCH_ORDERING);
homa::Options opts;
opts.patch_size = 512;
ord->applyOptions(opts);
ord->setGraph(Gp, Gi, n, nnz);
ord->init();

std::vector<int> perm, etree;
ord->compute_permutation(perm, etree, /*for_gpu=*/false);

// 2. Set matrix
homa::LinSysSolver* solver = homa::LinSysSolver::create(homa::LinSysSolverType::CPU_CHOLMOD);
solver->setMatrix(p, i, x, n, nnz);

// 3. Analyze pattern (symbolic factorization)
solver->ordering(perm, etree);
solver->analyze_pattern(perm, etree);

// 4. Factorize and solve
solver->factorize();
solver->solve(rhs, result);
```

---

## Bundled meshes

Five triangle meshes are included in `examples/meshes/`:
`dragon.obj`, `giraffe.obj`, `torus.obj`, `bunnyhead.obj`, `cloth.obj`.
