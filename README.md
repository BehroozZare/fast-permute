# Homa [![Build](https://github.com/BehroozZare/fast-permute/actions/workflows/build.yml/badge.svg)](https://github.com/BehroozZare/fast-permute/actions/workflows/build.yml)

Homa is a C++20 library with two complementary purposes:

1. **A fast fill-reducing ordering for sparse SPD matrices** — the reference
   implementation of the SIGGRAPH 2026 paper *Fast Sparse Matrix Permutation
   for Mesh-Based Direct Solvers* [Zarebavani et al. 2026]. Homa builds a
   mesh-induced patch decomposition, runs recursive nested dissection on the
   resulting quotient graph, and assembles a fill-reducing permutation
   bottom-up. The permutation drops into any sparse Cholesky factorization
   with no solver-side changes.

2. **A drop-in SPD linear solver wrapper** with a uniform
   `analyze -> factorize -> solve` API on top of three state-of-the-art
   sparse direct solvers: SuiteSparse **CHOLMOD**, Intel MKL **PARDISO**, and
   NVIDIA **cuDSS**. The user can pick a backend at construction time; the rest of your
   code stays the same.

The CMake build automatically fetches and configures all three solver
backends on both Windows and Linux—no manual dependency setup required.

---

## 1. Fast sparse matrix ordering

The standalone ordering API takes the sparsity pattern of a square SPD matrix
and returns a fill-reducing permutation (and the elimination tree).
It works even when no solver backend is built.

```cpp
#include <homa/ordering.h>

homa::Options opts;
opts.patch_size    = 512;   // Lloyd / METIS patch target
opts.compute_etree = true;  // required if you feed cuDSS

homa::OrderingResult ord = homa::compute_ordering(A, opts);
// ord.perm  — permutation of [0, n)
// ord.etree — elimination tree (only when opts.compute_etree)
```

The user then can hand the permutation to any third-party solver that supports a user ordering.

---

## 2. Sparse SPD linear solver

The `LinSysSolver` interface is a uniform front-end to all three backends, by switching backends through changing the `LinSysSolverType` argument. `LinSysSolver` is a class template parameterized on `Scalar`; both `float` and `double` are supported, with convenience aliases `homa::LinSysSolverD` and `homa::LinSysSolverF`.

```cpp
#include <homa/solvers/LinSysSolver.h>

std::unique_ptr<homa::LinSysSolverD> solver(
    homa::LinSysSolverD::create(homa::LinSysSolverType::CPU_CHOLMOD));
//                                                          ::CPU_MKL
//                                                          ::GPU_CUDSS

homa::Options opts;
opts.patch_size = 512;

solver->setMatrix(A);        // Eigen::SparseMatrix<double>
solver->ordering(opts);      // Homa ordering — skip to use the solver default
solver->analyze_pattern();   // consumes solver-owned ordering state
solver->factorize();
solver->solve(rhs, result);
```

`setMatrix(A)` borrows (shallow copy) Eigen's compressed sparse arrays; `A` must be square,
compressed, and outlive the analysis/factorization/solve using it.

Raw pointer and device-aware view APIs are also available:

```cpp
solver->setMatrix(p, i, x, n, nnz);

homa::SparseMatrixView<double> Adev{n, n, nnz, d_rowptr, d_colind, d_values,
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

---

## Building the examples

When examples are enabled, a solver example is built per enabled backend.
Disable the backends you do not have installed.

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

### Running the examples

```bash
./build/bin/cholmod_example -i input/meshes/dragon.obj
./build/bin/mkl_example     -i input/meshes/dragon.obj
./build/bin/cudss_example   -i input/meshes/dragon.obj
```

Optional flag: `-p <patch_size>` (default 512).

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

Call `homa_configure_runtime()` for executables that link Homa so Windows
DLLs and Linux build RPATHs are set for enabled solver backends.

---

## Citation

If you use Homa's ordering in academic work, please cite:

```bibtex
@inproceedings{Zarebavani:2026:LRA,
  title     = {Fast Sparse Matrix Permutation for Mesh-Based Direct Solvers},
  author    = {Zarebavani, Behrooz and Mahmoud, Ahmed H. and Dodik, Ana and Yuan, Changcheng and Porumbescu, Serban D. and Owens, John D. and Mehri Dehnavi, Maryam and Solomon, Justin},
  year      = {2026},
  isbn      = {979-8-4007-2554-8/2026/07},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  numpages  = {11},
  month     = jul,
  series    = {SIGGRAPH Conference Papers '26},
  booktitle = {Proceedings of the SIGGRAPH 2026 Conference Papers},
  doi       = {10.1145/3799902.3811189}
}
```
