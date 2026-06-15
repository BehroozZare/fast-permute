//
//  MKLSolver.cpp
//  Linear System Solver using Intel MKL PARDISO
//

#ifdef USE_MKL

#include <homa/solvers/MKLSolver.h>
#include "scalar_traits.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <mutex>
#include <new>
#include <stdexcept>
#include <spdlog/spdlog.h>

namespace homa {

static_assert(sizeof(MKL_INT) == sizeof(int),
              "Homa MKLSolver expects oneMKL LP64 because inputs use int CSR indices.");

#ifndef HOMA_MKL_INTERFACE_LAYER
#define HOMA_MKL_INTERFACE_LAYER MKL_INTERFACE_LP64
#endif

#ifndef HOMA_MKL_THREADING_LAYER
#define HOMA_MKL_THREADING_LAYER MKL_THREADING_INTEL
#endif

template <class Scalar>
MKLSolver<Scalar>::~MKLSolver()
{
    releasePardisoMemory();
}

template <class Scalar>
MKLSolver<Scalar>::MKLSolver()
{
    configureMKLRuntime();
    resetPardisoHandle();
    setMKLConfigParam();

    Ap = nullptr;
    Ai = nullptr;
    Ax = nullptr;
    N_MKL = 0;

    Base::initVariables();
}

template <class Scalar>
void MKLSolver<Scalar>::configureMKLRuntime()
{
    static std::once_flag configure_once;
    std::call_once(configure_once, []() {
        mkl_set_interface_layer(HOMA_MKL_INTERFACE_LAYER);
        mkl_set_threading_layer(HOMA_MKL_THREADING_LAYER);
    });
}

template <class Scalar>
void MKLSolver<Scalar>::resetPardisoHandle()
{
    for (int i = 0; i < 64; i++) {
        pt[i] = nullptr;
    }
    has_pardiso_memory_ = false;
    factorized_         = false;
}

template <class Scalar>
void MKLSolver<Scalar>::releasePardisoMemory()
{
    if (!has_pardiso_memory_) {
        return;
    }

    MKL_INT release_phase = -1;
    error                 = 0;
    MKL_INT* perm_ptr     = perm.empty() ? nullptr : perm.data();

    PARDISO(pt, &maxfct, &mnum, &mtype, &release_phase, &N_MKL, nullptr,
            nullptr, nullptr, perm_ptr, &nrhs, iparm, &msglvl, nullptr,
            nullptr, &error);

    if (error != 0) {
        spdlog::warn("MKL PARDISO: ERROR during memory release - code: {}", error);
    }

    resetPardisoHandle();
}

template <class Scalar>
void MKLSolver<Scalar>::setMKLConfigParam()
{
    for (int i = 0; i < 64; i++) {
        iparm[i] = 0;
    }

    iparm[0] = 1;   /* No solver default */
    iparm[1] = 2;   /* Fill-in reordering from METIS */
    iparm[2] = 0;
    iparm[3] = 0;   /* No iterative-direct algorithm */
    iparm[4] = 0;   /* User permutation is ignored by default */
    iparm[5] = 0;   /* Write solution into x */
    iparm[6] = 0;   /* Not in use */
    iparm[7] = 1;   /* Max numbers of iterative refinement steps */
    iparm[8] = 0;   /* Not in use */
    iparm[9] = 0;   /* Perturb the pivot elements with 1E-8 */
    iparm[10] = 0;  /* Use nonsymmetric permutation and scaling MPS */
    iparm[11] = 0;  /* A^TX=B */
    iparm[12] = 0;  /* Maximum weighted matching algorithm is switched-off */
    iparm[13] = 0;  /* Output: Number of perturbed pivots */
    iparm[14] = 0;  /* Not in use */
    iparm[15] = 0;  /* Not in use */
    iparm[16] = 0;  /* Not in use */
    iparm[17] = -1; /* Output: Number of nonzeros in the factor LU */
    iparm[18] = -1; /* Output: Mflops for LU factorization */
    iparm[19] = 0;  /* Output: Numbers of CG Iterations */
    iparm[20] = 1;  /* Using Bunch-Kaufman pivoting */
    iparm[55] = 0;  /* Diagonal and pivoting control, default is zero */
    iparm[59] = 1;  /* Use in-core intel MKL pardiso */

    iparm[26] = 1;
    // 0 = double precision, 1 = single precision (PARDISO consults this to
    // interpret the void* matrix/rhs/solution buffers we pass below).
    iparm[27] = detail::mkl_iparm27_v<Scalar>;
    iparm[34] = 1;  /* Zero-based indexing */
    iparm[30] = 0;
    iparm[35] = 0;

    maxfct = 1; /* Maximum number of numerical factorizations. */
    mnum = 1;   /* Which factorization to use. */
    msglvl = 0; /* Print statistical information in file */
    error = 0;  /* Initialize error flag */
    nrhs = 1;   /* Number of right hand sides. */
    mtype = 2;  /* Real and SPD matrices */
}

template <class Scalar>
void MKLSolver<Scalar>::clean_memory()
{
    releasePardisoMemory();
}

template <class Scalar>
void MKLSolver<Scalar>::setMatrix(int* p, int* i, Scalar* x, int A_N, int NNZ_in)
{
    releasePardisoMemory();

    assert(p[A_N] == NNZ_in);
    recordMatrixPattern(p, i, A_N, NNZ_in, SparseFormat::CSC, MemoryLocation::Host);
    this->N = A_N;
    this->NNZ = NNZ_in;
    this->N_MKL = A_N;

    Ap = p;
    Ai = i;
    Ax = x;
}

template <class Scalar>
void MKLSolver<Scalar>::innerAnalyze_pattern(std::vector<int>& user_defined_perm, std::vector<int>& etree)
{
    releasePardisoMemory();
    resetPardisoHandle();
    setMKLConfigParam();

    bool use_user_perm = (user_defined_perm.size() == static_cast<size_t>(N));
    perm.assign(static_cast<size_t>(N), 0);

    if (use_user_perm) {
        iparm[4] = 1; /* User permutation */
        perm.assign(user_defined_perm.begin(), user_defined_perm.end());
        spdlog::info("MKL PARDISO: Using user-provided permutation");
    } else {
        iparm[4] = 0; /* Use internal METIS ordering */
        spdlog::info("MKL PARDISO: Using DEFAULT ordering");
        if (ordering_type == "AMD") {
            iparm[1] = 0;
        } else if (ordering_type == "ParMETIS") {
            iparm[1] = 3;
        } else {
            iparm[1] = 2;
        }
    }

    assert(N == N_MKL);
    assert(Ap[N_MKL] == NNZ);

    phase = 11;
    error = 0;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL,
            static_cast<void*>(Ax), Ap, Ai,
            perm.data(), &nrhs, iparm, &msglvl,
            static_cast<void*>(&ddum), static_cast<void*>(&ddum), &error);

    if (error != 0) {
        spdlog::error("MKL PARDISO: ERROR during symbolic factorization - code: {}", error);
        throw std::runtime_error("Symbolic factorization failed with error code: " + std::to_string(error));
    }

    has_pardiso_memory_ = true;
    factorized_         = false;

    spdlog::info("MKL PARDISO: Symbolic analysis complete");
}

template <class Scalar>
void MKLSolver<Scalar>::innerFactorize(void)
{
    if (!has_pardiso_memory_) {
        throw std::runtime_error("MKL PARDISO factorize called before analyze_pattern");
    }

    phase = 22;
    error = 0;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL,
            static_cast<void*>(Ax), Ap, Ai,
            perm.data(), &nrhs, iparm, &msglvl,
            static_cast<void*>(&ddum), static_cast<void*>(&ddum), &error);

    L_NNZ = iparm[17];

    if (error != 0) {
        spdlog::error("MKL PARDISO: ERROR during numerical factorization - code: {}", error);
        throw std::runtime_error("Numerical factorization failed with error code: " + std::to_string(error));
    }

    factorized_ = true;

    spdlog::info("MKL PARDISO: Numerical factorization complete, L_NNZ = {}", L_NNZ);
}

template <class Scalar>
void MKLSolver<Scalar>::innerSolve(typename MKLSolver<Scalar>::Vec& rhs,
                                   typename MKLSolver<Scalar>::Vec& result)
{
    if (!factorized_) {
        throw std::runtime_error("MKL PARDISO solve called before factorize");
    }

    Scalar* x = static_cast<Scalar*>(
        mkl_calloc(static_cast<size_t>(rhs.size()) * nrhs, sizeof(Scalar), 64));
    if (!x) {
        throw std::bad_alloc();
    }
    
    phase = 33;
    error = 0;
    iparm[7] = 0; /* Max numbers of iterative refinement steps. */

    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL,
            static_cast<void*>(Ax), Ap, Ai,
            perm.data(), &nrhs, iparm, &msglvl,
            static_cast<void*>(rhs.data()), static_cast<void*>(x), &error);

    if (error != 0) {
        spdlog::error("MKL PARDISO: ERROR during solve - code: {}", error);
        mkl_free(x);
        throw std::runtime_error("Solve failed with error code: " + std::to_string(error));
    }

    result = Eigen::Map<Vec, Eigen::Unaligned>(x, N);
    mkl_free(x);
}

template <class Scalar>
void MKLSolver<Scalar>::innerSolve(typename MKLSolver<Scalar>::Mat& rhs,
                                   typename MKLSolver<Scalar>::Mat& result)
{
    // Delegate to raw pointer version
    result.resize(rhs.rows(), rhs.cols());
    innerSolveRaw(rhs.data(), static_cast<int>(rhs.rows()), static_cast<int>(rhs.cols()), result.data());
}

template <class Scalar>
void MKLSolver<Scalar>::innerSolveRaw(const Scalar* rhs_data, int rows, int cols, Scalar* result_data)
{
    // Solve column by column
    for (int c = 0; c < cols; c++) {
        // Map column c of input (column-major layout)
        Vec rhs_col = Eigen::Map<const Vec>(rhs_data + c * rows, rows);
        Vec result_col(rows);

        innerSolve(rhs_col, result_col);

        // Copy result to output column
        std::memcpy(result_data + c * rows, result_col.data(), rows * sizeof(Scalar));
    }
}

template <class Scalar>
void MKLSolver<Scalar>::resetSolver()
{
    releasePardisoMemory();
    resetPardisoHandle();
    setMKLConfigParam();
    
    Ap = nullptr;
    Ai = nullptr;
    Ax = nullptr;
    N_MKL = 0;
    perm.clear();
    
    Base::initVariables();
}

template <class Scalar>
LinSysSolverType MKLSolver<Scalar>::type() const
{
    return LinSysSolverType::CPU_MKL;
}

template class MKLSolver<float>;
template class MKLSolver<double>;

}  // namespace homa

#endif
