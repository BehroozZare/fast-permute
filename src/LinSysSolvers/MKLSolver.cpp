//
//  MKLSolver.cpp
//  Linear System Solver using Intel MKL PARDISO
//

#ifdef USE_MKL

#include "MKLSolver.hpp"
#include <cassert>
#include <iostream>
#include <spdlog/spdlog.h>

namespace RXMESH_SOLVER {

MKLSolver::~MKLSolver()
{
    phase = -1; /* Release internal memory. */
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, Ax, Ap, Ai, NULL, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);
}

MKLSolver::MKLSolver()
{
    setMKLConfigParam();

    Ap = nullptr;
    Ai = nullptr;
    Ax = nullptr;
    N_MKL = 0;

    Base::initVariables();
}

void MKLSolver::setMKLConfigParam()
{
    for (int i = 0; i < 64; i++) {
        pt[i] = 0;
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

void MKLSolver::clean_memory()
{
    phase = -1; /* Release internal memory. */
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, &Ax, Ap, Ai, NULL, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);
}

void MKLSolver::setMatrix(int* p, int* i, double* x, int A_N, int NNZ)
{
    assert(p[A_N] == NNZ);
    this->N = A_N;
    this->NNZ = NNZ;
    this->N_MKL = A_N;

    Ap = p;
    Ai = i;
    Ax = x;
}

void MKLSolver::innerAnalyze_pattern(std::vector<int>& user_defined_perm, std::vector<int>& etree)
{
    // Clean memory from previous factorization
    setMKLConfigParam();

    // Check if user provided permutation
    bool use_user_perm = (user_defined_perm.size() == static_cast<size_t>(N));
    
    if (use_user_perm) {
        iparm[4] = 1; /* User permutation */
        perm.resize(N);
        for (int j = 0; j < N; j++) {
            perm[j] = user_defined_perm[j];
        }
        spdlog::info("MKL PARDISO: Using user-provided permutation");
    } else {
        iparm[4] = 0; /* Use internal METIS ordering */
        spdlog::info("MKL PARDISO: Using DEFAULT ordering");
    }

    assert(N == N_MKL);
    assert(Ap[N_MKL] == NNZ);

    // Release any previous internal memory
    phase = -1;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, Ax, Ap, Ai, NULL, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);

    // Reinitialization for new analysis
    setMKLConfigParam();
    if (use_user_perm) {
        iparm[4] = 1;
    } else {
        if (ordering_type == "AMD") {
            iparm[1] = 0;
        } else if (ordering_type == "ParMETIS") {
            iparm[1] = 3;
        } else {
            iparm[1] = 2;
        }
    }

    // Symbolic factorization
    phase = 11;
    if (use_user_perm) {
        PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL, Ax, Ap, Ai, perm.data(),
                &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    } else {
        PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL, Ax, Ap, Ai, NULL,
                &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    }

    if (error != 0) {
        spdlog::error("MKL PARDISO: ERROR during symbolic factorization - code: {}", error);
        throw std::runtime_error("Symbolic factorization failed with error code: " + std::to_string(error));
    }

    spdlog::info("MKL PARDISO: Symbolic analysis complete");
}

void MKLSolver::innerFactorize(void)
{
    // Numerical factorization
    phase = 22;
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL, Ax, Ap, Ai, NULL, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);

    L_NNZ = iparm[17];

    if (error != 0) {
        spdlog::error("MKL PARDISO: ERROR during numerical factorization - code: {}", error);
        throw std::runtime_error("Numerical factorization failed with error code: " + std::to_string(error));
    }

    spdlog::info("MKL PARDISO: Numerical factorization complete, L_NNZ = {}", L_NNZ);
}

void MKLSolver::innerSolve(Eigen::VectorXd& rhs, Eigen::VectorXd& result)
{
    double* x = (double*)mkl_calloc(rhs.size() * nrhs, sizeof(double), 64);
    
    phase = 33;
    iparm[7] = 0; /* Max numbers of iterative refinement steps. */

    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N_MKL, Ax, Ap, Ai, NULL, &nrhs,
            iparm, &msglvl, rhs.data(), x, &error);

    if (error != 0) {
        spdlog::error("MKL PARDISO: ERROR during solve - code: {}", error);
        mkl_free(x);
        throw std::runtime_error("Solve failed with error code: " + std::to_string(error));
    }

    result = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(x, N);
    mkl_free(x);
}

void MKLSolver::innerSolve(Eigen::MatrixXd& rhs, Eigen::MatrixXd& result)
{
    // Delegate to raw pointer version
    result.resize(rhs.rows(), rhs.cols());
    innerSolveRaw(rhs.data(), static_cast<int>(rhs.rows()), static_cast<int>(rhs.cols()), result.data());
}

void MKLSolver::innerSolveRaw(const double* rhs_data, int rows, int cols, double* result_data)
{
    // Solve column by column
    for (int c = 0; c < cols; c++) {
        // Map column c of input (column-major layout)
        Eigen::VectorXd rhs_col = Eigen::Map<const Eigen::VectorXd>(rhs_data + c * rows, rows);
        Eigen::VectorXd result_col(rows);
        
        innerSolve(rhs_col, result_col);
        
        // Copy result to output column
        memcpy(result_data + c * rows, result_col.data(), rows * sizeof(double));
    }
}

void MKLSolver::resetSolver()
{
    phase = -1; /* Release internal memory. */
    PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &N, Ax, Ap, Ai, NULL, &nrhs,
            iparm, &msglvl, &ddum, &ddum, &error);

    setMKLConfigParam();
    
    Ap = nullptr;
    Ai = nullptr;
    Ax = nullptr;
    N_MKL = 0;
    perm.clear();
    
    Base::initVariables();
}

LinSysSolverType MKLSolver::type() const
{
    return LinSysSolverType::CPU_MKL;
}

}  // namespace RXMESH_SOLVER

#endif
