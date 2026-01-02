
#pragma once

#ifdef USE_MKL

#include "LinSysSolver.hpp"

#include <mkl.h>
#include <Eigen/Eigen>
#include <vector>

namespace RXMESH_SOLVER {

class MKLSolver : public LinSysSolver {
    typedef LinSysSolver Base;

public:
    MKL_INT mtype = 2; /* Real SPD matrix */
    MKL_INT nrhs = 1;  /* Number of right hand sides. */
    long int pt[64];
    /* Pardiso control parameters. */
    MKL_INT iparm[64];
    MKL_INT maxfct, mnum, phase, error, msglvl;
    /* Auxiliary variables. */
    double ddum;  /* Double dummy */
    MKL_INT idum; /* Integer dummy. */

    MKL_INT *Ap;
    MKL_INT *Ai;
    double *Ax;
    MKL_INT N_MKL;

    std::vector<MKL_INT> perm;

    ~MKLSolver();
    MKLSolver();

    void setMatrix(int *p, int *i, double *x, int A_N, int NNZ) override;
    void innerAnalyze_pattern(std::vector<int>& user_defined_perm, std::vector<int>& etree) override;
    void innerFactorize(void) override;
    void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
    void resetSolver() override;
    LinSysSolverType type() const override;

private:
    void setMKLConfigParam();
    void clean_memory();
};

}

#endif
