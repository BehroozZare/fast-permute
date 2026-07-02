
#pragma once

#ifdef USE_MKL

#include <homa/solvers/LinSysSolver.h>

#include <mkl.h>
#include <Eigen/Eigen>
#include <vector>

namespace homa {

template <class Scalar>
class MKLSolver : public LinSysSolver<Scalar>
{
   public:
    using Base = LinSysSolver<Scalar>;
    using Base::initVariables;
    using Base::L_NNZ;
    using Base::N;
    using Base::NNZ;
    using Base::ordering_result;
    using Base::ordering_type;
    using Base::recordMatrixPattern;
    using Base::setMatrix;
    using typename Base::Mat;
    using typename Base::Vec;

    MKL_INT mtype = 2; /* Real SPD matrix */
    MKL_INT nrhs  = 1; /* Number of right hand sides. */
    void*   pt[64];
    /* Pardiso control parameters. */
    MKL_INT iparm[64];
    MKL_INT maxfct, mnum, phase, error, msglvl;
    /* Auxiliary variables. */
    Scalar  ddum; /* Scalar dummy */
    MKL_INT idum; /* Integer dummy. */

    MKL_INT* Ap;
    MKL_INT* Ai;
    Scalar*  Ax;
    MKL_INT  N_MKL;

    std::vector<MKL_INT> perm;
    bool                 has_pardiso_memory_ = false;
    bool                 factorized_         = false;

    ~MKLSolver();
    MKLSolver();

    void setMatrix(int* p, int* i, Scalar* x, int A_N, int NNZ) override;
    void innerAnalyze_pattern(std::vector<int>& user_defined_perm,
                              std::vector<int>& etree) override;
    void innerFactorize(void) override;
    void innerSolve(Vec& rhs, Vec& result) override;
    void innerSolve(Mat& rhs, Mat& result) override;
    void innerSolveRaw(const Scalar* rhs_data,
                       int           rows,
                       int           cols,
                       Scalar*       result_data) override;
    void resetSolver() override;
    LinSysSolverType type() const override;

   private:
    void configureMKLRuntime();
    void resetPardisoHandle();
    void releasePardisoMemory();
    void setMKLConfigParam();
    void clean_memory();
};

extern template class MKLSolver<float>;
extern template class MKLSolver<double>;

using MKLSolverD = MKLSolver<double>;
using MKLSolverF = MKLSolver<float>;

}  // namespace homa

#endif
