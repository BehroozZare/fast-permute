
#pragma once

#ifdef USE_CHOLMOD

#include <homa/solvers/LinSysSolver.h>

#include <cholmod.h>
#include <Eigen/Eigen>
#include <string>
#include <vector>

namespace homa {

template <class Scalar>
class CHOLMODSolver : public LinSysSolver<Scalar> {
public:
    using Base = LinSysSolver<Scalar>;
    using typename Base::Vec;
    using typename Base::Mat;
    using Base::N;
    using Base::NNZ;
    using Base::L_NNZ;
    using Base::ordering_result;
    using Base::recordMatrixPattern;
    using Base::initVariables;
    using Base::setMatrix;

    cholmod_common cm;
    cholmod_sparse *A;
    cholmod_factor *L;
    cholmod_dense *b;

    cholmod_dense *x_solve;

    bool use_gpu = false;         // User-configured: whether to use GPU    
    std::vector<long int> p_long;
    std::vector<long int> i_long;

    void *Ai, *Ap, *Ax, *bx;
    ~CHOLMODSolver();
    CHOLMODSolver();

    // GPU initialization methods
    void initializeGPU();
    bool probeGPU();  // Returns true if GPU is available

    void setMatrix(int* p, int* i, Scalar* x, int A_N, int NNZ) override;
    void innerAnalyze_pattern(std::vector<int>& user_defined_perm, std::vector<int>& etree) override;
    void innerFactorize(void) override;
    void innerSolve(Vec& rhs, Vec& result) override;
    void innerSolve(Mat& rhs, Mat& result) override;
    void innerSolveRaw(const Scalar* rhs_data, int rows, int cols, Scalar* result_data) override;
    void resetSolver() override;
    void save_factor(const std::string& filePath);
    LinSysSolverType type() const override;

    void cholmod_clean_memory();
};

extern template class CHOLMODSolver<float>;
extern template class CHOLMODSolver<double>;

using CHOLMODSolverD = CHOLMODSolver<double>;
using CHOLMODSolverF = CHOLMODSolver<float>;

}  // namespace homa

#endif
