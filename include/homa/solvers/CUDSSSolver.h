
#pragma once

#ifdef USE_CUDSS

#include <homa/solvers/LinSysSolver.h>
#include <cuda.h>
#include <cudss.h>
#include <Eigen/Eigen>
#include <vector>

namespace homa {

template <class Scalar>
class CUDSSSolver : public LinSysSolver<Scalar> {
public:
    using Base = LinSysSolver<Scalar>;
    using typename Base::Vec;
    using typename Base::Mat;
    using Base::N;
    using Base::NNZ;
    using Base::L_NNZ;
    using Base::ordering_result;
    using Base::matrix_view_;
    using Base::owned_host_outer_;
    using Base::owned_host_inner_;
    using Base::recordMatrixPattern;
    using Base::initVariables;
    using Base::setMatrix;

    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t   data;
    cudssMatrix_t A;
    cudssMatrix_t b_mat;
    cudssMatrix_t x_mat;

    // Device pointers
    int*    rowOffsets_dev;
    int*    colIndices_dev;
    Scalar* values_dev;
    Scalar* bvalues_dev;
    Scalar* xvalues_dev;

    bool is_allocated;
    bool owns_sparse_matrix_mem;
    bool owns_bvalues_mem;
    bool owns_xvalues_mem;

    ~CUDSSSolver();
    CUDSSSolver();

    void setMatrix(SparseMatrixView<Scalar>& A) override;
    void setMatrix(int* p, int* i, Scalar* x, int A_N, int NNZ) override;
    void innerOrdering(std::vector<int>& user_defined_perm, std::vector<int>& etree) override;
    void innerAnalyze_pattern(std::vector<int>& user_defined_perm, std::vector<int>& etree) override;
    void innerFactorize(void) override;
    void innerSolve(Vec& rhs, Vec& result) override;
    void innerSolve(Mat& rhs, Mat& result) override;
    void innerSolveRaw(const Scalar* rhs_data, int rows, int cols, Scalar* result_data) override;
    void innerSolveView(DenseMatrixView<Scalar>& rhs, DenseMatrixView<Scalar>& result) override;
    void resetSolver() override;
    int  getFactorNNZ() override;
    void clean_sparse_matrix_mem();
    void clean_rhs_sol_mem();
    LinSysSolverType type() const override;

protected:
    void copyDeviceMatrixPatternToHost() override;
};

extern template class CUDSSSolver<float>;
extern template class CUDSSSolver<double>;

using CUDSSSolverD = CUDSSSolver<double>;
using CUDSSSolverF = CUDSSSolver<float>;

}  // namespace homa

#endif
