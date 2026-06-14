
#pragma once

#ifdef USE_CUDSS

#include <homa/solvers/LinSysSolver.h>
#include <cuda.h>
#include <cudss.h>
#include <Eigen/Eigen>
#include <vector>

namespace homa {

class CUDSSSolver : public LinSysSolver {
    typedef LinSysSolver Base; // The class
public:                // Access specifier

    cudssHandle_t             handle;
    cudssConfig_t             config;
    cudssData_t               data;
    cudssMatrix_t             A;
    cudssMatrix_t             b_mat;
    cudssMatrix_t             x_mat;

    //Device pointers
    int*    rowOffsets_dev;
    int*    colIndices_dev;
    double* values_dev;
    double* bvalues_dev;
    double* xvalues_dev;

    bool is_allocated;
    bool owns_sparse_matrix_mem;
    bool owns_bvalues_mem;
    bool owns_xvalues_mem;

    ~CUDSSSolver();
    CUDSSSolver();
    using Base::setMatrix;

    void setMatrix(SparseMatrixView& A) override;
    void setMatrix(int *p, int *i, double *x, int A_N, int NNZ) override;
    void innerOrdering(std::vector<int>& user_defined_perm, std::vector<int>& etree) override;
    void innerAnalyze_pattern(std::vector<int>& user_defined_perm, std::vector<int>& etree) override;
    void innerFactorize(void) override;
    void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
    void innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) override;
    void innerSolveRaw(const double* rhs_data, int rows, int cols, double* result_data) override;
    void innerSolveView(DenseMatrixView& rhs, DenseMatrixView& result) override;
    void resetSolver() override;
    int getFactorNNZ() override;
    void clean_sparse_matrix_mem();
    void clean_rhs_sol_mem();
    virtual LinSysSolverType type() const override;

protected:
    void copyDeviceMatrixPatternToHost() override;
};

}

#endif
