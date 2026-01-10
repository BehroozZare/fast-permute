
#pragma once

#include "LinSysSolver.hpp"
#ifdef USE_STRUMPACK
#include <StrumpackSparseSolver.hpp>
#include <StrumpackSparseSolverMixedPrecision.hpp>

#include <sparse/CSRMatrix.hpp>
#include <Eigen/Eigen>
#include <vector>

namespace RXMESH_SOLVER {

class STRUMPACKSolver : public LinSysSolver {
    typedef LinSysSolver Base; // The class
public:                // Access specifier
    strumpack::SparseSolverMixedPrecision<float,double,int> spss;
    strumpack::CSRMatrix<double, int> _A; // create matrix object
    strumpack::DenseMatrix<double> _rhs; // create right-hand side vector object
    strumpack::DenseMatrix<double> _x; // create solution vector object

    ~STRUMPACKSolver();
    STRUMPACKSolver();

    void setMatrix(int *p, int *i, double *x, int A_N, int NNZ) override;
    void innerAnalyze_pattern(std::vector<int>& user_defined_perm, std::vector<int>& etree) override;
    void innerFactorize(void) override;
    void innerSolve(Eigen::VectorXd &rhs, Eigen::VectorXd &result) override;
    void innerSolve(Eigen::MatrixXd &rhs, Eigen::MatrixXd &result) override;
    void innerSolveRaw(const double* rhs_data, int rows, int cols, double* result_data) override;
    void resetSolver() override;
    int getFactorNNZ() override;
    void clean_sparse_matrix_mem();
    void clean_rhs_sol_mem();
    virtual LinSysSolverType type() const override;

};

}
#endif