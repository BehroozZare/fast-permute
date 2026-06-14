#include "homa/solvers/LinSysSolver.h"

#include <Eigen/Sparse>
#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

namespace {

class ProbeSolver : public homa::LinSysSolver
{
public:
    explicit ProbeSolver(homa::LinSysSolverType solver_type)
        : solver_type_(solver_type)
    {
    }

    using homa::LinSysSolver::setMatrix;

    homa::LinSysSolverType type() const override { return solver_type_; }

    void setMatrix(int* p, int* i, double* x, int A_N, int nnz) override
    {
        p_   = p;
        i_   = i;
        x_   = x;
        N    = A_N;
        NNZ  = nnz;
        recordMatrixPattern(p,
                            i,
                            A_N,
                            nnz,
                            homa::SparseFormat::CSC,
                            homa::MemoryLocation::Host);
    }

    void innerAnalyze_pattern(std::vector<int>&, std::vector<int>&) override {}
    void innerFactorize() override {}
    void innerSolve(Eigen::VectorXd&, Eigen::VectorXd&) override {}
    void innerSolve(Eigen::MatrixXd&, Eigen::MatrixXd&) override {}
    void innerSolveRaw(const double*, int, int, double*) override {}
    void resetSolver() override { initVariables(); }

    int*    p_ = nullptr;
    int*    i_ = nullptr;
    double* x_ = nullptr;

private:
    homa::LinSysSolverType solver_type_;
};

Eigen::SparseMatrix<double> makeCompressedMatrix(
    int rows, const std::vector<Eigen::Triplet<double>>& triplets)
{
    Eigen::SparseMatrix<double> A(rows, rows);
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();
    return A;
}

} // namespace

TEST(LinSysSolverApi, EigenSetMatrixBorrowsOriginalPointers)
{
    Eigen::SparseMatrix<double> A = makeCompressedMatrix(
        3,
        {{0, 0, 4.0},
         {1, 0, 1.0},
         {1, 1, 5.0},
         {2, 1, 2.0},
         {2, 2, 6.0}});

    ProbeSolver solver(homa::LinSysSolverType::CPU_CHOLMOD);
    solver.setMatrix(A);

    EXPECT_EQ(solver.p_, A.outerIndexPtr());
    EXPECT_EQ(solver.i_, A.innerIndexPtr());
    EXPECT_EQ(solver.x_, A.valuePtr());
}

TEST(LinSysSolverApi, MklEigenSetMatrixRejectsUpperStorage)
{
    Eigen::SparseMatrix<double> A = makeCompressedMatrix(
        2,
        {{0, 0, 2.0},
         {0, 1, 1.0},
         {1, 0, 1.0},
         {1, 1, 3.0}});

    ProbeSolver solver(homa::LinSysSolverType::CPU_MKL);
    EXPECT_THROW(solver.setMatrix(A), std::invalid_argument);
}

TEST(LinSysSolverApi, MklEigenSetMatrixBorrowsLowerStorage)
{
    Eigen::SparseMatrix<double> A_lower = makeCompressedMatrix(
        2,
        {{0, 0, 2.0},
         {1, 0, 1.0},
         {1, 1, 3.0}});

    ProbeSolver solver(homa::LinSysSolverType::CPU_MKL);
    solver.setMatrix(A_lower);

    EXPECT_EQ(solver.p_, A_lower.outerIndexPtr());
    EXPECT_EQ(solver.i_, A_lower.innerIndexPtr());
    EXPECT_EQ(solver.x_, A_lower.valuePtr());
}
