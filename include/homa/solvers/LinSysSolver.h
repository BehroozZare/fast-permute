//
//  LinSysSolver.h
//  IPC
//
//  Created by Minchen Li on 6/30/18.
//
#pragma once


#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cassert>
#include <string>
#include <vector>

#include <homa/matrix_view.h>
#include <homa/types.h>


namespace homa {

enum class LinSysSolverType
{   
    CPU_CHOLMOD,
    CPU_MKL,
    GPU_CUDSS
};

class LinSysSolver
{
   public:

    int    L_NNZ    = 0;
    int    NNZ      = 0;
    int    N        = 0;
    double residual = 0;
    std::string ordering_name = "DEFAULT";
    std::string ordering_type = "METIS";
    OrderingResult ordering_result;

   public:
    virtual ~LinSysSolver(void) {};

    static LinSysSolver* create(const LinSysSolverType type);

    virtual LinSysSolverType type() const = 0;

   public:
    void setMatrix(const Eigen::SparseMatrix<double>& A);

    virtual void setMatrix(SparseMatrixView& A);

    virtual void setMatrix(int*              p,
                           int*              i,
                           double*           x,
                           int               A_N,
                           int               NNZ) = 0;

    void ordering(const Options& opts = {});

    void setOrdering(const OrderingResult& ordering);

    const OrderingResult& orderingResult() const
    {
        return ordering_result;
    }

    virtual void ordering(std::vector<int>& perm, std::vector<int>&etree)
    {
        if (static_cast<int>(perm.size()) == N) {
            ordering_result.perm  = perm;
            ordering_result.etree = etree;
        }
        innerOrdering(perm, etree);
        ordering_applied_ = true;
    }

    virtual void innerOrdering(std::vector<int>& user_defined_perm, std::vector<int>& etree) {return;}

    virtual void analyze_pattern()
    {
        if (static_cast<int>(ordering_result.perm.size()) == N) {
            innerAnalyze_pattern(ordering_result.perm, ordering_result.etree);
            return;
        }

        std::vector<int> empty;
        if (!ordering_applied_) {
            innerOrdering(empty, empty);
            ordering_applied_ = true;
        }
        innerAnalyze_pattern(empty, empty);
    }

    virtual void analyze_pattern(std::vector<int>& user_defined_perm, std::vector<int>& etree)
    {
        if (static_cast<int>(user_defined_perm.size()) == N) {
            ordering_result.perm  = user_defined_perm;
            ordering_result.etree = etree;
            innerOrdering(ordering_result.perm, ordering_result.etree);
            ordering_applied_ = true;
        } else if (!ordering_applied_) {
            std::vector<int> empty;
            innerOrdering(empty, empty);
            ordering_applied_ = true;
        }
        innerAnalyze_pattern(user_defined_perm, etree);
    }

    virtual void innerAnalyze_pattern(std::vector<int>& user_defined_perm, std::vector<int>& etree) = 0;

    virtual void factorize(void)
    {
        innerFactorize();
    }

    virtual void innerFactorize(void) = 0;
    virtual int getFactorNNZ(void)
    {
        return L_NNZ;
    }

    virtual void solve(Eigen::VectorXd& rhs, Eigen::VectorXd& result)
    {
        innerSolve(rhs, result);
    }

    virtual void solve(Eigen::MatrixXd& rhs, Eigen::MatrixXd& result)
    {
        // Use raw pointer interface to avoid ABI issues between GCC and NVCC
        // when passing Eigen types across compilation boundaries
        result.resize(rhs.rows(), rhs.cols());
        innerSolveRaw(rhs.data(), static_cast<int>(rhs.rows()), static_cast<int>(rhs.cols()), result.data());
    }

    virtual void solve(DenseMatrixView& rhs, DenseMatrixView& result)
    {
        innerSolveView(rhs, result);
    }

    virtual void computeResidual(Eigen::SparseMatrix<double>& mtr, Eigen::VectorXd& sol, Eigen::VectorXd& rhs)
    {
        assert(mtr.rows() == mtr.cols());
        assert(rhs.rows() == mtr.rows());
        assert(sol.rows() == mtr.rows());
        residual = (rhs - mtr * sol).norm();
    }

    virtual void innerSolve(Eigen::VectorXd& rhs, Eigen::VectorXd& result) = 0;
    virtual void innerSolve(Eigen::MatrixXd& rhs, Eigen::MatrixXd& result) = 0;
    
    // Raw pointer interface to avoid ABI issues between compilers (GCC vs NVCC)
    // rhs_data and result_data are column-major arrays of size rows x cols
    virtual void innerSolveRaw(const double* rhs_data, int rows, int cols, double* result_data) = 0;
    virtual void innerSolveView(DenseMatrixView& rhs, DenseMatrixView& result);


    virtual void resetSolver() = 0;

   public:
    double getResidual(void)
    {
        return residual;
    }

    virtual void initVariables()
    {
        L_NNZ    = 0;
        NNZ      = 0;
        N        = 0;
        residual = 0;
        ordering_result = {};
        ordering_applied_ = false;
        matrix_view_ = {};
        owned_host_outer_.clear();
        owned_host_inner_.clear();
    }

   protected:
    virtual void copyDeviceMatrixPatternToHost();

    void recordMatrixPattern(const int*     outer,
                             const int*     inner,
                             int            n,
                             int            nnz,
                             SparseFormat   format,
                             MemoryLocation location);


    SparseMatrixView matrix_view_;
    std::vector<int> owned_host_outer_;
    std::vector<int> owned_host_inner_;
    bool ordering_applied_ = false;
};

}  // namespace PARTH_SOLVER
