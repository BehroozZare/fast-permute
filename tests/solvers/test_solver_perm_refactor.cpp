#include <algorithm>
#include <memory>
#include <type_traits>
#include <vector>

#include <Eigen/Sparse>
#include <gtest/gtest.h>

#include "homa/homa.h"
#include "homa/solvers/LinSysSolver.h"
#include "homa/utils/check_valid_permutation.h"

#ifdef USE_CUDSS
#include <cuda_runtime.h>

#include "homa/utils/cuda_error_handler.h"
#endif

template <class Scalar>
using SpMat = Eigen::SparseMatrix<Scalar>;

struct Case
{
    bool                           homa;
    homa::Options::SeparatorMethod sep;
    homa::Options::LocalMethod     local;
    homa::Options::PatchMethod     patch;
};

template <class Scalar>
SpMat<Scalar> make_matrix(Scalar shift)
{
    constexpr int side = 50, n = side * side;

    std::vector<Eigen::Triplet<Scalar>> t;
    for (int r = 0; r < side; ++r) {
        for (int c = 0; c < side; ++c) {
            int i = r * side + c;
            t.emplace_back(i, i, Scalar(5) + shift + Scalar(i) * Scalar(0.01));
            if (r + 1 < side) {
                int j = i + side;
                t.emplace_back(i, j, Scalar(-1));
                t.emplace_back(j, i, Scalar(-1));
            }
            if (c + 1 < side) {
                int j = i + 1;
                t.emplace_back(i, j, Scalar(-1));
                t.emplace_back(j, i, Scalar(-1));
            }
        }
    }
    SpMat<Scalar> A(n, n);
    A.setFromTriplets(t.begin(), t.end());
    A.makeCompressed();
    return A;
}

template <class Scalar>
SpMat<Scalar> solver_matrix(homa::LinSysSolverType type, const SpMat<Scalar>& A)
{
    if (type == homa::LinSysSolverType::GPU_CUDSS) {
        return A;
    }
    SpMat<Scalar> lower = A.template triangularView<Eigen::Lower>();
    lower.makeCompressed();
    return lower;
}

template <class Scalar>
void copy_values(SpMat<Scalar>& dst, const SpMat<Scalar>& src)
{
    ASSERT_EQ(dst.nonZeros(), src.nonZeros());
    std::copy_n(src.valuePtr(), src.nonZeros(), dst.valuePtr());
}

template <class Scalar>
void check_solve(const SpMat<Scalar>&                            A,
                 homa::LinSysSolver<Scalar>&                     solver,
                 const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& x_true)
{
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rhs = A * x_true, x;
    solver.solve(rhs, x);
    double rel = static_cast<double>((A * x - rhs).norm() / rhs.norm());
    double tol = std::is_same_v<Scalar, float> ? 1e-3 : 1e-8;
    EXPECT_LT(rel, tol);
}

template <class Scalar>
void run_case(homa::LinSysSolverType type, const Case& c)
{
    auto A0 = make_matrix<Scalar>(Scalar(0));
    auto A1 = make_matrix<Scalar>(Scalar(0.5));
    auto S0 = solver_matrix(type, A0);
    auto S1 = solver_matrix(type, A1);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x_true =
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::LinSpaced(
            A0.rows(), Scalar(1), Scalar(2));
    std::unique_ptr<homa::LinSysSolver<Scalar>> solver(
        homa::LinSysSolver<Scalar>::create(type));
    ASSERT_NE(solver, nullptr);

#ifdef USE_CUDSS
    int *   d_p = nullptr, *d_i = nullptr;
    Scalar* d_x = nullptr;
    if (type == homa::LinSysSolverType::GPU_CUDSS) {
        CUDA_CHECK(
            cudaMalloc((void**)&d_p, (S0.outerSize() + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_i, S0.nonZeros() * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&d_x, S0.nonZeros() * sizeof(Scalar)));
        CUDA_CHECK(cudaMemcpy(d_p,
                              S0.outerIndexPtr(),
                              (S0.outerSize() + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_i,
                              S0.innerIndexPtr(),
                              S0.nonZeros() * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_x,
                              S0.valuePtr(),
                              S0.nonZeros() * sizeof(Scalar),
                              cudaMemcpyHostToDevice));
        homa::SparseMatrixView<Scalar> view{int(S0.rows()),
                                            int(S0.cols()),
                                            int(S0.nonZeros()),
                                            d_p,
                                            d_i,
                                            d_x,
                                            homa::SparseFormat::CSR,
                                            homa::MemoryLocation::Device};
        solver->setMatrix(view);
    } else
#endif
    {
        solver->setMatrix(S0);
    }

    if (c.homa) {
        homa::Options opts;
        opts.separator_method = c.sep;
        opts.local_method     = c.local;
        opts.patch_method     = c.patch;
        auto ord              = homa::compute_ordering(A0, opts);
        ASSERT_TRUE(homa::check_valid_permutation(ord.perm.data(),
                                                  int(ord.perm.size())));
        solver->ordering(ord.perm, ord.etree);
    }

    solver->analyze_pattern();
    solver->factorize();
    check_solve(A0, *solver, x_true);

    copy_values(S0, S1);
#ifdef USE_CUDSS
    if (type == homa::LinSysSolverType::GPU_CUDSS) {
        CUDA_CHECK(cudaMemcpy(d_x,
                              S0.valuePtr(),
                              S0.nonZeros() * sizeof(Scalar),
                              cudaMemcpyHostToDevice));
    }
#endif
    solver->factorize();
    check_solve(A1, *solver, x_true);

#ifdef USE_CUDSS
    if (type == homa::LinSysSolverType::GPU_CUDSS) {
        solver.reset();
        CUDA_CHECK(cudaFree(d_p));
        CUDA_CHECK(cudaFree(d_i));
        CUDA_CHECK(cudaFree(d_x));
    }
#endif
}

template <class Scalar>
void run_all()
{
    std::vector<homa::LinSysSolverType> solvers;
#ifdef USE_CHOLMOD
    solvers.push_back(homa::LinSysSolverType::CPU_CHOLMOD);
#endif
#ifdef USE_MKL
    solvers.push_back(homa::LinSysSolverType::CPU_MKL);
#endif
#ifdef USE_CUDSS
    int devices = 0;
    if (cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0) {
        solvers.push_back(homa::LinSysSolverType::GPU_CUDSS);
    }
#endif
    std::vector<Case> cases = {
        {false, {}, {}, {}},
        {true,
         homa::Options::SeparatorMethod::AUTO,
         homa::Options::LocalMethod::AMD,
         homa::Options::PatchMethod::GREEDY},
        {true,
         homa::Options::SeparatorMethod::QUOTIENT,
         homa::Options::LocalMethod::NONE,
         homa::Options::PatchMethod::GREEDY},
        {true,
         homa::Options::SeparatorMethod::DIRECT_METIS,
         homa::Options::LocalMethod::AMD,
         homa::Options::PatchMethod::METIS},
        {true,
         homa::Options::SeparatorMethod::AUTO,
         homa::Options::LocalMethod::METIS,
         homa::Options::PatchMethod::GREEDY},
    };
    for (auto solver : solvers) {
        for (const auto& c : cases) {
            run_case<Scalar>(solver, c);
        }
    }
}


TEST(SolverPermutationRefactorization, Float)
{
    run_all<float>();
}
TEST(SolverPermutationRefactorization, Double)
{
    run_all<double>();
}
