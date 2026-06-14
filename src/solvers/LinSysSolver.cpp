#include <homa/solvers/LinSysSolver.h>
#include <homa/ordering.h>
#include <homa/utils/remove_diagonal.h>
#include <iostream>
#include <stdexcept>

#ifdef USE_CHOLMOD
#include <homa/solvers/CHOLMODSolver.h>
#endif

#ifdef USE_CUDSS
#include <homa/solvers/CUDSSSolver.h>
#endif


#ifdef USE_MKL
#include <homa/solvers/MKLSolver.h>
#endif


namespace homa {

    void LinSysSolver::setMatrix(const Eigen::SparseMatrix<double>& A) {
        if (A.rows() != A.cols()) {
            throw std::invalid_argument("LinSysSolver::setMatrix expects a square matrix");
        }
        if (!A.isCompressed()) {
            throw std::invalid_argument(
                "LinSysSolver::setMatrix expects a compressed Eigen::SparseMatrix; call makeCompressed() first");
        }

        if (type() == LinSysSolverType::CPU_MKL) {
            for (int c = 0; c < A.outerSize(); ++c) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, c); it; ++it) {
                    if (it.row() < it.col()) {
                        throw std::invalid_argument(
                            "LinSysSolver::setMatrix with CPU_MKL expects lower-triangular storage; pass A.triangularView<Eigen::Lower>().makeCompressed()");
                    }
                }
            }
        }

        setMatrix(const_cast<int*>(A.outerIndexPtr()),
                  const_cast<int*>(A.innerIndexPtr()),
                  const_cast<double*>(A.valuePtr()),
                  static_cast<int>(A.rows()),
                  static_cast<int>(A.nonZeros()));
    }

    void LinSysSolver::setMatrix(SparseMatrixView& A) {
        if (A.location != MemoryLocation::Host) {
            throw std::invalid_argument("This solver backend does not accept device matrix pointers");
        }
        if (A.rows <= 0 || A.cols <= 0 || A.rows != A.cols || A.nnz <= 0) {
            throw std::invalid_argument("LinSysSolver::setMatrix received invalid matrix dimensions");
        }
        if (A.outer == nullptr || A.inner == nullptr || A.values == nullptr) {
            throw std::invalid_argument("LinSysSolver::setMatrix received a null matrix pointer");
        }

        setMatrix(A.outer, A.inner, A.values, A.rows, A.nnz);
    }

    void LinSysSolver::recordMatrixPattern(const int*     outer,
                                           const int*     inner,
                                           int            n,
                                           int            nnz,
                                           SparseFormat   format,
                                           MemoryLocation location) {
        if (outer == nullptr || inner == nullptr) {
            throw std::invalid_argument("LinSysSolver::recordMatrixPattern received a null pointer");
        }
        if (n <= 0 || nnz <= 0) {
            throw std::invalid_argument("LinSysSolver::recordMatrixPattern received invalid dimensions");
        }

        matrix_view_ = {n,
                        n,
                        nnz,
                        const_cast<int*>(outer),
                        const_cast<int*>(inner),
                        nullptr,
                        format,
                        location};
        ordering_result = {};
        ordering_applied_ = false;
    }

    void LinSysSolver::copyDeviceMatrixPatternToHost() {
        throw std::runtime_error(
            "LinSysSolver::ordering cannot copy a device matrix pattern for this backend");
    }

    void LinSysSolver::ordering(const Options& opts) {
        if (N <= 0 || matrix_view_.outer == nullptr || matrix_view_.inner == nullptr) {
            throw std::runtime_error("LinSysSolver::ordering called before setMatrix");
        }

        const int* outer = matrix_view_.outer;
        const int* inner = matrix_view_.inner;
        if (matrix_view_.location != MemoryLocation::Host) {
            copyDeviceMatrixPatternToHost();
            outer = owned_host_outer_.data();
            inner = owned_host_inner_.data();
        }
        if (outer == nullptr || inner == nullptr) {
            throw std::runtime_error("LinSysSolver::ordering could not obtain a host matrix pattern");
        }

        std::vector<int> Gp, Gi;
        remove_diagonal(N, outer, inner, Gp, Gi);

        Options local_opts = opts;
        if (type() == LinSysSolverType::GPU_CUDSS) {
            local_opts.compute_etree = true;
        }

        ordering_result = compute_ordering(N, Gp.data(), Gi.data(), local_opts);
        innerOrdering(ordering_result.perm, ordering_result.etree);
        ordering_applied_ = true;
    }

    void LinSysSolver::setOrdering(const OrderingResult& ordering) {
        ordering_result = ordering;
        innerOrdering(ordering_result.perm, ordering_result.etree);
        ordering_applied_ = true;
    }

    void LinSysSolver::innerSolveView(DenseMatrixView& rhs, DenseMatrixView& result) {
        if (rhs.location != MemoryLocation::Host || result.location != MemoryLocation::Host) {
            throw std::invalid_argument("This solver backend only accepts host dense vectors/matrices");
        }
        if (rhs.values == nullptr || result.values == nullptr) {
            throw std::invalid_argument("LinSysSolver::solve received a null dense pointer");
        }
        if (rhs.rows != N || result.rows != N || rhs.cols != result.cols) {
            throw std::invalid_argument("LinSysSolver::solve received incompatible dense dimensions");
        }

        const int rhs_ld = rhs.leading_dim == 0 ? rhs.rows : rhs.leading_dim;
        const int res_ld = result.leading_dim == 0 ? result.rows : result.leading_dim;
        if (rhs_ld != rhs.rows || res_ld != result.rows) {
            throw std::invalid_argument("LinSysSolver::solve expects packed column-major dense views");
        }

        innerSolveRaw(rhs.values, rhs.rows, rhs.cols, result.values);
    }

    LinSysSolver *LinSysSolver::create(const LinSysSolverType type) {
        switch (type) {


#ifdef USE_CHOLMOD
            case LinSysSolverType::CPU_CHOLMOD:
                return new CHOLMODSolver();
#endif

#ifdef USE_CUDSS
            case LinSysSolverType::GPU_CUDSS:
                return new CUDSSSolver();
#endif


#ifdef USE_MKL
            case LinSysSolverType::CPU_MKL:
                return new MKLSolver();
#endif
            default:
                std::cerr << "Uknown linear system solver type" << std::endl;
                return nullptr;
        }
    }


} // namespace homa
