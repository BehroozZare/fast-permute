#ifdef USE_CUDSS

#include <spdlog/spdlog.h>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "homa/solvers/CUDSSSolver.h"
#include "homa/solvers/scalar_traits.h"

#include "homa/utils/CUDA_CHECK_handler.h"

#include <spdlog/spdlog.h>
#include "omp.h"

namespace homa {
namespace {
bool envFlagEnabled(const char* env_name)
{
    const char* value = std::getenv(env_name);
    if (value == nullptr) {
        return false;
    }
    return std::strcmp(value, "1") == 0 || std::strcmp(value, "true") == 0 ||
           std::strcmp(value, "TRUE") == 0 || std::strcmp(value, "on") == 0 ||
           std::strcmp(value, "ON") == 0;
}

#ifdef CUDSS_CONFIG_HOST_NTHREADS
bool parsePositiveIntEnv(const char* env_name, int& out_value)
{
    const char* value = std::getenv(env_name);
    if (value == nullptr || value[0] == '\0') {
        return false;
    }

    char* endptr = nullptr;
    long  parsed = std::strtol(value, &endptr, 10);
    if (endptr == value || *endptr != '\0' || parsed <= 0 || parsed > INT_MAX) {
        spdlog::warn(
            "CUDSSSolver: Ignoring invalid {}='{}' (expected positive integer)",
            env_name,
            value);
        return false;
    }
    out_value = static_cast<int>(parsed);
    return true;
}
#endif
}  // namespace

template <class Scalar>
CUDSSSolver<Scalar>::~CUDSSSolver()
{
    cudssConfigDestroy(config);
    cudssDataDestroy(handle, data);
    clean_sparse_matrix_mem();
    clean_rhs_sol_mem();
    cudssDestroy(handle);
}

template <class Scalar>
CUDSSSolver<Scalar>::CUDSSSolver()
{
    rowOffsets_dev = nullptr;
    colIndices_dev = nullptr;
    values_dev     = nullptr;
    bvalues_dev    = nullptr;
    xvalues_dev    = nullptr;
    x_mat          = nullptr;
    b_mat          = nullptr;
    data           = nullptr;
    handle         = nullptr;
    config         = nullptr;
    A              = nullptr;
    N              = 0;
    NNZ            = 0;
    auto status    = cudssCreate(&handle);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error(
            "CUDSSSolver::constructor - cudssCreate failed with status: {}",
            static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    const bool  mt_strict          = envFlagEnabled("RX_CUDSS_MT_STRICT");
    const char* rx_threading_layer = std::getenv("RX_CUDSS_THREADING_LIB");
    const char* cudss_threading_layer_env = std::getenv("CUDSS_THREADING_LIB");
    if (rx_threading_layer != nullptr || cudss_threading_layer_env != nullptr) {
        // If RX_CUDSS_THREADING_LIB is set, use that explicit path; otherwise
        // let cuDSS read CUDSS_THREADING_LIB internally by passing nullptr.
        const char* layer_path =
            rx_threading_layer != nullptr ? rx_threading_layer : nullptr;
        status = cudssSetThreadingLayer(handle, layer_path);
        if (status == CUDSS_STATUS_SUCCESS) {
            if (rx_threading_layer != nullptr) {
                spdlog::info(
                    "CUDSSSolver: MT mode enabled using "
                    "RX_CUDSS_THREADING_LIB='{}'",
                    rx_threading_layer);
            } else {
                spdlog::info(
                    "CUDSSSolver: MT mode enabled using "
                    "CUDSS_THREADING_LIB='{}'",
                    cudss_threading_layer_env);
            }
        } else if (mt_strict) {
            spdlog::error(
                "CUDSSSolver: Failed to enable MT mode (status: {}) with "
                "strict mode enabled",
                static_cast<int>(status));
            throw std::runtime_error(
                "CUDSSSolver: fatal error (see log above for details)");
        } else {
            spdlog::warn(
                "CUDSSSolver: Failed to enable MT mode (status: {}); falling "
                "back to "
                "single-threaded reordering",
                static_cast<int>(status));
        }
    } else {
        spdlog::info(
            "CUDSSSolver: MT mode not configured (set CUDSS_THREADING_LIB or "
            "RX_CUDSS_THREADING_LIB to enable)");
    }

    status = cudssConfigCreate(&config);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error(
            "CUDSSSolver::constructor - cudssConfigCreate failed with status: "
            "{}",
            static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

#ifdef CUDSS_CONFIG_HOST_NTHREADS
    int host_nthreads = 0;
    if (parsePositiveIntEnv("RX_CUDSS_HOST_NTHREADS", host_nthreads)) {
        status = cudssConfigSet(config,
                                CUDSS_CONFIG_HOST_NTHREADS,
                                &host_nthreads,
                                sizeof(host_nthreads));
        if (status == CUDSS_STATUS_SUCCESS) {
            spdlog::info(
                "CUDSSSolver: Host MT thread cap set to {} via "
                "RX_CUDSS_HOST_NTHREADS",
                host_nthreads);
        } else if (mt_strict) {
            spdlog::error(
                "CUDSSSolver: Failed to set CUDSS_CONFIG_HOST_NTHREADS "
                "(status: {}) with "
                "strict mode enabled",
                static_cast<int>(status));
            throw std::runtime_error(
                "CUDSSSolver: fatal error (see log above for details)");
        } else {
            spdlog::warn(
                "CUDSSSolver: Failed to set CUDSS_CONFIG_HOST_NTHREADS "
                "(status: {}); "
                "continuing with cuDSS default thread policy",
                static_cast<int>(status));
        }
    }
#else
    if (std::getenv("RX_CUDSS_HOST_NTHREADS") != nullptr) {
        spdlog::warn(
            "CUDSSSolver: RX_CUDSS_HOST_NTHREADS is set but this cuDSS build "
            "does not expose "
            "CUDSS_CONFIG_HOST_NTHREADS");
    }
#endif

    status = cudssDataCreate(handle, &data);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error(
            "CUDSSSolver::constructor - cudssDataCreate failed with status: {}",
            static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    is_allocated           = false;
    owns_sparse_matrix_mem = false;
    owns_bvalues_mem       = false;
    owns_xvalues_mem       = false;
}

template <class Scalar>
void CUDSSSolver<Scalar>::clean_sparse_matrix_mem()
{
    if (owns_sparse_matrix_mem && rowOffsets_dev != nullptr)
        CUDA_CHECK(cudaFree(rowOffsets_dev));
    if (owns_sparse_matrix_mem && colIndices_dev != nullptr)
        CUDA_CHECK(cudaFree(colIndices_dev));
    if (owns_sparse_matrix_mem && values_dev != nullptr)
        CUDA_CHECK(cudaFree(values_dev));
    if (A != nullptr)
        cudssMatrixDestroy(A);
    rowOffsets_dev         = nullptr;
    colIndices_dev         = nullptr;
    values_dev             = nullptr;
    A                      = nullptr;
    is_allocated           = false;
    owns_sparse_matrix_mem = false;
}

template <class Scalar>
void CUDSSSolver<Scalar>::clean_rhs_sol_mem()
{
    if (x_mat != nullptr)
        cudssMatrixDestroy(x_mat);
    if (b_mat != nullptr)
        cudssMatrixDestroy(b_mat);
    if (owns_xvalues_mem && xvalues_dev != nullptr)
        CUDA_CHECK(cudaFree(xvalues_dev));
    if (owns_bvalues_mem && bvalues_dev != nullptr)
        CUDA_CHECK(cudaFree(bvalues_dev));
    xvalues_dev      = nullptr;
    bvalues_dev      = nullptr;
    x_mat            = nullptr;
    b_mat            = nullptr;
    owns_bvalues_mem = false;
    owns_xvalues_mem = false;
}


template <class Scalar>
void CUDSSSolver<Scalar>::setMatrix(int*    p,
                                    int*    i,
                                    Scalar* x,
                                    int     A_N,
                                    int     NNZ_in)
{
    // Validate input parameters
    if (A_N <= 0 || NNZ_in <= 0) {
        spdlog::error(
            "CUDSSSolver::setMatrix - Invalid dimensions: N={}, NNZ={}",
            A_N,
            NNZ_in);
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }
    if (p == nullptr || i == nullptr || x == nullptr) {
        spdlog::error("CUDSSSolver::setMatrix - Null pointer passed");
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    // Log matrix properties for debugging

    spdlog::info("CUDSSSolver::setMatrix - N={}, NNZ={}", A_N, NNZ_in);
    recordMatrixPattern(
        p, i, A_N, NNZ_in, SparseFormat::CSR, MemoryLocation::Host);
    // Check if reallocation is needed BEFORE updating member variables
    bool needs_realloc = !is_allocated || !owns_sparse_matrix_mem ||

                         this->NNZ != NNZ_in || this->N != A_N;
    this->N   = A_N;
    this->NNZ = NNZ_in;

    // Allocating memory
    if (needs_realloc) {
        clean_sparse_matrix_mem();
        CUDA_CHECK(
            cudaMalloc((void**)&rowOffsets_dev, (A_N + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&colIndices_dev, NNZ_in * sizeof(int)));
        CUDA_CHECK(cudaMalloc((void**)&values_dev, NNZ_in * sizeof(Scalar)));
        is_allocated           = true;
        owns_sparse_matrix_mem = true;
    } else if (A != nullptr) {
        cudssMatrixDestroy(A);
        A = nullptr;
    }


    // Copying data to device
    CUDA_CHECK(cudaMemcpy(
        rowOffsets_dev, p, (A_N + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        colIndices_dev, i, NNZ_in * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(
        values_dev, x, NNZ_in * sizeof(Scalar), cudaMemcpyHostToDevice));

    // Creating matrix
    // Note: Eigen::SparseMatrix is CSC (column-major) by default
    // The caller should expand symmetric matrices to full format using
    // selfadjointView
    auto status = cudssMatrixCreateCsr(
        &A,
        N,                              // nrows
        N,                              // ncols
        NNZ,                            // nnz
        rowOffsets_dev,                 // csrRowOffsets
        nullptr,                        // csrRowOffsetsEnd (optional)
        colIndices_dev,                 // csrColInd
        values_dev,                     // csrValues
        CUDA_R_32I,                     // csrRowOffsetsType
        detail::cudss_dtype_v<Scalar>,  // csrValuesType
        CUDSS_MTYPE_SPD,   // matrixType (Symmetric Positive Definite)
        CUDSS_MVIEW_FULL,  // viewType (full matrix expected)
        CUDSS_BASE_ZERO);  // indexBase
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::matrix creation failed with status: {}",
                      static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }
}

template <class Scalar>
void CUDSSSolver<Scalar>::setMatrix(SparseMatrixView<Scalar>& matrix)
{
    if (matrix.location == MemoryLocation::Host) {
        setMatrix(
            matrix.outer, matrix.inner, matrix.values, matrix.rows, matrix.nnz);
        return;
    }

    if (matrix.rows <= 0 || matrix.cols <= 0 || matrix.rows != matrix.cols ||
        matrix.nnz <= 0) {
        spdlog::error(
            "CUDSSSolver::setMatrix(device) - Invalid dimensions: {}x{}, "
            "NNZ={}",
            matrix.rows,
            matrix.cols,
            matrix.nnz);
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }
    if (matrix.outer == nullptr || matrix.inner == nullptr ||
        matrix.values == nullptr) {
        spdlog::error("CUDSSSolver::setMatrix(device) - Null pointer passed");
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }
    if (matrix.format != SparseFormat::CSR) {
        throw std::invalid_argument(
            "CUDSSSolver::setMatrix(device) expects CSR device matrix views");
    }

    clean_sparse_matrix_mem();

    this->N   = matrix.rows;
    this->NNZ = matrix.nnz;
    recordMatrixPattern(matrix.outer,
                        matrix.inner,
                        matrix.rows,
                        matrix.nnz,
                        matrix.format,
                        MemoryLocation::Device);
    rowOffsets_dev         = matrix.outer;
    colIndices_dev         = matrix.inner;
    values_dev             = matrix.values;
    is_allocated           = true;
    owns_sparse_matrix_mem = false;

    auto status = cudssMatrixCreateCsr(&A,
                                       N,
                                       N,
                                       NNZ,
                                       rowOffsets_dev,
                                       nullptr,
                                       colIndices_dev,
                                       values_dev,
                                       CUDA_R_32I,
                                       detail::cudss_dtype_v<Scalar>,
                                       CUDSS_MTYPE_SPD,
                                       CUDSS_MVIEW_FULL,
                                       CUDSS_BASE_ZERO);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error(
            "CUDSSSolver::device matrix creation failed with status: {}",
            static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }
}

template <class Scalar>
void CUDSSSolver<Scalar>::copyDeviceMatrixPatternToHost()
{
    if (matrix_view_.location != MemoryLocation::Device) {
        return;
    }

    if (matrix_view_.outer == nullptr || matrix_view_.inner == nullptr ||
        N <= 0 || matrix_view_.nnz <= 0) {
        throw std::runtime_error(
            "CUDSSSolver::ordering cannot copy an invalid device matrix "
            "pattern");
    }

    owned_host_outer_.resize(static_cast<size_t>(N) + 1);
    owned_host_inner_.resize(static_cast<size_t>(matrix_view_.nnz));

    CUDA_CHECK(cudaMemcpy(owned_host_outer_.data(),
                          matrix_view_.outer,
                          (static_cast<size_t>(N) + 1) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(owned_host_inner_.data(),
                          matrix_view_.inner,
                          static_cast<size_t>(matrix_view_.nnz) * sizeof(int),
                          cudaMemcpyDeviceToHost));
}

template <class Scalar>
void CUDSSSolver<Scalar>::innerOrdering(std::vector<int>& user_defined_perm,
                                        std::vector<int>& etree)
{
    assert(is_allocated);
    cudssStatus_t status;

    if (static_cast<int>(user_defined_perm.size()) == N && etree.size() > 0) {
#ifndef NDEBUG
        int sum = 0;
        for (auto& e : etree) {
            sum += e;
        }
        assert(sum == N);
#endif
        // REUSE MODE: Both perm and etree provided
        spdlog::info(
            "CUDSS: Reusing user-defined permutation (size={}) and etree "
            "(size={})",
            N,
            etree.size());


        // Set user permutation (device pointer)
        status = cudssDataSet(handle,
                              data,
                              CUDSS_DATA_USER_PERM,
                              user_defined_perm.data(),
                              N * sizeof(int));
        if (status != CUDSS_STATUS_SUCCESS) {
            spdlog::error(
                "CUDSSSolver::cudssDataSet for user permutation failed with "
                "status: {}",
                static_cast<int>(status));
            throw std::runtime_error(
                "CUDSSSolver: fatal error (see log above for details)");
        }

        // Set user elimination tree (HOST pointer per CUDSS docs)
        status = cudssDataSet(handle,
                              data,
                              CUDSS_DATA_USER_ELIMINATION_TREE,
                              etree.data(),
                              etree.size() * sizeof(int));
        if (status != CUDSS_STATUS_SUCCESS) {
            spdlog::error(
                "CUDSSSolver::cudssDataSet for user elimination tree failed "
                "with status: {}",
                static_cast<int>(status));
            throw std::runtime_error(
                "CUDSSSolver: fatal error (see log above for details)");
        }


        // Compute the number of levels in the etree
        int level =
            static_cast<int>(std::log2(static_cast<double>(etree.size() + 1)));
        spdlog::info("CUDSS: Number of levels in the etree: {}", level);
        // Set the number of levels in the etree
        status = cudssConfigSet(
            config, CUDSS_CONFIG_ND_NLEVELS, &level, sizeof(int));
        if (status != CUDSS_STATUS_SUCCESS) {
            spdlog::error(
                "CUDSSSolver::cudssConfigSet for ND_NLEVELS failed with "
                "status: {}",
                static_cast<int>(status));
            throw std::runtime_error(
                "CUDSSSolver: fatal error (see log above for details)");
        }

        // Execute reordering phase (cuDSS still needs this to process user
        // perm/etree)
        status = cudssExecute(
            handle, CUDSS_PHASE_REORDERING, config, data, A, nullptr, nullptr);
        if (status != CUDSS_STATUS_SUCCESS) {
            spdlog::error("CUDSSSolver::reordering failed with status: {}",
                          static_cast<int>(status));
            throw std::runtime_error(
                "CUDSSSolver: fatal error (see log above for details)");
        }
    } else {
        // DEFAULT MODE: Let CUDSS handle reordering
        spdlog::info("CUDSS: Using default reordering");

        // Run both reordering and symbolic factorization
        status = cudssExecute(
            handle, CUDSS_PHASE_REORDERING, config, data, A, nullptr, nullptr);
        if (status != CUDSS_STATUS_SUCCESS) {
            spdlog::error("CUDSSSolver::reordering failed with status: {}",
                          static_cast<int>(status));
            throw std::runtime_error(
                "CUDSSSolver: fatal error (see log above for details)");
        }
    }
}


template <class Scalar>
void CUDSSSolver<Scalar>::innerAnalyze_pattern(
    std::vector<int>& user_defined_perm,
    std::vector<int>& etree)
{
    assert(is_allocated);
    cudssStatus_t status;
    // Execute symbolic factorization
    status = cudssExecute(handle,
                          CUDSS_PHASE_SYMBOLIC_FACTORIZATION,
                          config,
                          data,
                          A,
                          nullptr,
                          nullptr);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error(
            "CUDSSSolver::symbolic factorization failed with status: {}",
            static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }
}

template <class Scalar>
void CUDSSSolver<Scalar>::innerFactorize(void)
{
    assert(is_allocated);
    auto status = cudssExecute(
        handle, CUDSS_PHASE_FACTORIZATION, config, data, A, nullptr, nullptr);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::factorization failed!");
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }
}


template <class Scalar>
void CUDSSSolver<Scalar>::innerSolve(typename CUDSSSolver<Scalar>::Mat& rhs,
                                     typename CUDSSSolver<Scalar>::Mat& result)
{
    // Delegate to raw pointer version to avoid ABI issues
    result.resize(rhs.rows(), rhs.cols());
    innerSolveRaw(rhs.data(),
                  static_cast<int>(rhs.rows()),
                  static_cast<int>(rhs.cols()),
                  result.data());
}

template <class Scalar>
void CUDSSSolver<Scalar>::innerSolveRaw(const Scalar* rhs_data,
                                        int           rows,
                                        int           cols,
                                        Scalar*       result_data)
{
    // Validate input dimensions
    if (rows != N) {
        spdlog::error(
            "CUDSSSolver::solve - RHS rows {} doesn't match matrix size {}",
            rows,
            N);
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    int nrhs = cols;
    spdlog::info("CUDSSSolver::solve - Matrix size: {}, nrhs: {}", N, nrhs);

    // Clean up any existing memory
    clean_rhs_sol_mem();

    // Allocate device memory for N × nrhs matrices
    CUDA_CHECK(cudaMalloc((void**)&bvalues_dev, N * nrhs * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc((void**)&xvalues_dev, N * nrhs * sizeof(Scalar)));
    owns_bvalues_mem = true;
    owns_xvalues_mem = true;

    // Copy RHS to device (data is column-major)
    CUDA_CHECK(cudaMemcpy(bvalues_dev,
                          rhs_data,
                          N * nrhs * sizeof(Scalar),
                          cudaMemcpyHostToDevice));

    // Create RHS dense matrix (N × nrhs, leading dimension = N)
    auto status = cudssMatrixCreateDn(&b_mat,
                                      N,
                                      nrhs,
                                      N,
                                      bvalues_dev,
                                      detail::cudss_dtype_v<Scalar>,
                                      CUDSS_LAYOUT_COL_MAJOR);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::RHS matrix creation failed with status: {}",
                      static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    // Create solution dense matrix (N × nrhs, leading dimension = N)
    status = cudssMatrixCreateDn(&x_mat,
                                 N,
                                 nrhs,
                                 N,
                                 xvalues_dev,
                                 detail::cudss_dtype_v<Scalar>,
                                 CUDSS_LAYOUT_COL_MAJOR);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error(
            "CUDSSSolver::solution matrix creation failed with status: {}",
            static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    // Execute solve phase
    status =
        cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, x_mat, b_mat);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::solve failed with status: {}",
                      static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(result_data,
                          xvalues_dev,
                          N * nrhs * sizeof(Scalar),
                          cudaMemcpyDeviceToHost));

    clean_rhs_sol_mem();
}

template <class Scalar>
void CUDSSSolver<Scalar>::innerSolveView(DenseMatrixView<Scalar>& rhs,
                                         DenseMatrixView<Scalar>& result)
{
    if (rhs.rows != N || result.rows != N || rhs.cols != result.cols) {
        spdlog::error("CUDSSSolver::solve - incompatible dense dimensions");
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }
    if (rhs.values == nullptr || result.values == nullptr) {
        spdlog::error("CUDSSSolver::solve - Null dense pointer passed");
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    const int nrhs   = rhs.cols;
    const int rhs_ld = rhs.leading_dim == 0 ? rhs.rows : rhs.leading_dim;
    const int res_ld =
        result.leading_dim == 0 ? result.rows : result.leading_dim;

    if (rhs.location == MemoryLocation::Host && rhs_ld != rhs.rows) {
        throw std::invalid_argument(
            "CUDSSSolver::solve host RHS views must be packed column-major");
    }
    if (result.location == MemoryLocation::Host && res_ld != result.rows) {
        throw std::invalid_argument(
            "CUDSSSolver::solve host result views must be packed column-major");
    }

    spdlog::info("CUDSSSolver::solve - Matrix size: {}, nrhs: {}", N, nrhs);

    clean_rhs_sol_mem();

    if (rhs.location == MemoryLocation::Device) {
        bvalues_dev      = rhs.values;
        owns_bvalues_mem = false;
    } else {
        CUDA_CHECK(cudaMalloc((void**)&bvalues_dev, N * nrhs * sizeof(Scalar)));
        owns_bvalues_mem = true;
        CUDA_CHECK(cudaMemcpy(bvalues_dev,
                              rhs.values,
                              N * nrhs * sizeof(Scalar),
                              cudaMemcpyHostToDevice));
    }

    if (result.location == MemoryLocation::Device) {
        xvalues_dev      = result.values;
        owns_xvalues_mem = false;
    } else {
        CUDA_CHECK(cudaMalloc((void**)&xvalues_dev, N * nrhs * sizeof(Scalar)));
        owns_xvalues_mem = true;
    }

    auto status = cudssMatrixCreateDn(&b_mat,
                                      N,
                                      nrhs,
                                      rhs_ld,
                                      bvalues_dev,
                                      detail::cudss_dtype_v<Scalar>,
                                      CUDSS_LAYOUT_COL_MAJOR);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::RHS matrix creation failed with status: {}",
                      static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    status = cudssMatrixCreateDn(&x_mat,
                                 N,
                                 nrhs,
                                 res_ld,
                                 xvalues_dev,
                                 detail::cudss_dtype_v<Scalar>,
                                 CUDSS_LAYOUT_COL_MAJOR);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error(
            "CUDSSSolver::solution matrix creation failed with status: {}",
            static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    status =
        cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, x_mat, b_mat);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::solve failed with status: {}",
                      static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    if (result.location == MemoryLocation::Host) {
        CUDA_CHECK(cudaMemcpy(result.values,
                              xvalues_dev,
                              N * nrhs * sizeof(Scalar),
                              cudaMemcpyDeviceToHost));
    }

    clean_rhs_sol_mem();
}

template <class Scalar>
void CUDSSSolver<Scalar>::innerSolve(typename CUDSSSolver<Scalar>::Vec& rhs,
                                     typename CUDSSSolver<Scalar>::Vec& result)
{
    // Validate input dimensions
    if (rhs.size() != N) {
        spdlog::error(
            "CUDSSSolver::solve - RHS size {} doesn't match matrix size {}",
            rhs.size(),
            N);
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    spdlog::info(
        "CUDSSSolver::solve - Matrix size: {}, RHS size: {}", N, rhs.size());

    // Clean up any existing memory first
    clean_rhs_sol_mem();
    CUDA_CHECK(cudaMalloc((void**)&bvalues_dev, N * sizeof(Scalar)));
    CUDA_CHECK(cudaMalloc((void**)&xvalues_dev, N * sizeof(Scalar)));
    owns_bvalues_mem = true;
    owns_xvalues_mem = true;
    // Copying the memory
    CUDA_CHECK(cudaMemcpy(
        bvalues_dev, rhs.data(), N * sizeof(Scalar), cudaMemcpyHostToDevice));
    // Creating the rhs matrix (N×1 column vector)
    spdlog::info("Creating RHS dense matrix: N={}, ncols=1, ld={}", N, N);
    auto status = cudssMatrixCreateDn(&b_mat,
                                      N,
                                      1,
                                      N,
                                      bvalues_dev,
                                      detail::cudss_dtype_v<Scalar>,
                                      CUDSS_LAYOUT_COL_MAJOR);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::RHS matrix creation failed with status: {}",
                      static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    // Creating solution matrix (N×1 column vector)
    spdlog::info("Creating solution dense matrix: N={}, ncols=1, ld={}", N, N);
    status = cudssMatrixCreateDn(&x_mat,
                                 N,
                                 1,
                                 N,
                                 xvalues_dev,
                                 detail::cudss_dtype_v<Scalar>,
                                 CUDSS_LAYOUT_COL_MAJOR);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error(
            "CUDSSSolver::solution matrix creation failed with status: {}",
            static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    // Execute solve phase
    spdlog::info("Executing CUDSS solve phase...");
    status =
        cudssExecute(handle, CUDSS_PHASE_SOLVE, config, data, A, x_mat, b_mat);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error("CUDSSSolver::solve failed with status: {} ",
                      static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }
    spdlog::info("CUDSS solve completed successfully");
    // Copy the result back
    result.conservativeResize(rhs.size());
    CUDA_CHECK(cudaMemcpy(result.data(),
                          xvalues_dev,
                          result.rows() * sizeof(Scalar),
                          cudaMemcpyDeviceToHost));
    clean_rhs_sol_mem();
}

template <class Scalar>
void CUDSSSolver<Scalar>::resetSolver()
{
    // Destroy old data object (contains stale symbolic factorization)
    if (data != nullptr) {
        cudssDataDestroy(handle, data);
        data = nullptr;
    }

    // Clean matrix and RHS/solution memory
    clean_sparse_matrix_mem();
    clean_rhs_sol_mem();

    // Recreate data object for new matrix structure
    auto status = cudssDataCreate(handle, &data);
    if (status != CUDSS_STATUS_SUCCESS) {
        spdlog::error(
            "CUDSSSolver::resetSolver - cudssDataCreate failed with status: {}",
            static_cast<int>(status));
        throw std::runtime_error(
            "CUDSSSolver: fatal error (see log above for details)");
    }

    Base::initVariables();

    spdlog::info("CUDSSSolver::resetSolver - Solver state reset successfully");
}

template <class Scalar>
LinSysSolverType CUDSSSolver<Scalar>::type() const
{
    return LinSysSolverType::GPU_CUDSS;
}

template <class Scalar>
int CUDSSSolver<Scalar>::getFactorNNZ()
{
    return 0;
}

template class CUDSSSolver<float>;
template class CUDSSSolver<double>;
}  // namespace homa

#endif
