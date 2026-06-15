//
//  CHOLMODSolver.cpp
//  IPC
//
//  Created by Minchen Li on 6/22/18.
//

#ifdef USE_CHOLMOD

#include <homa/solvers/CHOLMODSolver.h>
#include "scalar_traits.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include "omp.h"
#include <spdlog/spdlog.h>

namespace homa {

template <class Scalar>
CHOLMODSolver<Scalar>::~CHOLMODSolver()
{
    if (use_gpu) {
        // GPU mode cleanup with int64 functions
        if (A) {
            A->i = Ai;
            A->p = Ap;
            A->x = Ax;
            cholmod_l_free_sparse(&A, &cm);
        }

        if (L) {
            cholmod_l_free_factor(&L, &cm);
        }

        if (b) {
            b->x = bx;
            cholmod_l_free_dense(&b, &cm);
        }

        if (x_solve) {
            cholmod_l_free_dense(&x_solve, &cm);
        }

        cholmod_l_finish(&cm);
    } else {
        // CPU mode cleanup with regular functions
        if (A) {
            A->i = Ai;
            A->p = Ap;
            A->x = Ax;
            cholmod_free_sparse(&A, &cm);
        }

        if (L) {
            cholmod_free_factor(&L, &cm);
        }

        if (b) {
            b->x = bx;
            cholmod_free_dense(&b, &cm);
        }

        if (x_solve) {
            cholmod_free_dense(&x_solve, &cm);
        }

        cholmod_finish(&cm);
    }
}

template <class Scalar>
CHOLMODSolver<Scalar>::CHOLMODSolver()
{
    // Use int64 version for GPU mode, regular for CPU mode
    if (use_gpu) {
        cholmod_l_start(&cm);
        spdlog::info("CHOLMOD initialized in GPU mode (int64)");
    } else {
        cholmod_start(&cm);
        spdlog::info("CHOLMOD initialized in CPU-only mode");
    }

    // Precision is selected per matrix via xdtype on allocate_* calls.
    bx      = NULL;
    A       = NULL;
    L       = NULL;
    b       = NULL;
    x_solve = NULL;
    Ai = Ap = Ax = NULL;
    
    //If GPU mode requested, initialize and probe GPU
    if (use_gpu) {
        initializeGPU();
        if (!probeGPU()) {
            // GPU requested but not available - throw error (ungraceful)
            spdlog::error("GPU mode requested but GPU not available");
            throw std::runtime_error("GPU not available for CHOLMOD - GPU mode was requested but GPU is not available");
        }
    } else {
        spdlog::info("CPU-only mode - GPU support not required");
    }
}

template <class Scalar>
void CHOLMODSolver<Scalar>::initializeGPU()
{
    // Enable GPU usage for CHOLMOD
    cm.useGPU = 1;
    // Force supernodal factorization for GPU (required for GPU acceleration)
    cm.supernodal = CHOLMOD_SUPERNODAL;
    // Set additional GPU-friendly parameters
    cm.print = 3;  // Print more diagnostic info
    
    // Set numerical tolerance parameters to help with potential precision issues
    // GPU computations can be more sensitive to numerical issues
    cm.final_ll = 1;  // Force LL' factorization (more stable than LDL')
    
    spdlog::info("CHOLMOD GPU mode enabled with supernodal LL' factorization");
}

template <class Scalar>
bool CHOLMODSolver<Scalar>::probeGPU()
{
    // Probe for GPU availability
    #if defined(CHOLMOD_HAS_CUDA)
    int gpu_status = cholmod_l_gpu_probe(&cm);
    if (gpu_status == 1) {        
        spdlog::info("CHOLMOD GPU detected and available");
        // Allocate GPU memory
        cholmod_l_gpu_allocate(&cm);
        spdlog::info("CHOLMOD GPU memory allocated");
        return true;
    } else {
        spdlog::error("CHOLMOD GPU not available (probe returned {})", gpu_status);
        return false;
    }
    #else
    spdlog::error("CHOLMOD was not compiled with CUDA support");
    return false;
    #endif
}

template <class Scalar>
void CHOLMODSolver<Scalar>::cholmod_clean_memory()
{
    if (use_gpu) {
        // GPU mode cleanup with int64 functions
        if (A) {
            A->i = Ai;
            A->p = Ap;
            A->x = Ax;
            cholmod_l_free_sparse(&A, &cm);
        }

        if (b) {
            b->x = bx;
            cholmod_l_free_dense(&b, &cm);
        }

        if (x_solve) {
            cholmod_l_free_dense(&x_solve, &cm);
        }
    } else {
        // CPU mode cleanup with regular functions
        if (A) {
            A->i = Ai;
            A->p = Ap;
            A->x = Ax;
            cholmod_free_sparse(&A, &cm);
        }

        if (b) {
            b->x = bx;
            cholmod_free_dense(&b, &cm);
        }

        if (x_solve) {
            cholmod_free_dense(&x_solve, &cm);
        }
    }

    A       = NULL;
    b       = NULL;
    x_solve = NULL;
    Ai = Ap = Ax = NULL;
}

template <class Scalar>
void CHOLMODSolver<Scalar>::setMatrix(int*    p,
                                      int*    i,
                                      Scalar* x,
                                      int     A_N,
                                      int     NNZ_in)
{
    assert(p[A_N] == NNZ_in);
    recordMatrixPattern(p, i, A_N, NNZ_in, SparseFormat::CSC, MemoryLocation::Host);
    this->N   = A_N;
    this->NNZ = NNZ_in;

    this->cholmod_clean_memory();

    if (!A) {
        if (use_gpu) {
            A = cholmod_l_allocate_sparse(
                N, N, NNZ, true, true, -1, detail::cholmod_xdtype_v<Scalar>, &cm);
            //Convert the values in p and i to long int (int64_t)
            p_long.resize(N + 1);
            i_long.resize(NNZ);
            
            for (int idx = 0; idx < N + 1; idx++) {
                p_long[idx] = static_cast<long int>(p[idx]);
            }
            
            for (int idx = 0; idx < NNZ; idx++) {
                i_long[idx] = static_cast<long int>(i[idx]);
            }

            this->Ap = A->p;
            this->Ax = A->x;
            this->Ai = A->i;

            A->p = p_long.data();
            A->i = i_long.data();
            A->x = x;
            
            spdlog::info("Matrix allocated for GPU mode: {}x{}, NNZ={}, stype=-1 (lower triangular)", N, N, NNZ);
            
            // Verify matrix is properly sorted (required for CHOLMOD)
            int status = cholmod_l_check_sparse(A, &cm);
            if (status == 0) {
                spdlog::error("Matrix check failed! Matrix may not be properly formed");
            } else {
                spdlog::info("Matrix check passed (GPU mode)");
            }
        } else {
            A = cholmod_allocate_sparse(
                N, N, NNZ, true, true, -1, detail::cholmod_xdtype_v<Scalar>, &cm);

            this->Ap = A->p;
            this->Ax = A->x;
            this->Ai = A->i;

            A->p = p;
            A->i = i;
            A->x = x;
            
            spdlog::info("Matrix allocated for CPU mode: {}x{}, NNZ={}", N, N, NNZ);
        }

        // -1: upper right part will be ignored during computation (stype = -1 means lower triangular stored)
    }


}

template <class Scalar>
void CHOLMODSolver<Scalar>::innerAnalyze_pattern(std::vector<int>& user_defined_perm, std::vector<int>& etree)
{
    if (use_gpu) {
        cholmod_l_free_factor(&L, &cm);
        std::vector<long int> long_user_defined_perm(user_defined_perm.size());
        for (size_t i = 0; i < user_defined_perm.size(); i++) {
            long_user_defined_perm[i] = user_defined_perm[i];
        }

        // Ensure GPU and supernodal settings are still active
        cm.useGPU = 1;
        cm.supernodal = CHOLMOD_SUPERNODAL;
        
        spdlog::info("Matrix size: {}, NNZ: {} (GPU mode)", N, NNZ);
        
        if (static_cast<int>(user_defined_perm.size()) == N) {
            spdlog::info("Using user provided permutation (GPU mode)");
            cm.nmethods           = 1;
            cm.method[0].ordering = CHOLMOD_GIVEN;
            L                     = cholmod_l_analyze_p(A, (int64_t*)long_user_defined_perm.data(), NULL, 0, &cm);
        } else {
            spdlog::info("Using METIS permutation (GPU mode)");
            cm.nmethods           = 1;
            cm.method[0].ordering = CHOLMOD_METIS;
            L                     = cholmod_l_analyze(A, &cm);
        }
        
        spdlog::info("CHOLMOD symbolic analysis complete with GPU enabled");
    } else {
        cholmod_free_factor(&L, &cm);
        
        cm.useGPU = 0;
        cm.supernodal = CHOLMOD_SUPERNODAL;
        
        if (static_cast<int>(user_defined_perm.size()) == N) {
            spdlog::info("Using user provided permutation (CPU mode)");
            cm.nmethods           = 1;
            cm.method[0].ordering = CHOLMOD_GIVEN;
            L                     = cholmod_analyze_p(A, user_defined_perm.data(), NULL, 0, &cm);
        } else {
            spdlog::info("Using METIS permutation (CPU mode)");
            cm.nmethods           = 1;
            cm.method[0].ordering = CHOLMOD_METIS;
            L                     = cholmod_analyze(A, &cm);
        }
        
        spdlog::info("CHOLMOD symbolic analysis complete in CPU mode");
    }
    
    assert(L != nullptr);
    if (L == nullptr) {
        std::cerr << "ERROR during symbolic factorization:" << std::endl;
        throw std::runtime_error("Symbolic factorization failed");
    }
    L_NNZ = static_cast<int>(cm.lnz * 2 - N);
}

template <class Scalar>
void CHOLMODSolver<Scalar>::innerFactorize(void)
{
    if (use_gpu) {
        cholmod_l_factorize(A, L, &cm);
    } else {
        cholmod_factorize(A, L, &cm);
    }
    
    if (cm.status == CHOLMOD_NOT_POSDEF) {
        std::cerr << "ERROR during numerical factorization - matrix not positive definite"
                  << std::endl;
        std::cerr << "Matrix size: " << N << ", NNZ: " << NNZ << std::endl;
        if (use_gpu) {
            std::cerr << "GPU mode was enabled" << std::endl;
            #if defined(CHOLMOD_HAS_CUDA)
            spdlog::error("Printing GPU statistics:");
            cholmod_l_gpu_stats(&cm);
            #endif
        }
        throw std::runtime_error("Numerical factorization failed - matrix not positive definite");
    }
    
    //A bit of debugging flags
    if (L->is_super) {
        if (use_gpu) {
            spdlog::info("CHOLMOD Choose Supernodal computation (GPU accelerated)");
            #if defined(CHOLMOD_HAS_CUDA)
            spdlog::info("Printing GPU statistics:");
            cholmod_l_gpu_stats(&cm);
            #endif
        } else {
            spdlog::info("CHOLMOD Choose Supernodal computation (CPU)");
        }
    } else {
        spdlog::info("CHOLMOD Choose simplicial computation");
    }
}

template <class Scalar>
void CHOLMODSolver<Scalar>::innerSolve(typename CHOLMODSolver<Scalar>::Mat& rhs,
                                       typename CHOLMODSolver<Scalar>::Mat& result)
{
    // Delegate to raw pointer version
    result.resize(rhs.rows(), rhs.cols());
    innerSolveRaw(rhs.data(), static_cast<int>(rhs.rows()), static_cast<int>(rhs.cols()), result.data());
}

template <class Scalar>
void CHOLMODSolver<Scalar>::innerSolveRaw(const Scalar* rhs_data, int rows, int cols, Scalar* result_data)
{
    // Solve column by column since CHOLMOD doesn't have built-in multi-RHS support
    for (int c = 0; c < cols; c++) {
        // Map column c of input (column-major layout)
        Vec rhs_col = Eigen::Map<const Vec>(rhs_data + c * rows, rows);
        Vec result_col(rows);

        innerSolve(rhs_col, result_col);

        // Copy result to output column
        std::memcpy(result_data + c * rows, result_col.data(), rows * sizeof(Scalar));
    }
}

template <class Scalar>
void CHOLMODSolver<Scalar>::innerSolve(typename CHOLMODSolver<Scalar>::Vec& rhs,
                                       typename CHOLMODSolver<Scalar>::Vec& result)
{
    if (use_gpu) {
        if (!b) {
            b  = cholmod_l_allocate_dense(N, 1, N, detail::cholmod_xdtype_v<Scalar>, &cm);
            bx = b->x;
        }
        b->x = rhs.data();

        if (x_solve) {
            cholmod_l_free_dense(&x_solve, &cm);
        }

        x_solve = cholmod_l_solve(CHOLMOD_A, L, b, &cm);
    } else {
        if (!b) {
            b  = cholmod_allocate_dense(N, 1, N, detail::cholmod_xdtype_v<Scalar>, &cm);
            bx = b->x;
        }
        b->x = rhs.data();

        if (x_solve) {
            cholmod_free_dense(&x_solve, &cm);
        }

        x_solve = cholmod_solve(CHOLMOD_A, L, b, &cm);
    }

    result.conservativeResize(rhs.size());
    std::memcpy(result.data(), x_solve->x, result.rows() * sizeof(Scalar));
}


template <class Scalar>
void CHOLMODSolver<Scalar>::resetSolver()
{
    cholmod_clean_memory();

    A       = NULL;
    L       = NULL;
    b       = NULL;
    x_solve = NULL;
    Ai = Ap = Ax = NULL;
    bx           = NULL;

    Base::initVariables();
}

template <class Scalar>
void CHOLMODSolver<Scalar>::save_factor(
    const std::string &filePath) {
    cholmod_sparse *spm;
    
    if (use_gpu) {
        spm = cholmod_l_factor_to_sparse(L, &cm);
    } else {
        spm = cholmod_factor_to_sparse(L, &cm);
    }

    FILE *out = fopen(filePath.c_str(), "w");
    assert(out);

    if (use_gpu) {
        cholmod_l_write_sparse(out, spm, NULL, "", &cm);
    } else {
        cholmod_write_sparse(out, spm, NULL, "", &cm);
    }

    fclose(out);
}

template <class Scalar>
LinSysSolverType CHOLMODSolver<Scalar>::type() const
{
    return LinSysSolverType::CPU_CHOLMOD;
}

template class CHOLMODSolver<float>;
template class CHOLMODSolver<double>;

}  // namespace homa

#endif
