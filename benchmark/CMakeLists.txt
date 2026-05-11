# =============================================================================
# Benchmark CMakeLists.txt - Main entry point for all benchmarks
# =============================================================================

# -----------------------------------------------------------------------------
# benchmark_solvers: STATIC library wrapping all linear-system solver backends.
# CHOLMOD, CUDSS, MKL, STRUMPACK live here; they must NOT be linked into homa.
# -----------------------------------------------------------------------------
add_library(benchmark_solvers STATIC
    LinSysSolvers/LinSysSolver.cpp
    LinSysSolvers/CUDSSSolver.cu
    LinSysSolvers/CHOLMODSolver.cpp
    LinSysSolvers/MKLSolver.cpp
    LinSysSolvers/STRUMPACKSolver.cpp
    get_factor_nnz.cpp
)

target_include_directories(benchmark_solvers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}          # for get_factor_nnz.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LinSysSolvers
    ${CMAKE_SOURCE_DIR}/src/utils        # for compute_inverse_perm.h etc.
)

target_link_libraries(benchmark_solvers PUBLIC
    rxmesh_dev_helpers
    gpu_ordering_deps
)

set_property(TARGET benchmark_solvers PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET benchmark_solvers PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

# -----------------------------------------------------------------------------
# Helper function to configure common benchmark target properties
# -----------------------------------------------------------------------------
function(configure_benchmark_target target_name)
    # Configure CUDA device linking
    set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${target_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # Link benchmark_solvers (solver backends) and rxmesh_dev_helpers (ordering)
    target_link_libraries(${target_name} PUBLIC benchmark_solvers rxmesh_dev_helpers)

    # Disable interprocedural optimization for debug builds
    set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF)

    # Debug compile options
    target_compile_options(${target_name} PUBLIC
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-O0;-Xptxas=-O0;-lineinfo;-Xcompiler=-O0,-g3>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g3;-fno-inline>
    )

    # Configure RPATH for STRUMPACK dependencies if enabled
    if(${PROJECT_NAME}_WITH_STRUMPACK)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "${STRUMPACK_RPATH_DIRS}"
            INSTALL_RPATH "${STRUMPACK_RPATH_DIRS}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
            LINK_FLAGS "-Wl,--disable-new-dtags"
        )
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Add subdirectories for organized benchmark modules
# -----------------------------------------------------------------------------
add_subdirectory(single_factorization)
add_subdirectory(multiple_factorization)
add_subdirectory(render_helper)
add_subdirectory(patch_metrics)