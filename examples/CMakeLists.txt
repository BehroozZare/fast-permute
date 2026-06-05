# =============================================================================
# Examples CMakeLists.txt
# =============================================================================

option(HOMA_EXAMPLE_WITH_CHOLMOD "Build CHOLMOD example" OFF)
option(HOMA_EXAMPLE_WITH_MKL     "Build MKL example"    OFF)
option(HOMA_EXAMPLE_WITH_CUDSS   "Build cuDSS example"  OFF)

# -----------------------------------------------------------------------------
# homa_example_solvers: static library wrapping solver backends used by examples
# -----------------------------------------------------------------------------
set(_example_solver_sources
    LinSysSolvers/LinSysSolver.cpp
)

if(HOMA_EXAMPLE_WITH_CHOLMOD)
    list(APPEND _example_solver_sources LinSysSolvers/CHOLMODSolver.cpp)
endif()

if(HOMA_EXAMPLE_WITH_CUDSS)
    list(APPEND _example_solver_sources LinSysSolvers/CUDSSSolver.cu)
endif()

if(HOMA_EXAMPLE_WITH_MKL)
    list(APPEND _example_solver_sources LinSysSolvers/MKLSolver.cpp)
endif()

add_library(homa_example_solvers STATIC ${_example_solver_sources})

target_include_directories(homa_example_solvers PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/LinSysSolvers
    ${CMAKE_SOURCE_DIR}/src/utils
)

target_link_libraries(homa_example_solvers PUBLIC    
    homa_deps
)

set_property(TARGET homa_example_solvers PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET homa_example_solvers PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

# -----------------------------------------------------------------------------
# Helper to configure CUDA + link for each example executable
# -----------------------------------------------------------------------------
function(configure_example_target target_name)
    set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${target_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(${target_name} PRIVATE homa_example_solvers homa)
    set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF)
    target_compile_options(${target_name} PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-O0;-Xptxas=-O0;-lineinfo;-Xcompiler=-O0,-g3>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g3;-fno-inline>
    )
endfunction()

# -----------------------------------------------------------------------------
# Example executables
# -----------------------------------------------------------------------------
if(HOMA_EXAMPLE_WITH_CHOLMOD)
    add_executable(cholmod_example cholmod_example.cpp)
    configure_example_target(cholmod_example)
endif()

if(HOMA_EXAMPLE_WITH_MKL)
    add_executable(mkl_example mkl_example.cpp)
    configure_example_target(mkl_example)
endif()

if(HOMA_EXAMPLE_WITH_CUDSS)
    add_executable(cudss_example cudss_example.cpp)
    configure_example_target(cudss_example)
endif()
