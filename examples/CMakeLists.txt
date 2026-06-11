if((DEFINED HOMA_WITH_MKL AND HOMA_WITH_MKL) OR
   (DEFINED ${PROJECT_NAME}_WITH_MKL AND ${PROJECT_NAME}_WITH_MKL))
    set(HOMA_EXAMPLE_WITH_MKL ON CACHE BOOL "Build MKL example" FORCE)
endif()

if(HOMA_EXAMPLE_WITH_MKL)
    set(HOMA_MKL_ALLOW_DOWNLOAD ON CACHE BOOL "Download Intel MKL wheels when an installed MKL is not found")
    set(HOMA_MKL_VERSION "2025.3.0" CACHE STRING "Intel MKL wheel fallback version")
    set(HOMA_MKL_INTERFACE "lp64" CACHE STRING "Intel MKL interface layer")
    set(HOMA_MKL_THREADING "intel" CACHE STRING "Intel MKL threading layer")
    set(HOMA_MKL_LINKING "sdl" CACHE STRING "Intel MKL linking mode")
    set_property(CACHE HOMA_MKL_INTERFACE PROPERTY STRINGS lp64)
    set_property(CACHE HOMA_MKL_THREADING PROPERTY STRINGS intel sequential)
    set_property(CACHE HOMA_MKL_LINKING PROPERTY STRINGS sdl)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/mkl.cmake)
endif()

if(HOMA_EXAMPLE_WITH_CHOLMOD)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/cholmod.cmake)
endif()

if(HOMA_EXAMPLE_WITH_CUDSS)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/cudss.cmake)
endif()

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
    Eigen3::Eigen
    spdlog::spdlog
)

if(HOMA_EXAMPLE_WITH_MKL)
    target_link_libraries(homa_example_solvers PUBLIC Homa::MKL)
    target_compile_definitions(homa_example_solvers PUBLIC USE_MKL)
endif()

if(HOMA_EXAMPLE_WITH_CHOLMOD)
    target_link_libraries(homa_example_solvers PUBLIC SuiteSparse::CHOLMOD)
    target_compile_definitions(homa_example_solvers PUBLIC USE_CHOLMOD)
endif()

if(HOMA_EXAMPLE_WITH_CUDSS)
    target_link_libraries(homa_example_solvers PUBLIC cudss)
    target_compile_definitions(homa_example_solvers PUBLIC USE_CUDSS)
endif()

set_property(TARGET homa_example_solvers PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET homa_example_solvers PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ${HOMA_EXAMPLE_WITH_CUDSS})

# -----------------------------------------------------------------------------
# Helper to configure CUDA + link for each example executable
# -----------------------------------------------------------------------------
function(configure_example_target target_name)
    set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${target_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    target_link_libraries(${target_name} PRIVATE homa_example_solvers homa CLI11::CLI11)
    if(TARGET libigl_utils)
        target_link_libraries(${target_name} PRIVATE libigl_utils)
    endif()
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
    homa_configure_mkl_runtime(mkl_example)
endif()

if(HOMA_EXAMPLE_WITH_CUDSS)
    add_executable(cudss_example cudss_example.cpp)
    configure_example_target(cudss_example)
    homa_configure_cudss_runtime(cudss_example)
endif()

add_executable(matrix_example matrix_example.cpp)
configure_example_target(matrix_example)
if(COMMAND homa_configure_cudss_runtime)
    homa_configure_cudss_runtime(matrix_example)
endif()
