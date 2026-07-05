cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(Homa
        VERSION 0.1.0
        LANGUAGES C CXX)

set(HOMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(HOMA_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_CONFIGURATION_TYPES AND "${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()

################################################################################
# HOMA OPTIONS
################################################################################
option(HOMA_BUILD_EXAMPLE      "Build example executables"          OFF)
option(HOMA_BUILD_TESTS        "Build unit tests"                   OFF)
option(HOMA_BUILD_PYTHON       "Build Python bindings"              OFF)
option(HOMA_WITH_CUDA          "Enable CUDA / GPU backend"          ON)
option(HOMA_WITH_CHOLMOD       "Enable CHOLMOD linear solver"       ON)
option(HOMA_WITH_MKL           "Enable MKL linear solver"           ON)
option(HOMA_WITH_CUDSS         "Enable cuDSS linear solver"         ON)
option(HOMA_WITH_LLOYD_PATCHER "Enable Lloyd-clustering patcher"    ON)
option(HOMA_WITH_PROFILE       "Use Profile for benchmark"          OFF)

set(_homa_needs_cuda OFF)
if(HOMA_WITH_CUDA OR HOMA_WITH_CUDSS)
    set(_homa_needs_cuda ON)
endif()

message(STATUS "HOMA_BUILD_EXAMPLE: ${HOMA_BUILD_EXAMPLE}")
message(STATUS "HOMA_BUILD_TESTS: ${HOMA_BUILD_TESTS}")
message(STATUS "HOMA_BUILD_PYTHON: ${HOMA_BUILD_PYTHON}")
message(STATUS "HOMA_WITH_CUDA: ${HOMA_WITH_CUDA}")
message(STATUS "HOMA_WITH_CHOLMOD: ${HOMA_WITH_CHOLMOD}")
message(STATUS "HOMA_WITH_MKL: ${HOMA_WITH_MKL}")
message(STATUS "HOMA_WITH_CUDSS: ${HOMA_WITH_CUDSS}")
message(STATUS "HOMA_WITH_LLOYD_PATCHER: ${HOMA_WITH_LLOYD_PATCHER}")
message(STATUS "HOMA_WITH_PROFILE: ${HOMA_WITH_PROFILE}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()


set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(_homa_needs_cuda AND
   NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND
   "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()

if(_homa_needs_cuda)
    enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(_homa_needs_cuda)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set(CUDA_SEPARABLE_COMPILATION ON)
endif()

################################################################################
# SECTION 1: Include Required Dependencies
################################################################################

set(cxx_flags
        $<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp:experimental /MP /std:c++20 /bigobj>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:-Wall -m64 -fopenmp -O3 -Wno-unused-function>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:-Wall -m64 -fopenmp -O0 -g3 -Wno-unused-function>
)

set(MSVC_XCOMPILER_FLAGS "/openmp:experimental /MP /std:c++20 /Zi")

set(cuda_flags
        #Windows stuff
        -Xcompiler=$<$<CXX_COMPILER_ID:MSVC>:${MSVC_XCOMPILER_FLAGS}>
        # Release
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:
        -Xcompiler=-rdynamic,-Wall,-fopenmp,-O3,-Wno-unused-function
        -O3
        >
        # Debug
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:
        -G
        -O0
        -Xptxas=-O0
        -lineinfo
        -Xcompiler=-rdynamic,-Wall,-fopenmp,-O0,-g3,-Wno-unused-function
        >
        # other common CUDA flags here
        -rdc=true
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -Xptxas -warn-spills -res-usage
        --ptxas-options=-v
)

add_library(homa_flags INTERFACE)
target_compile_options(homa_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
        $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)

# OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(homa_flags INTERFACE OpenMP::OpenMP_CXX)
endif()

if(_homa_needs_cuda)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(homa_flags INTERFACE CUDA::cudart CUDA::cusparse CUDA::cusolver)
endif()

# Core deps — always needed for the homa library
include(${HOMA_SOURCE_DIR}/cmake/recipes/eigen.cmake)
include(${HOMA_SOURCE_DIR}/cmake/recipes/metis.cmake)
include(${HOMA_SOURCE_DIR}/cmake/recipes/spdlog.cmake)
# SuiteSparse/AMD is always needed: cpu_ordering_with_patch uses amd.h for local ordering
include(${HOMA_SOURCE_DIR}/cmake/recipes/suitesparse.cmake)

if(HOMA_WITH_LLOYD_PATCHER)
    include(${HOMA_SOURCE_DIR}/cmake/recipes/lloyd_clustering.cmake)
endif()

if(HOMA_WITH_MKL)
    set(HOMA_MKL_ALLOW_DOWNLOAD ON CACHE BOOL "Download Intel MKL wheels when an installed MKL is not found")
    set(HOMA_MKL_VERSION "2025.3.0" CACHE STRING "Intel MKL wheel fallback version")
    set(HOMA_MKL_INTERFACE "lp64" CACHE STRING "Intel MKL interface layer")
    set(HOMA_MKL_THREADING "intel" CACHE STRING "Intel MKL threading layer")
    set(HOMA_MKL_LINKING "sdl" CACHE STRING "Intel MKL linking mode")
    set_property(CACHE HOMA_MKL_INTERFACE PROPERTY STRINGS lp64)
    set_property(CACHE HOMA_MKL_THREADING PROPERTY STRINGS intel sequential)
    set_property(CACHE HOMA_MKL_LINKING PROPERTY STRINGS sdl)
    include(${HOMA_SOURCE_DIR}/cmake/recipes/mkl.cmake)
endif()

if(HOMA_WITH_CHOLMOD)
    include(${HOMA_SOURCE_DIR}/cmake/recipes/cholmod.cmake)
endif()

if(HOMA_WITH_CUDSS)
    set(HOMA_CUDSS_ALLOW_DOWNLOAD ON CACHE BOOL "Download NVIDIA cuDSS wheels when an installed cuDSS is not found")
    set(HOMA_CUDSS_VERSION "0.7.1.6" CACHE STRING "NVIDIA cuDSS wheel fallback version")
    include(${HOMA_SOURCE_DIR}/cmake/recipes/cudss.cmake)
endif()

if(HOMA_BUILD_EXAMPLE)
    include(${HOMA_SOURCE_DIR}/cmake/recipes/cli11.cmake)
    include(${HOMA_SOURCE_DIR}/cmake/recipes/libigl.cmake)
endif()

function(homa_configure_runtime target_name)
    if(NOT TARGET ${target_name})
        message(FATAL_ERROR "homa_configure_runtime: target '${target_name}' does not exist")
    endif()

    get_property(_homa_runtime_configured TARGET ${target_name} PROPERTY homa_runtime_configured)
    if(_homa_runtime_configured)
        return()
    endif()

    if(COMMAND homa_configure_mkl_runtime)
        homa_configure_mkl_runtime(${target_name})
    endif()
    if(COMMAND homa_configure_cudss_runtime)
        homa_configure_cudss_runtime(${target_name})
    endif()

    set_property(TARGET ${target_name} PROPERTY homa_runtime_configured TRUE)
endfunction()

# Create a dependency interface target to hold configuration
add_library(homa_deps INTERFACE)

################################################################################
# SECTION 3: Configure Dependencies
################################################################################

if(HOMA_WITH_PROFILE)
    target_compile_definitions(homa_deps INTERFACE USE_PROFILE)
endif()

if(TARGET Eigen3::Eigen)
    get_target_property(EIGEN_INCLUDES Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "=== EIGEN DEBUG ===")
    message(STATUS "Eigen3::Eigen include dirs: ${EIGEN_INCLUDES}")
    message(STATUS "===================")
endif()

set(_homa_sources
    src/ordering/homa.cpp    
    src/ordering/cpu_ordering_with_patch.cpp
    src/ordering/amd_order_helper.cpp
    src/patching/greedy_patcher.cpp
    src/patching/metis_patcher.cpp
    src/utils/remove_diagonal.cpp
    src/utils/check_valid_permutation.cpp
	src/utils/compress_hessian.cpp
    src/solvers/LinSysSolver.cpp	
    include/homa/solvers/LinSysSolver.h
	include/homa/solvers/CHOLMODSolver.h
	include/homa/solvers/CUDSSSolver.h
	include/homa/solvers/MKLSolver.h
	include/homa/solvers/scalar_traits.h
	include/homa/matrix_view.h
	include/homa/homa.h
	include/homa/patcher.h
	include/homa/types.h
	include/homa/utils/check_valid_permutation.h
	include/homa/utils/compress_hessian.h
	include/homa/utils/nvtx_helper.h
	include/homa/utils/remove_diagonal.h
	include/homa/utils/SPD_cot_matrix.h
)
if(HOMA_WITH_LLOYD_PATCHER)
    list(APPEND _homa_sources src/patching/lloyd_patcher.cpp)
endif()
if(HOMA_WITH_CHOLMOD)
    list(APPEND _homa_sources 
        src/solvers/CHOLMODSolver.cpp
        include/homa/solvers/CHOLMODSolver.h)
endif()
if(HOMA_WITH_MKL)
    list(APPEND _homa_sources
        src/solvers/MKLSolver.cpp
        include/homa/solvers/MKLSolver.h)
endif()
if(HOMA_WITH_CUDSS)
    list(APPEND _homa_sources 
        src/solvers/CUDSSSolver.cu
        include/homa/solvers/CUDSSSolver.h
        include/homa/utils/cuda_error_handler.h)
endif()

add_library(homa STATIC ${_homa_sources})

target_include_directories(homa
    PUBLIC  include
    PRIVATE src
)

set(_homa_public_libs
    homa_deps
    Eigen3::Eigen
    spdlog::spdlog
)

if(HOMA_WITH_MKL)
    list(APPEND _homa_public_libs Homa::MKL)
endif()

list(APPEND _homa_public_libs metis)

target_link_libraries(homa
    PUBLIC ${_homa_public_libs}
    PRIVATE
        homa_flags
        SuiteSparse::AMD
)
target_include_directories(homa PRIVATE ${AMD_INCLUDE_DIR} ${SUITESPARSE_CONFIG_INCLUDE_DIR})
target_compile_definitions(homa PUBLIC
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

if(_homa_needs_cuda)
    target_link_libraries(homa PUBLIC CUDA::cudart)
endif()

if(HOMA_WITH_LLOYD_PATCHER)
    target_link_libraries(homa PRIVATE mesh_clustering_lib)
    target_compile_definitions(homa PRIVATE HOMA_WITH_LLOYD_PATCHER)
endif()

if(HOMA_WITH_MKL)
    target_compile_definitions(homa PUBLIC USE_MKL)
endif()

if(HOMA_WITH_CHOLMOD)
    target_link_libraries(homa PUBLIC SuiteSparse::CHOLMOD)
    target_compile_definitions(homa PUBLIC USE_CHOLMOD)
endif()

if(HOMA_WITH_CUDSS)
    target_link_libraries(homa PUBLIC cudss)
    target_compile_definitions(homa PUBLIC USE_CUDSS)
    set_property(TARGET homa PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET homa PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

add_library(Homa::homa ALIAS homa)

if(HOMA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(HOMA_BUILD_EXAMPLE)
    add_library(libigl_utils INTERFACE)
    target_link_libraries(libigl_utils INTERFACE igl::core Eigen3::Eigen)
    target_include_directories(libigl_utils INTERFACE src/utils)
    add_subdirectory(examples)

endif()

if(HOMA_BUILD_PYTHON)
    add_subdirectory(python)
endif()
