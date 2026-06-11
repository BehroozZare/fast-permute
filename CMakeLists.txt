cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

project(Homa
        VERSION 0.1.0
        LANGUAGES C CXX)

################################################################################
# HOMA OPTIONS
################################################################################
option(HOMA_BUILD_BENCHMARKS   "Build benchmark executables"      OFF)
option(HOMA_BUILD_TESTS        "Build unit tests"                 OFF)
option(HOMA_BUILD_PYTHON       "Build Python bindings"            OFF)
option(HOMA_WITH_CUDA          "Enable CUDA / GPU backend"        ON)
option(HOMA_WITH_LLOYD_PATCHER "Enable Lloyd-clustering patcher"  ON)
option(HOMA_WITH_PROFILE       "Use Profile for benchmark"        OFF)
option(HOMA_EXAMPLE_WITH_CHOLMOD "Build CHOLMOD example"          ON)
option(HOMA_EXAMPLE_WITH_MKL     "Build MKL example"              ON)
option(HOMA_EXAMPLE_WITH_CUDSS   "Build cuDSS example"            ON)

message(STATUS "HOMA_BUILD_BENCHMARKS: ${HOMA_BUILD_BENCHMARKS}")
message(STATUS "HOMA_BUILD_TESTS: ${HOMA_BUILD_TESTS}")
message(STATUS "HOMA_BUILD_PYTHON: ${HOMA_BUILD_PYTHON}")
message(STATUS "HOMA_WITH_CUDA: ${HOMA_WITH_CUDA}")
message(STATUS "HOMA_WITH_LLOYD_PATCHER: ${HOMA_WITH_LLOYD_PATCHER}")
message(STATUS "HOMA_WITH_PROFILE: ${HOMA_WITH_PROFILE}")
message(STATUS "HOMA_EXAMPLE_WITH_CHOLMOD: ${HOMA_EXAMPLE_WITH_CHOLMOD}")
message(STATUS "HOMA_EXAMPLE_WITH_MKL: ${HOMA_EXAMPLE_WITH_MKL}")
message(STATUS "HOMA_EXAMPLE_WITH_CUDSS: ${HOMA_EXAMPLE_WITH_CUDSS}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


if((HOMA_WITH_CUDA OR HOMA_BUILD_BENCHMARKS) AND
   NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND
   "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
endif()

if(HOMA_WITH_CUDA OR HOMA_BUILD_BENCHMARKS)
    enable_language(CUDA)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(HOMA_WITH_CUDA OR HOMA_BUILD_BENCHMARKS)
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

if(HOMA_WITH_CUDA OR HOMA_BUILD_BENCHMARKS)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(homa_flags INTERFACE CUDA::cusparse CUDA::cusolver)
endif()

# Core deps — always needed for the homa library
include(${CMAKE_SOURCE_DIR}/cmake/recipes/eigen.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/recipes/metis.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/recipes/spdlog.cmake)
# SuiteSparse/AMD is always needed: cpu_ordering_with_patch uses amd.h for local ordering
include(${CMAKE_SOURCE_DIR}/cmake/recipes/suitesparse.cmake)

if(HOMA_WITH_LLOYD_PATCHER)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/lloyd_clustering.cmake)
endif()

if(HOMA_BUILD_BENCHMARKS)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/cli11.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/libigl.cmake)
endif()

# Create a dependency interface target to hold configuration
add_library(homa_deps INTERFACE)
target_link_libraries(homa_deps INTERFACE homa_flags)

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
    src/patching/metis_patcher.cpp
    src/utils/remove_diagonal.cpp
    src/utils/check_valid_permutation.cpp
    src/utils/compute_inverse_perm.cpp
    src/utils/min_vertex_cover_bipartite.cpp
)
if(HOMA_WITH_LLOYD_PATCHER)
    list(APPEND _homa_sources src/patching/lloyd_patcher.cpp)
endif()

add_library(homa STATIC ${_homa_sources})

target_include_directories(homa
    PUBLIC  include
    PRIVATE src
)
target_link_libraries(homa
    PUBLIC
        Eigen3::Eigen
        spdlog::spdlog
        metis
    PRIVATE
        SuiteSparse::AMD
)
target_include_directories(homa PRIVATE ${AMD_INCLUDE_DIR} ${SUITESPARSE_CONFIG_INCLUDE_DIR})
target_compile_definitions(homa PUBLIC
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

if(HOMA_WITH_LLOYD_PATCHER)
    target_link_libraries(homa PRIVATE mesh_clustering_lib)
    target_compile_definitions(homa PRIVATE HOMA_WITH_LLOYD_PATCHER)
endif()

add_library(Homa::homa ALIAS homa)

if(HOMA_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(HOMA_BUILD_BENCHMARKS)
    add_library(libigl_utils INTERFACE)
    target_link_libraries(libigl_utils INTERFACE igl::core Eigen3::Eigen)
    target_include_directories(libigl_utils INTERFACE src/utils)
    add_subdirectory(examples)

endif()
