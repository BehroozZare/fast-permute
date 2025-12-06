cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(gpu_ordering
        VERSION 0.1.0
        LANGUAGES C CXX CUDA)

# Enable GLM experimental features globally
add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)

# Language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Auto-detect CUDA Architecture
include(${CMAKE_SOURCE_DIR}/cmake/AutoDetectCudaArch.cmake)

################################################################################
# SECTION 1: Include Required Dependencies
################################################################################

# ==============================================================================
# Compiler Configuration (Migrated from original RXMesh config)
# ==============================================================================
# C++ compiler flags
set(cxx_flags
        $<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp:experimental /MP /std:c++17 /bigobj>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:-Wall -m64 -fopenmp -O3 -std=c++20 -Wno-unused-function>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:-Wall -m64 -fopenmp -O0 -g3 -std=c++20 -Wno-unused-function>
)

# CUDA compiler flags
set(MSVC_XCOMPILER_FLAGS "/openmp:experimental /MP /std:c++17 /Zi")

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

# Apply flags to global targets (optional, or attached to specific targets later)
# For this project, we'll attach them to our dependency interface so everyone gets them
add_library(gpu_ordering_flags INTERFACE)
target_compile_options(gpu_ordering_flags INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>
        $<$<COMPILE_LANGUAGE:CUDA>:${cuda_flags}>
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g3")
endif()


# OpenMP
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    target_link_libraries(gpu_ordering_flags INTERFACE OpenMP::OpenMP_CXX)
endif()

# CUDA Toolkit (cuSPARSE, cuSOLVER)
find_package(CUDAToolkit REQUIRED)
target_link_libraries(gpu_ordering_flags INTERFACE CUDA::cusparse CUDA::cusolver)



# Disable Cereal (and other deps) warnings as errors and unnecessary targets
set(WITH_WERROR OFF CACHE BOOL "Compile with '-Werror' C++ compiler flag" FORCE)
set(BUILD_SANDBOX OFF CACHE BOOL "Build sandbox examples" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)

include(${CMAKE_SOURCE_DIR}/cmake/recipes/rxmesh.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/recipes/cli11.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/recipes/eigen.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/recipes/libigl.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/recipes/metis.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/recipes/suitesparse.cmake)


# Create a dependency interface target to hold configuration
add_library(gpu_ordering_deps INTERFACE)
target_link_libraries(gpu_ordering_deps INTERFACE gpu_ordering_flags)

################################################################################
# SECTION 2: Development Build Options
################################################################################
option(${PROJECT_NAME}_WITH_PARTH "Use Parth for fill-reducing orderings" ON)
option(${PROJECT_NAME}_WITH_SUITESPARSE "Use SuiteSparse for benchmark" ON)
option(${PROJECT_NAME}_WITH_STRUMPACK "Use Strumpack for benchmark" OFF)
option(${PROJECT_NAME}_WITH_PROFILE "Use Profile for benchmark" ON)

################################################################################
# SECTION 3: Configure Dependencies (formerly attached to main lib)
################################################################################

# Find MPI (required for parallel features, especially STRUMPACK)
if (${PROJECT_NAME}_WITH_STRUMPACK)
    find_package(MPI REQUIRED COMPONENTS CXX)
    if(NOT MPI_FOUND)
        message(FATAL_ERROR "MPI not found - STRUMPACK will not be available")
    endif()
    if(MPI_FOUND)
        message(STATUS "Found MPI:")
        message(STATUS "  MPI C++ Compiler: ${MPI_CXX_COMPILER}")
        message(STATUS "  MPI Include Path: ${MPI_CXX_INCLUDE_DIRS}")
        message(STATUS "  MPI Libraries: ${MPI_CXX_LIBRARIES}")
    endif()
else()
    message(STATUS "MPI not found - STRUMPACK will not be available")
endif()

# SuiteSparse - Must be configured before Parth since Parth depends on it
# Note: suitesparse.cmake is included at line 96 which also includes openblas.cmake
# for cross-platform BLAS/LAPACK detection
if (${PROJECT_NAME}_WITH_SUITESPARSE)
    # Add SuiteSparse to the dependencies target
    # The suitesparse.cmake recipe sets SUITESPARSE_LIBRARIES and SUITESPARSE_INCLUDE_DIRS
    target_link_libraries(gpu_ordering_deps INTERFACE ${SUITESPARSE_LIBRARIES})
    target_include_directories(gpu_ordering_deps INTERFACE ${SUITESPARSE_INCLUDE_DIRS})
    target_compile_definitions(gpu_ordering_deps INTERFACE USE_SUITESPARSE)
    message(STATUS "CHOLMOD configured for CPU-only support")
endif ()

# Parth for fill-reducing orderings
if(${PROJECT_NAME}_WITH_PARTH)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/parth.cmake)
    # Add Parth to the dependencies target
    target_link_libraries(gpu_ordering_deps INTERFACE Parth::parth)
    target_compile_definitions(gpu_ordering_deps INTERFACE USE_PARTH)
endif()

if(${PROJECT_NAME}_WITH_PROFILE)
    target_compile_definitions(gpu_ordering_deps INTERFACE USE_PROFILE)
endif()


# STRUMPACK - Sparse direct solver with CUDA support
if(${PROJECT_NAME}_WITH_STRUMPACK)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/strumpack.cmake)
    target_link_libraries(gpu_ordering_deps INTERFACE STRUMPACK::strumpack_full)
    target_compile_definitions(gpu_ordering_deps INTERFACE USE_STRUMPACK)
endif()


# Create rxmesh_dev_helpers library
add_library(rxmesh_dev_helpers
        src/LinSysSolvers/LinSysSolver.cpp
        src/LinSysSolvers/CUDSSSolver.cu
        src/LinSysSolvers/CHOLMODSolver.cpp
        src/LinSysSolvers/STRUMPACKSolver.cpp
        src/utils/get_factor_nnz.cpp
        src/utils/remove_diagonal.cpp
        src/utils/check_valid_permutation.cpp
        src/utils/compute_inverse_perm.cpp
        src/utils/min_vertex_cover_bipartite.cpp
        src/PatchOrdering/cpu_ordering_with_patch.cpp
        src/PatchOrdering/gpu_ordering_with_patch.cu
        src/Ordering/ordering.cu
        src/Ordering/metis_ordering.cpp
        src/Ordering/rxmesh_ordering.cu
        src/Ordering/neutral_ordering.cpp
        src/Ordering/patch_ordering.cu
        src/Ordering/parth_ordering.cpp
)

# Link additional dependencies required by dev helpers
target_link_libraries(rxmesh_dev_helpers PUBLIC
        igl::core
        spdlog::spdlog
        Eigen3::Eigen
        CLI11::CLI11
)
target_link_libraries(rxmesh_dev_helpers PUBLIC gpu_ordering_deps PRIVATE RXMesh)


# Add dev code include directories
target_include_directories(rxmesh_dev_helpers PUBLIC
        src/LinSysSolvers
        src/Ordering
        src/utils
        src/PatchOrdering
)

# Configure CUDA compilation for the dev helper library
# Enable separable compilation but DON'T resolve device symbols yet
# Device symbols will be resolved in the final executable
set_property(TARGET rxmesh_dev_helpers PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET rxmesh_dev_helpers PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

# Inherit CUDA architecture from the global settings (from AutoDetectCudaArch)
# No need to manual copy unless necessary

# Add debug definitions based on build type
# In Debug builds: DEBUG is defined, NDEBUG is NOT defined
# In Release builds: NDEBUG is defined, DEBUG is NOT defined
target_compile_definitions(rxmesh_dev_helpers PUBLIC
        $<$<CONFIG:Debug>:DEBUG>
        $<$<CONFIG:Release>:NDEBUG>
)

# CUDA: real debug: disable device optimizations and keep line info
target_compile_options(rxmesh_dev_helpers PUBLIC
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-O0;-Xptxas=-O0;-lineinfo;-Xcompiler=-O0,-g>
)

# Host C++ files in the same target
target_compile_options(rxmesh_dev_helpers PUBLIC
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g;-fno-inline>
)

################################################################################
# SECTION 5: Create Development Targets (Executables)
################################################################################
# Development targets use both the RXMesh library and the dev helpers library

add_subdirectory(benchmark)