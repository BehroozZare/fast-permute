cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(Homa
        VERSION 0.1.0
        LANGUAGES C CXX)

################################################################################
# HOMA OPTIONS
################################################################################
option(HOMA_BUILD_BENCHMARKS   "Build benchmark executables"      OFF)
option(HOMA_BUILD_TESTS        "Build unit tests"                 OFF)
option(HOMA_BUILD_PYTHON       "Build Python bindings"            OFF)
option(HOMA_WITH_CUDA          "Enable CUDA / GPU backend"        OFF)
option(HOMA_WITH_RXMESH        "Enable RxMesh patcher"            OFF)
option(HOMA_WITH_FAST_PATCHER  "Enable fast-patcher backend"      OFF)
option(HOMA_WITH_SUITESPARSE   "SuiteSparse for benchmarks"       OFF)
option(HOMA_WITH_MKL           "Intel MKL PARDISO for benchmarks" OFF)
option(HOMA_WITH_STRUMPACK     "STRUMPACK for benchmarks"         OFF)
option(HOMA_WITH_PARTH         "Parth ordering for benchmarks"    OFF)

# CUDA is enabled either explicitly or when legacy options require it
if(HOMA_WITH_CUDA OR HOMA_WITH_RXMESH)
    enable_language(CUDA)
endif()

# Legacy options used by the old build system (kept for transition period)
option(${PROJECT_NAME}_WITH_PARTH      "Use Parth for fill-reducing orderings" OFF)
option(${PROJECT_NAME}_WITH_SUITESPARSE "Use SuiteSparse for benchmark"        OFF)
option(${PROJECT_NAME}_WITH_MKL        "Use Intel MKL PARDISO for benchmark"   OFF)
option(${PROJECT_NAME}_WITH_STRUMPACK  "Use Strumpack for benchmark"           OFF)
option(${PROJECT_NAME}_WITH_PROFILE    "Use Profile for benchmark"             OFF)

# Sync legacy options from HOMA_ prefix
if(HOMA_WITH_PARTH)
    set(${PROJECT_NAME}_WITH_PARTH ON CACHE BOOL "" FORCE)
endif()
if(HOMA_WITH_SUITESPARSE)
    set(${PROJECT_NAME}_WITH_SUITESPARSE ON CACHE BOOL "" FORCE)
endif()
if(HOMA_WITH_MKL)
    set(${PROJECT_NAME}_WITH_MKL ON CACHE BOOL "" FORCE)
endif()
if(HOMA_WITH_STRUMPACK)
    set(${PROJECT_NAME}_WITH_STRUMPACK ON CACHE BOOL "" FORCE)
endif()

# Enable GLM experimental features globally
add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)

# Language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(HOMA_WITH_CUDA OR HOMA_WITH_RXMESH OR HOMA_BUILD_BENCHMARKS)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
    set(CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    set(CUDA_SEPARABLE_COMPILATION ON)
    include(${CMAKE_SOURCE_DIR}/cmake/AutoDetectCudaArch.cmake)
endif()

################################################################################
# SECTION 1: Include Required Dependencies
################################################################################

# ==============================================================================
# Compiler Configuration (Migrated from original RXMesh config)
# ==============================================================================
# C++ compiler flags
set(cxx_flags
        $<$<CXX_COMPILER_ID:MSVC>:-D_SCL_SECURE_NO_WARNINGS /openmp:experimental /MP /std:c++17 /bigobj>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Release>>:-Wall -m64 -fopenmp -O3 -Wno-unused-function>
        $<$<AND:$<CXX_COMPILER_ID:GNU>,$<CONFIG:Debug>>:-Wall -m64 -fopenmp -O0 -g3 -Wno-unused-function>
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

if(HOMA_WITH_CUDA OR HOMA_WITH_RXMESH OR HOMA_BUILD_BENCHMARKS)
    # CUDA Toolkit (cuSPARSE, cuSOLVER)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(gpu_ordering_flags INTERFACE CUDA::cusparse CUDA::cusolver)

    # cuDSS - Auto-detect correct version based on CUDA toolkit
    string(REGEX MATCH "^[0-9]+" CUDA_VERSION_MAJOR "${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "Detected CUDA toolkit version: ${CMAKE_CUDA_COMPILER_VERSION} (major: ${CUDA_VERSION_MAJOR})")

    if(NOT WIN32 AND NOT cudss_DIR)
        set(_cudss_search_paths
                "/usr/lib/x86_64-linux-gnu/libcudss/${CUDA_VERSION_MAJOR}/cmake/cudss"
                "/usr/lib/aarch64-linux-gnu/libcudss/${CUDA_VERSION_MAJOR}/cmake/cudss"
                "/usr/local/cuda-${CUDA_VERSION_MAJOR}/lib64/cmake/cudss"
        )
        foreach(_path ${_cudss_search_paths})
            if(EXISTS "${_path}/cudss-config.cmake")
                set(cudss_DIR "${_path}" CACHE PATH "Path to cuDSS CMake config" FORCE)
                message(STATUS "Auto-detected cuDSS for CUDA ${CUDA_VERSION_MAJOR} at: ${cudss_DIR}")
                break()
            endif()
        endforeach()
        unset(_cudss_search_paths)
    endif()

    find_package(cudss REQUIRED)
    message(STATUS "Found cuDSS version ${cudss_VERSION} (Library: ${cudss_LIBRARY_DIR})")
    target_link_directories(gpu_ordering_flags INTERFACE ${cudss_LIBRARY_DIR})
    target_link_libraries(gpu_ordering_flags INTERFACE cudss)
    target_compile_definitions(gpu_ordering_flags INTERFACE USE_CUDSS)
endif()

# Disable Cereal (and other deps) warnings as errors and unnecessary targets
set(WITH_WERROR OFF CACHE BOOL "Compile with '-Werror' C++ compiler flag" FORCE)
set(BUILD_SANDBOX OFF CACHE BOOL "Build sandbox examples" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "Build tests" FORCE)

# Core deps — always needed for the homa library
include(${CMAKE_SOURCE_DIR}/cmake/recipes/eigen.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/recipes/metis.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/recipes/spdlog.cmake)
# SuiteSparse/AMD is always needed: cpu_ordering_with_patch uses amd.h for local ordering
include(${CMAKE_SOURCE_DIR}/cmake/recipes/suitesparse.cmake)

# Benchmark / GPU deps — only when explicitly requested
if(HOMA_WITH_RXMESH OR HOMA_WITH_CUDA OR HOMA_BUILD_BENCHMARKS)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/rxmesh.cmake)
endif()

if(HOMA_BUILD_BENCHMARKS)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/cli11.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/libigl.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/recipes/polyscope.cmake)
endif()

# Create a dependency interface target to hold configuration
add_library(gpu_ordering_deps INTERFACE)
target_link_libraries(gpu_ordering_deps INTERFACE gpu_ordering_flags)

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
    if(NOT BLAS_LIBRARIES)
        message(FATAL_ERROR "BLAS_LIBRARIES not set. Pass -DBLAS_LIBRARIES=...")
    endif()
    
    target_link_libraries(gpu_ordering_deps INTERFACE 
        ${SUITESPARSE_LIBRARIES}
        ${BLAS_LIBRARIES}
        ${LAPACK_LIBRARIES}
    )
	if(OPENBLAS_INCLUDE_DIR)
		target_include_directories(gpu_ordering_deps INTERFACE ${OPENBLAS_INCLUDE_DIR})
    endif()
    target_include_directories(gpu_ordering_deps INTERFACE ${SUITESPARSE_INCLUDE_DIRS})
    target_compile_definitions(gpu_ordering_deps INTERFACE USE_SUITESPARSE)
    message(STATUS "CHOLMOD configured for CPU-only support")
endif ()

# Intel MKL PARDISO
if(${PROJECT_NAME}_WITH_MKL)
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/find")
    find_package(MKL REQUIRED)
    target_link_libraries(gpu_ordering_deps INTERFACE ${MKL_LIBRARIES})
    target_include_directories(gpu_ordering_deps INTERFACE ${MKL_INCLUDE_DIRS})
    target_compile_definitions(gpu_ordering_deps INTERFACE USE_MKL)
    message(STATUS "Intel MKL PARDISO configured")
endif()

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


# Print Eigen info for debugging
if(TARGET Eigen3::Eigen)
    get_target_property(EIGEN_INCLUDES Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "=== EIGEN DEBUG ===")
    message(STATUS "Eigen3::Eigen include dirs: ${EIGEN_INCLUDES}")
    message(STATUS "===================")
endif()

################################################################################
# homa: clean, CPU-only public library (always built)
################################################################################
add_library(homa STATIC
    src/ordering/homa.cpp
    src/ordering/cpu_ordering_with_patch.cpp
    src/ordering/neutral_ordering.cpp
    src/patching/metis_patcher.cpp
    src/utils/remove_diagonal.cpp
    src/utils/check_valid_permutation.cpp
    src/utils/compute_inverse_perm.cpp
    src/utils/min_vertex_cover_bipartite.cpp
)
if(${PROJECT_NAME}_WITH_PARTH)
    target_sources(homa PRIVATE src/ordering/parth_ordering.cpp)
endif()

target_include_directories(homa
    PUBLIC  include
    PRIVATE src
)
target_link_libraries(homa PUBLIC
    Eigen3::Eigen
    spdlog::spdlog
    metis
    SuiteSparse::AMD
)
target_include_directories(homa PRIVATE ${AMD_INCLUDE_DIR} ${SUITESPARSE_CONFIG_INCLUDE_DIR})
target_compile_definitions(homa PUBLIC
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)
add_library(Homa::homa ALIAS homa)

if(HOMA_BUILD_BENCHMARKS)
# Create a header-only INTERFACE library for libigl-dependent code
# Header-only ensures the code is compiled in the caller's translation unit,
# avoiding ABI issues with Eigen across library boundaries
add_library(libigl_utils INTERFACE)
target_link_libraries(libigl_utils INTERFACE
        igl::core
        Eigen3::Eigen
)
target_include_directories(libigl_utils INTERFACE src/utils)

# Create rxmesh_dev_helpers library (benchmark-only: includes GPU/CUDA sources)
add_library(rxmesh_dev_helpers
        src/utils/remove_diagonal.cpp
        src/utils/check_valid_permutation.cpp
        src/utils/compute_inverse_perm.cpp
        src/utils/min_vertex_cover_bipartite.cpp
        src/ordering/cpu_ordering_with_patch.cpp
        src/ordering/gpu_ordering_with_patch.cu
        src/ordering/ordering_factory.cu
        src/ordering/neutral_ordering.cpp
        src/ordering/patch_ordering.cu
        src/ordering/parth_ordering.cpp
        src/patching/metis_patcher.cpp
        src/patching/rxmesh_patcher.cu
)

# Remesher library for mesh remeshing operations
add_library(remesher
    src/remesher/collapse_edges.cpp
    src/remesher/equalize_valences.cpp
    src/remesher/remesh_botsch.cpp
    src/remesher/split_edges.cpp
    src/remesher/split_edges_until_bound.cpp
    src/remesher/tangential_relaxation.cpp
)
target_link_libraries(remesher PUBLIC igl::core Eigen3::Eigen)
target_include_directories(remesher PUBLIC src/remesher)
# Force include Eigen compatibility header that:
# 1. Includes cassert (fixes libigl's missing include)
# 2. Brings Eigen::placeholders::all into Eigen namespace (fixes Eigen/libigl version mismatch)
target_compile_options(remesher PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-include$<SEMICOLON>${CMAKE_SOURCE_DIR}/src/remesher/eigen_compat.h>
)

# Address Sanitizer (ASAN) for remesher only - helps detect memory errors in remeshing code
#option(ENABLE_REMESHER_ASAN "Enable Address Sanitizer for remesher library" OFF)
#if(ENABLE_REMESHER_ASAN)
#    message(STATUS "Address Sanitizer (ASAN) ENABLED for remesher library")
#    target_compile_options(remesher PRIVATE -fsanitize=address -fno-omit-frame-pointer -g)
#    target_link_options(remesher PUBLIC -fsanitize=address)
#endif()

# Link additional dependencies required by dev helpers
target_link_libraries(rxmesh_dev_helpers PUBLIC
        libigl_utils
        igl::core
        spdlog::spdlog
        Eigen3::Eigen
        CLI11::CLI11
)
target_link_libraries(rxmesh_dev_helpers PUBLIC gpu_ordering_deps PRIVATE RXMesh)


# Add dev code include directories
target_include_directories(rxmesh_dev_helpers PUBLIC
        src/ordering
        src/patching
        src/utils
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
    add_subdirectory(tools)

endif() # HOMA_BUILD_BENCHMARKS