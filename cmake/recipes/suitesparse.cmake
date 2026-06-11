# SuiteSparse core recipe.
# The homa library only needs AMD for local ordering, so keep CHOLMOD/BLAS out
# of the default dependency graph. Optional examples can add CHOLMOD separately.

include(FetchContent)

FetchContent_Declare(
    suitesparse
    GIT_REPOSITORY https://github.com/DrTimothyAldenDavis/SuiteSparse.git
    GIT_TAG v7.11.0
    GIT_SHALLOW TRUE
)

FetchContent_GetProperties(suitesparse)
if(NOT suitesparse_POPULATED)
    message(STATUS "Fetching SuiteSparse...")
    if(POLICY CMP0169)
        cmake_policy(PUSH)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_Populate(suitesparse)
    if(POLICY CMP0169)
        cmake_policy(POP)
    endif()
endif()

set(HOMA_SUITESPARSE_SOURCE_DIR "${suitesparse_SOURCE_DIR}" CACHE INTERNAL "SuiteSparse source directory")
set(HOMA_SUITESPARSE_BINARY_DIR "${suitesparse_BINARY_DIR}" CACHE INTERNAL "SuiteSparse binary directory")

function(_homa_suitesparse_alias namespaced_target static_target shared_target)
    if(TARGET ${namespaced_target})
        return()
    endif()

    if(TARGET ${static_target})
        add_library(${namespaced_target} ALIAS ${static_target})
    elseif(TARGET ${shared_target})
        add_library(${namespaced_target} ALIAS ${shared_target})
    endif()
endfunction()

list(APPEND CMAKE_MODULE_PATH "${HOMA_SUITESPARSE_SOURCE_DIR}/SuiteSparse_config/cmake_modules")

set(SUITESPARSE_ROOT_CMAKELISTS ON)
set(SUITESPARSE_USE_CUDA OFF CACHE BOOL "Disable CUDA in SuiteSparse" FORCE)
set(SUITESPARSE_USE_OPENMP OFF CACHE BOOL "Disable OpenMP in SuiteSparse" FORCE)
set(SUITESPARSE_USE_FORTRAN OFF CACHE BOOL "Disable Fortran in SuiteSparse" FORCE)
set(SUITESPARSE_REQUIRE_BLAS OFF CACHE BOOL "AMD-only SuiteSparse does not require BLAS" FORCE)
set(SUITESPARSE_DEMOS OFF CACHE BOOL "Disable SuiteSparse demos" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "Build SuiteSparse static libraries" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Disable SuiteSparse shared libraries" FORCE)

if(NOT TARGET SuiteSparseConfig_static AND NOT TARGET SuiteSparseConfig)
    add_subdirectory(
        "${HOMA_SUITESPARSE_SOURCE_DIR}/SuiteSparse_config"
        "${HOMA_SUITESPARSE_BINARY_DIR}/SuiteSparse_config"
        EXCLUDE_FROM_ALL
    )
endif()

_homa_suitesparse_alias(SuiteSparse::SuiteSparseConfig SuiteSparseConfig_static SuiteSparseConfig)
_homa_suitesparse_alias(SuiteSparse::SuiteSparseConfig_static SuiteSparseConfig_static SuiteSparseConfig)

if(NOT TARGET AMD_static AND NOT TARGET AMD)
    add_subdirectory(
        "${HOMA_SUITESPARSE_SOURCE_DIR}/AMD"
        "${HOMA_SUITESPARSE_BINARY_DIR}/AMD"
        EXCLUDE_FROM_ALL
    )
endif()

_homa_suitesparse_alias(SuiteSparse::AMD AMD_static AMD)
_homa_suitesparse_alias(SuiteSparse::AMD_static AMD_static AMD)

set(AMD_INCLUDE_DIR "${HOMA_SUITESPARSE_SOURCE_DIR}/AMD/Include" CACHE PATH "AMD include directory" FORCE)
set(SUITESPARSE_CONFIG_INCLUDE_DIR "${HOMA_SUITESPARSE_SOURCE_DIR}/SuiteSparse_config" CACHE PATH "SuiteSparse_config include directory" FORCE)
set(SUITESPARSE_INCLUDE_DIRS
    "${AMD_INCLUDE_DIR}"
    "${SUITESPARSE_CONFIG_INCLUDE_DIR}"
    CACHE PATH "SuiteSparse include directories"
    FORCE
)

if(TARGET AMD_static)
    set(AMD_LIBRARY AMD_static CACHE STRING "AMD library" FORCE)
    set(SUITESPARSE_CONFIG_LIBRARY SuiteSparseConfig_static CACHE STRING "SuiteSparse_config library" FORCE)
    set(SUITESPARSE_LIBRARIES AMD_static SuiteSparseConfig_static CACHE STRING "SuiteSparse libraries" FORCE)
else()
    set(AMD_LIBRARY AMD CACHE STRING "AMD library" FORCE)
    set(SUITESPARSE_CONFIG_LIBRARY SuiteSparseConfig CACHE STRING "SuiteSparse_config library" FORCE)
    set(SUITESPARSE_LIBRARIES AMD SuiteSparseConfig CACHE STRING "SuiteSparse libraries" FORCE)
endif()

set(SuiteSparse_FOUND TRUE CACHE BOOL "SuiteSparse found" FORCE)
set(SUITESPARSE_FOUND TRUE CACHE BOOL "SuiteSparse found" FORCE)

message(STATUS "SuiteSparse configured with AMD only")
