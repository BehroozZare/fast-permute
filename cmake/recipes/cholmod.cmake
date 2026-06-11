# Optional CHOLMOD recipe for examples.
# CHOLMOD is intentionally not part of the core homa dependency graph.

include("${CMAKE_CURRENT_LIST_DIR}/suitesparse.cmake")

set(SUITESPARSE_USE_CUDA OFF CACHE BOOL "Disable CUDA in SuiteSparse" FORCE)
set(SUITESPARSE_USE_OPENMP OFF CACHE BOOL "Disable OpenMP in SuiteSparse" FORCE)
set(SUITESPARSE_USE_FORTRAN OFF CACHE BOOL "Disable Fortran in SuiteSparse" FORCE)
set(SUITESPARSE_REQUIRE_BLAS OFF CACHE BOOL "CHOLMOD example uses simplicial CPU CHOLMOD without BLAS" FORCE)

set(CHOLMOD_USE_CUDA OFF CACHE BOOL "Disable CUDA acceleration in CHOLMOD" FORCE)
set(CHOLMOD_GPL ON CACHE BOOL "Enable GPL CHOLMOD modules" FORCE)
set(CHOLMOD_CHOLESKY ON CACHE BOOL "Enable CHOLMOD Cholesky module" FORCE)
set(CHOLMOD_CAMD ON CACHE BOOL "Enable CHOLMOD CAMD/CCOLAMD support" FORCE)
set(CHOLMOD_PARTITION ON CACHE BOOL "Enable CHOLMOD METIS partition module" FORCE)
set(CHOLMOD_SUPERNODAL OFF CACHE BOOL "Disable CHOLMOD supernodal module to avoid BLAS dependency" FORCE)
set(CHOLMOD_MATRIXOPS OFF CACHE BOOL "Disable CHOLMOD MatrixOps module" FORCE)
set(CHOLMOD_MODIFY OFF CACHE BOOL "Disable CHOLMOD Modify module" FORCE)

foreach(_homa_suitesparse_package IN ITEMS COLAMD CAMD CCOLAMD)
    string(TOLOWER "${_homa_suitesparse_package}" _homa_suitesparse_package_lower)
    if(NOT TARGET ${_homa_suitesparse_package}_static AND NOT TARGET ${_homa_suitesparse_package})
        add_subdirectory(
            "${HOMA_SUITESPARSE_SOURCE_DIR}/${_homa_suitesparse_package}"
            "${HOMA_SUITESPARSE_BINARY_DIR}/${_homa_suitesparse_package}"
            EXCLUDE_FROM_ALL
        )
    endif()

    _homa_suitesparse_alias(
        SuiteSparse::${_homa_suitesparse_package}
        ${_homa_suitesparse_package}_static
        ${_homa_suitesparse_package}
    )
    _homa_suitesparse_alias(
        SuiteSparse::${_homa_suitesparse_package}_static
        ${_homa_suitesparse_package}_static
        ${_homa_suitesparse_package}
    )
endforeach()
unset(_homa_suitesparse_package)
unset(_homa_suitesparse_package_lower)

if(NOT TARGET CHOLMOD_static AND NOT TARGET CHOLMOD)
    add_subdirectory(
        "${HOMA_SUITESPARSE_SOURCE_DIR}/CHOLMOD"
        "${HOMA_SUITESPARSE_BINARY_DIR}/CHOLMOD"
        EXCLUDE_FROM_ALL
    )
endif()

_homa_suitesparse_alias(SuiteSparse::CHOLMOD CHOLMOD_static CHOLMOD)
_homa_suitesparse_alias(SuiteSparse::CHOLMOD_static CHOLMOD_static CHOLMOD)

set(COLAMD_INCLUDE_DIR "${HOMA_SUITESPARSE_SOURCE_DIR}/COLAMD/Include" CACHE PATH "COLAMD include directory" FORCE)
set(CAMD_INCLUDE_DIR "${HOMA_SUITESPARSE_SOURCE_DIR}/CAMD/Include" CACHE PATH "CAMD include directory" FORCE)
set(CCOLAMD_INCLUDE_DIR "${HOMA_SUITESPARSE_SOURCE_DIR}/CCOLAMD/Include" CACHE PATH "CCOLAMD include directory" FORCE)
set(CHOLMOD_INCLUDE_DIR "${HOMA_SUITESPARSE_SOURCE_DIR}/CHOLMOD/Include" CACHE PATH "CHOLMOD include directory" FORCE)

set(SUITESPARSE_INCLUDE_DIRS
    "${AMD_INCLUDE_DIR}"
    "${COLAMD_INCLUDE_DIR}"
    "${CAMD_INCLUDE_DIR}"
    "${CCOLAMD_INCLUDE_DIR}"
    "${CHOLMOD_INCLUDE_DIR}"
    "${SUITESPARSE_CONFIG_INCLUDE_DIR}"
    CACHE PATH "SuiteSparse include directories"
    FORCE
)

if(TARGET CHOLMOD_static)
    set(COLAMD_LIBRARY COLAMD_static CACHE STRING "COLAMD library" FORCE)
    set(CAMD_LIBRARY CAMD_static CACHE STRING "CAMD library" FORCE)
    set(CCOLAMD_LIBRARY CCOLAMD_static CACHE STRING "CCOLAMD library" FORCE)
    set(CHOLMOD_LIBRARY CHOLMOD_static CACHE STRING "CHOLMOD library" FORCE)
    set(SUITESPARSE_LIBRARIES
        CHOLMOD_static
        AMD_static
        COLAMD_static
        CAMD_static
        CCOLAMD_static
        SuiteSparseConfig_static
        CACHE STRING "SuiteSparse libraries"
        FORCE
    )
else()
    set(COLAMD_LIBRARY COLAMD CACHE STRING "COLAMD library" FORCE)
    set(CAMD_LIBRARY CAMD CACHE STRING "CAMD library" FORCE)
    set(CCOLAMD_LIBRARY CCOLAMD CACHE STRING "CCOLAMD library" FORCE)
    set(CHOLMOD_LIBRARY CHOLMOD CACHE STRING "CHOLMOD library" FORCE)
    set(SUITESPARSE_LIBRARIES
        CHOLMOD
        AMD
        COLAMD
        CAMD
        CCOLAMD
        SuiteSparseConfig
        CACHE STRING "SuiteSparse libraries"
        FORCE
    )
endif()

message(STATUS "SuiteSparse CHOLMOD configured for examples (CPU-only, no BLAS)")
