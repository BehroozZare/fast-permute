# BLAS/LAPACK recipe: find installed or download and build OpenBLAS
#
# This recipe searches for BLAS/LAPACK in the following order:
# 0. Pre-set BLAS_LIBRARIES/LAPACK_LIBRARIES (command line override)
# 1. Intel OneAPI MKL (optimized for Intel processors)
# 2. OpenBLAS (system-installed)
# 3. Any other system BLAS/LAPACK
# 4. Build OpenBLAS from source as fallback

if(TARGET BLAS::BLAS)
    return()
endif()

set(BLAS_FOUND FALSE)
set(LAPACK_FOUND FALSE)

# Optional: Allow user to specify OpenBLAS include directory (for Windows pre-built binaries)
set(OPENBLAS_INCLUDE_DIR "" CACHE PATH "OpenBLAS include directory (optional, for pre-built binaries)")

# =============================================================================
# Step 0: Check for pre-set BLAS/LAPACK libraries (command line override)
# =============================================================================
if(BLAS_LIBRARIES AND LAPACK_LIBRARIES)
    message(STATUS "Using pre-set BLAS_LIBRARIES: ${BLAS_LIBRARIES}")
    message(STATUS "Using pre-set LAPACK_LIBRARIES: ${LAPACK_LIBRARIES}")
    set(BLAS_FOUND TRUE)
    set(LAPACK_FOUND TRUE)
    set(BLAS_VENDOR_FOUND "Pre-set")
    
    # Set include directories for SuiteSparse's SuiteSparseBLAS.cmake
    # This prevents SuiteSparse from trying to run find_package(BLAS) again
    if(OPENBLAS_INCLUDE_DIR)
        set(BLAS_INCLUDE_DIRS "${OPENBLAS_INCLUDE_DIR}" CACHE PATH "BLAS include directories" FORCE)
        set(LAPACK_INCLUDE_DIRS "${OPENBLAS_INCLUDE_DIR}" CACHE PATH "LAPACK include directories" FORCE)
        message(STATUS "Using pre-set BLAS_INCLUDE_DIRS: ${BLAS_INCLUDE_DIRS}")
    endif()
endif()

# =============================================================================
# Step 1: Try Intel OneAPI MKL first (best performance on Intel processors)
# =============================================================================
if(NOT BLAS_FOUND OR NOT LAPACK_FOUND)
    message(STATUS "Searching for Intel OneAPI MKL...")
    set(BLA_VENDOR Intel10_64lp)
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
    
    if(BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "Found Intel MKL BLAS: ${BLAS_LIBRARIES}")
        message(STATUS "Found Intel MKL LAPACK: ${LAPACK_LIBRARIES}")
        set(BLAS_VENDOR_FOUND "Intel MKL")
    endif()
endif()

# =============================================================================
# Step 2: Try OpenBLAS if MKL not found
# =============================================================================
if(NOT BLAS_FOUND OR NOT LAPACK_FOUND)
    message(STATUS "Intel MKL not found, searching for OpenBLAS...")
    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
    
    if(BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "Found OpenBLAS: ${BLAS_LIBRARIES}")
        message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
        set(BLAS_VENDOR_FOUND "OpenBLAS")
    endif()
endif()

# =============================================================================
# Step 3: Try any system BLAS/LAPACK (e.g., Apple Accelerate, ATLAS, etc.)
# =============================================================================
if(NOT BLAS_FOUND OR NOT LAPACK_FOUND)
    message(STATUS "OpenBLAS not found, searching for any system BLAS/LAPACK...")
    unset(BLA_VENDOR)
    find_package(BLAS QUIET)
    find_package(LAPACK QUIET)
    
    if(BLAS_FOUND AND LAPACK_FOUND)
        message(STATUS "Found system BLAS: ${BLAS_LIBRARIES}")
        message(STATUS "Found system LAPACK: ${LAPACK_LIBRARIES}")
        set(BLAS_VENDOR_FOUND "System")
    endif()
endif()

# =============================================================================
# Create imported targets if system BLAS/LAPACK was found
# =============================================================================
if(BLAS_FOUND AND LAPACK_FOUND)
    message(STATUS "Using ${BLAS_VENDOR_FOUND} BLAS/LAPACK")
    
    # Create imported targets if they don't exist (CMake < 3.18 compatibility)
    if(NOT TARGET BLAS::BLAS)
        add_library(BLAS::BLAS INTERFACE IMPORTED)
        set_target_properties(BLAS::BLAS PROPERTIES
            INTERFACE_LINK_LIBRARIES "${BLAS_LIBRARIES}"
        )
    endif()
    
    if(NOT TARGET LAPACK::LAPACK)
        add_library(LAPACK::LAPACK INTERFACE IMPORTED)
        set_target_properties(LAPACK::LAPACK PROPERTIES
            INTERFACE_LINK_LIBRARIES "${LAPACK_LIBRARIES}"
        )
    endif()
else()
    # =============================================================================
    # Step 4: Build OpenBLAS from source as fallback
    # =============================================================================
    message(STATUS "No system BLAS/LAPACK found, downloading and building OpenBLAS from source...")
    message(STATUS "Note: Building OpenBLAS from source may take several minutes.")
    
    include(FetchContent)
    
    # Configure OpenBLAS build options before fetching
    set(BUILD_WITHOUT_LAPACK OFF CACHE BOOL "Build OpenBLAS with LAPACK")
    set(BUILD_TESTING OFF CACHE BOOL "Disable OpenBLAS tests")
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build static OpenBLAS")
    set(BUILD_STATIC_LIBS ON CACHE BOOL "Build static OpenBLAS")
    
    # Use pthreads for threading (more portable)
    set(USE_OPENMP OFF CACHE BOOL "Disable OpenMP in OpenBLAS")
    
    # Disable features not needed
    set(BUILD_RELAPACK OFF CACHE BOOL "Disable ReLAPACK")
    set(BUILD_LAPACK_DEPRECATED OFF CACHE BOOL "Disable deprecated LAPACK functions")
    
    # Use dynamic architecture detection for portability
    set(DYNAMIC_ARCH ON CACHE BOOL "Enable dynamic architecture detection")
    
    # Disable Fortran if not available
    include(CheckLanguage)
    check_language(Fortran)
    if(NOT CMAKE_Fortran_COMPILER)
        set(NOFORTRAN ON CACHE BOOL "Build without Fortran")
        set(C_LAPACK ON CACHE BOOL "Use C version of LAPACK")
    endif()
    
    FetchContent_Declare(
        openblas
        GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
        GIT_TAG v0.3.28
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(openblas)
    
    # Create BLAS::BLAS interface target pointing to OpenBLAS
    if(NOT TARGET BLAS::BLAS)
        add_library(BLAS::BLAS INTERFACE IMPORTED)
        if(TARGET openblas)
            set_target_properties(BLAS::BLAS PROPERTIES
                INTERFACE_LINK_LIBRARIES openblas
            )
        elseif(TARGET openblas_static)
            set_target_properties(BLAS::BLAS PROPERTIES
                INTERFACE_LINK_LIBRARIES openblas_static
            )
        endif()
    endif()
    
    # Create LAPACK::LAPACK interface target (OpenBLAS includes LAPACK)
    if(NOT TARGET LAPACK::LAPACK)
        add_library(LAPACK::LAPACK INTERFACE IMPORTED)
        if(TARGET openblas)
            set_target_properties(LAPACK::LAPACK PROPERTIES
                INTERFACE_LINK_LIBRARIES openblas
            )
        elseif(TARGET openblas_static)
            set_target_properties(LAPACK::LAPACK PROPERTIES
                INTERFACE_LINK_LIBRARIES openblas_static
            )
        endif()
    endif()
    
    # Set BLAS/LAPACK found flags
    set(BLAS_FOUND TRUE CACHE BOOL "BLAS found (via OpenBLAS)" FORCE)
    set(LAPACK_FOUND TRUE CACHE BOOL "LAPACK found (via OpenBLAS)" FORCE)
    
    # Set library variables for other modules that may need them
    if(TARGET openblas)
        set(BLAS_LIBRARIES openblas CACHE STRING "BLAS libraries" FORCE)
        set(LAPACK_LIBRARIES openblas CACHE STRING "LAPACK libraries" FORCE)
    elseif(TARGET openblas_static)
        set(BLAS_LIBRARIES openblas_static CACHE STRING "BLAS libraries" FORCE)
        set(LAPACK_LIBRARIES openblas_static CACHE STRING "LAPACK libraries" FORCE)
    endif()
    
    # Set include directories for SuiteSparse's SuiteSparseBLAS.cmake
    # This prevents SuiteSparse from trying to run find_package(BLAS) again
    set(BLAS_INCLUDE_DIRS "${openblas_SOURCE_DIR}" CACHE PATH "BLAS include directories" FORCE)
    set(LAPACK_INCLUDE_DIRS "${openblas_SOURCE_DIR}" CACHE PATH "LAPACK include directories" FORCE)
    
    message(STATUS "OpenBLAS will be built from source: ${openblas_SOURCE_DIR}")
endif()