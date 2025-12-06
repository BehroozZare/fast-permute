#
# Parth CMake Recipe
# Downloads and builds the Parth library for fill-reducing orderings in sparse Cholesky factorization
#
# Uses FetchContent_Populate + add_subdirectory to isolate Parth's CMake settings
# from propagating to the parent project.
#

if(TARGET parth)
    return()
endif()

include(FetchContent)
FetchContent_Declare(
    parth
    GIT_REPOSITORY https://github.com/BehroozZare/Parth.git
    GIT_TAG main
)

# Disable Parth's optional components
set(PARTH_WITH_TESTS OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_API_DEMO OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_CHOLMOD_DEMO OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_ACCELERATE_DEMO OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_MKL_DEMO OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_IPC_DEMO OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_REMESHING_DEMO OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_SOLVER_WRAPPER_DEMO OFF CACHE BOOL "" FORCE)

# Manually populate instead of FetchContent_MakeAvailable for better isolation
FetchContent_GetProperties(parth)
if(NOT parth_POPULATED)
    FetchContent_Populate(parth)
    # EXCLUDE_FROM_ALL: prevents Parth's install rules from running
    # SYSTEM (CMake 3.25+): treats Parth's includes as system headers (suppresses warnings)
    add_subdirectory(${parth_SOURCE_DIR} ${parth_BINARY_DIR} EXCLUDE_FROM_ALL SYSTEM)
endif()

# Parth::parth target is created by the Parth CMakeLists.txt
message(STATUS "Parth library configured successfully")