#
# Parth CMake Recipe
# Downloads and builds the Parth library for fill-reducing orderings in sparse Cholesky factorization
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

set(PARTH_WITH_TESTS OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_API_DEMO OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_CHOLMOD_DEMO OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_ACCELERATE_DEMO OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_MKL_DEMO OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_IPC_DEMO OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_REMESHING_DEMO OFF CACHE BOOL "" FORCE)
set(PARTH_WITH_SOLVER_WRAPPER_DEMO OFF CACHE BOOL "" FORCE)

# Make Parth available
FetchContent_MakeAvailable(parth)

# Parth::parth target is already created by the Parth CMakeLists.txt
message(STATUS "Parth library configured successfully")