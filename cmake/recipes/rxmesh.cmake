#
# RXMesh CMake Recipe
# Downloads and configures the RXMesh library
#

if(TARGET RXMesh)
    return()
endif()

message(STATUS "Third-party: creating target 'RXMesh'")

include(FetchContent)

# Download RXMesh from GitHub
FetchContent_Declare(
    rxmesh
    GIT_REPOSITORY https://github.com/BehroozZare/RXMesh-dev.git
    GIT_TAG main
    GIT_SHALLOW TRUE
)

# Set build options for RXMesh
set(RX_BUILD_TESTS OFF CACHE BOOL "Build RXMesh unit test" FORCE)
set(RX_BUILD_APPS OFF CACHE BOOL "Build RXMesh applications" FORCE)
set(RX_WITH_DEV OFF CACHE BOOL "Add DEV folder to build" FORCE)
set(RX_USE_POLYSCOPE OFF CACHE BOOL "Enable Ployscope for visualization" FORCE)
# Enable CUDSS support if available/needed
set(RX_USE_CUDSS ON CACHE BOOL "Use cuDSS" FORCE)

# Make RXMesh available
FetchContent_MakeAvailable(rxmesh)

message(STATUS "RXMesh library configured successfully")

