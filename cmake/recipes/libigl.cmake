if(TARGET igl::core)
    return()
endif()

include(FetchContent)

# IMPORTANT: Force libigl to use our Eigen (already fetched in eigen.cmake)
# This prevents libigl from fetching its own Eigen version
set(LIBIGL_USE_STATIC_LIBRARY OFF CACHE BOOL "" FORCE)
set(LIBIGL_EIGEN_DIR "" CACHE PATH "" FORCE)

# Tell libigl that Eigen is already available
if(TARGET Eigen3::Eigen)
    message(STATUS "libigl: Using existing Eigen3::Eigen target")
    set(EIGEN3_INCLUDE_DIR "${EIGEN3_INCLUDE_DIR}" CACHE PATH "" FORCE)
endif()

FetchContent_Declare(
        libigl
        GIT_REPOSITORY https://github.com/libigl/libigl.git
        GIT_TAG main
)
FetchContent_MakeAvailable(libigl)

# Ensure igl::core uses our Eigen
if(TARGET igl::core AND TARGET Eigen3::Eigen)
    get_target_property(IGL_INCLUDES igl::core INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "libigl include dirs: ${IGL_INCLUDES}")
endif()
