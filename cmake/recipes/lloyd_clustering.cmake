include(FetchContent)

# Disable demo executables to avoid pulling libigl, polyscope, and CLI11 as
# transitive deps — fast_ordering already manages those for its benchmarks.
# Exposes target: mesh_clustering_lib
set(LLOYD_BUILD_DEMOS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    lloyd_clustering
    GIT_REPOSITORY https://github.com/BehroozZare/lloyd-clustering.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(lloyd_clustering)
