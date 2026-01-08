# =============================================================================
# Single Factorization Benchmarks
# =============================================================================

# -----------------------------------------------------------------------------
# Laplace Benchmark
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_tri_mesh_laplace_benchmark tri_mesh_laplace_benchmark.cpp)
target_include_directories(${PROJECT_NAME}_tri_mesh_laplace_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
configure_benchmark_target(${PROJECT_NAME}_tri_mesh_laplace_benchmark)

# -----------------------------------------------------------------------------
# General Benchmark
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_tet_mesh_laplace_benchmark tet_mesh_laplace_benchmark.cpp)
target_include_directories(${PROJECT_NAME}_tet_mesh_laplace_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
configure_benchmark_target(${PROJECT_NAME}_tet_mesh_laplace_benchmark)

