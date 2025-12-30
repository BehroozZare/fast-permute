# =============================================================================
# Single Factorization Benchmarks
# =============================================================================

# -----------------------------------------------------------------------------
# Laplace Benchmark
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_laplace_benchmark laplace_benchmark.cpp)
target_include_directories(${PROJECT_NAME}_laplace_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
configure_benchmark_target(${PROJECT_NAME}_laplace_benchmark)

# -----------------------------------------------------------------------------
# General Benchmark
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_general_benchmark general_benchmark.cpp)
target_include_directories(${PROJECT_NAME}_general_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
configure_benchmark_target(${PROJECT_NAME}_general_benchmark)

