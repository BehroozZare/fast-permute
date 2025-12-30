# =============================================================================
# Multiple Factorization Benchmarks
# =============================================================================

# -----------------------------------------------------------------------------
# General Benchmark (Multiple Factorization)
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_multi_general_benchmark general_benchmark.cu)
target_include_directories(${PROJECT_NAME}_multi_general_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
configure_benchmark_target(${PROJECT_NAME}_multi_general_benchmark)

# -----------------------------------------------------------------------------
# IPC Benchmark (Multiple Factorization)
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_multi_ipc_benchmark ipc_benchmark.cu)
target_include_directories(${PROJECT_NAME}_multi_ipc_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
configure_benchmark_target(${PROJECT_NAME}_multi_ipc_benchmark)

