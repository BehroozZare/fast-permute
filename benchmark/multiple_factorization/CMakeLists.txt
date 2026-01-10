# =============================================================================
# Multiple Factorization Benchmarks
# =============================================================================

# -----------------------------------------------------------------------------
# IPC Benchmark (Multiple Factorization)
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_multi_ipc_benchmark ipc_benchmark.cu)
target_include_directories(${PROJECT_NAME}_multi_ipc_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
configure_benchmark_target(${PROJECT_NAME}_multi_ipc_benchmark)

# -----------------------------------------------------------------------------
# SLIM Benchmark (Multiple Factorization)
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_multi_slim_benchmark slim_benchmark.cpp)
target_include_directories(${PROJECT_NAME}_multi_slim_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
configure_benchmark_target(${PROJECT_NAME}_multi_slim_benchmark)

# -----------------------------------------------------------------------------
# Mesh Smoothing Benchmark (Multiple Factorization)
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_multi_mesh_smoothing_benchmark mesh_smoothing_benchmark.cpp)
target_include_directories(${PROJECT_NAME}_multi_mesh_smoothing_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)
configure_benchmark_target(${PROJECT_NAME}_multi_mesh_smoothing_benchmark)
target_link_libraries(${PROJECT_NAME}_multi_mesh_smoothing_benchmark PRIVATE polyscope)

