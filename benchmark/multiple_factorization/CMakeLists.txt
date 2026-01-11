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

# -----------------------------------------------------------------------------
# Patch Remeshing Benchmark (Multiple Factorization with changing sparsity)
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_patch_remeshing_benchmark 
    patch_remeshing_benchmark.cpp
    utils/createPatch.cpp
)
target_include_directories(${PROJECT_NAME}_patch_remeshing_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_SOURCE_DIR}/benchmark/single_factorization/utils  # For create_patch_with_metis.h
)
configure_benchmark_target(${PROJECT_NAME}_patch_remeshing_benchmark)
target_link_libraries(${PROJECT_NAME}_patch_remeshing_benchmark PRIVATE remesher polyscope)
