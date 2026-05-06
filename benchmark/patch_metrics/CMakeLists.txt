# =============================================================================
# Patch Metrics Benchmarks
# =============================================================================

# -----------------------------------------------------------------------------
# Compare Patching Benchmark
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME}_compare_patching compare_patching.cpp)
target_include_directories(${PROJECT_NAME}_compare_patching PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
configure_benchmark_target(${PROJECT_NAME}_compare_patching)
