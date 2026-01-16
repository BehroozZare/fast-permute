# =============================================================================
# Render Separator Generator
# =============================================================================

add_executable(${PROJECT_NAME}_gen_sep_data gen_sep_data.cpp)
target_include_directories(${PROJECT_NAME}_gen_sep_data PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
configure_benchmark_target(${PROJECT_NAME}_gen_sep_data)
