# =============================================================================
# Remesher Tool - Mesh upsampling and decimation
# =============================================================================

add_executable(${PROJECT_NAME}_remesher remesher.cpp)

target_include_directories(${PROJECT_NAME}_remesher PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Configure the tool with common dependencies
configure_tool_target(${PROJECT_NAME}_remesher)
