# =============================================================================
# Tools CMakeLists.txt - Main entry point for all tools
# =============================================================================

# -----------------------------------------------------------------------------
# Helper function to configure common tool target properties
# -----------------------------------------------------------------------------
function(configure_tool_target target_name)
    # Link common dependencies for tools (CPU-only, no CUDA required)
    target_link_libraries(${target_name} PRIVATE
        igl::core
        Eigen3::Eigen
        CLI11::CLI11
        spdlog::spdlog
    )

    # Debug compile options
    target_compile_options(${target_name} PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g3;-fno-inline>
    )
endfunction()

# -----------------------------------------------------------------------------
# Add subdirectories for tools
# -----------------------------------------------------------------------------
add_subdirectory(Remesher)
