# =============================================================================
# Benchmark CMakeLists.txt - Main entry point for all benchmarks
# =============================================================================

# -----------------------------------------------------------------------------
# Helper function to configure common benchmark target properties
# -----------------------------------------------------------------------------
function(configure_benchmark_target target_name)
    # Configure CUDA device linking
    set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    set_property(TARGET ${target_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

    # Link rxmesh_dev_helpers (which already includes RXMesh)
    target_link_libraries(${target_name} PUBLIC rxmesh_dev_helpers)

    # Disable interprocedural optimization for debug builds
    set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF)

    # Debug compile options
    target_compile_options(${target_name} PUBLIC
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-O0;-Xptxas=-O0;-lineinfo;-Xcompiler=-O0,-g3>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g3;-fno-inline>
    )

    # Configure RPATH for STRUMPACK dependencies if enabled
    if(${PROJECT_NAME}_WITH_STRUMPACK)
        set_target_properties(${target_name} PROPERTIES
            BUILD_RPATH "${STRUMPACK_RPATH_DIRS}"
            INSTALL_RPATH "${STRUMPACK_RPATH_DIRS}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
            LINK_FLAGS "-Wl,--disable-new-dtags"
        )
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Add subdirectories for organized benchmark modules
# -----------------------------------------------------------------------------
add_subdirectory(single_factorization)
add_subdirectory(multiple_factorization)
add_subdirectory(render_helper)