
# Benchmark executable (using app_benchmark.cpp as the main benchmark)
add_executable(${PROJECT_NAME}_cudss_benchmark cudss_benchmark/cudss_benchmark.cpp)

# Configure CUDA device linking for the benchmark executable
# Enable separable compilation and resolve device symbols at the executable level
set_property(TARGET ${PROJECT_NAME}_cudss_benchmark PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME}_cudss_benchmark PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link only rxmesh_dev_helpers (which already includes RXMesh)
target_link_libraries(${PROJECT_NAME}_cudss_benchmark PUBLIC rxmesh_dev_helpers)

# Add include directory for cudss_benchmark headers
target_include_directories(${PROJECT_NAME}_cudss_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cudss_benchmark)

# Configure RPATH for STRUMPACK dependencies if enabled
if(${PROJECT_NAME}_WITH_STRUMPACK)
    message(STATUS "Setting RPATH for ${PROJECT_NAME}_cudss_benchmark")
    set_target_properties(${PROJECT_NAME}_cudss_benchmark PROPERTIES
            BUILD_RPATH "${STRUMPACK_RPATH_DIRS}"
            INSTALL_RPATH "${STRUMPACK_RPATH_DIRS}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
            # Use RPATH instead of RUNPATH to propagate to transitive dependencies
            LINK_FLAGS "-Wl,--disable-new-dtags"
    )
endif()

# Ordering Benchmark executable
add_executable(${PROJECT_NAME}_ordering_benchmark ordering_benchmark.cpp)

# Configure CUDA device linking for the ordering benchmark executable
set_property(TARGET ${PROJECT_NAME}_ordering_benchmark PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME}_ordering_benchmark PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Link rxmesh_dev_helpers
target_link_libraries(${PROJECT_NAME}_ordering_benchmark PUBLIC rxmesh_dev_helpers)

# Configure RPATH for STRUMPACK dependencies if enabled
if(${PROJECT_NAME}_WITH_STRUMPACK)
    set_target_properties(${PROJECT_NAME}_ordering_benchmark PROPERTIES
            BUILD_RPATH "${STRUMPACK_RPATH_DIRS}"
            INSTALL_RPATH "${STRUMPACK_RPATH_DIRS}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
            LINK_FLAGS "-Wl,--disable-new-dtags"
    )
endif()

# Etree Analysis Benchmark executable
add_executable(${PROJECT_NAME}_etree_analysis etree_analysis/etree_analysis.cpp)

# Configure CUDA device linking for the etree analysis executable
set_property(TARGET ${PROJECT_NAME}_etree_analysis PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${PROJECT_NAME}_etree_analysis PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Add include directory for etree_analysis headers
target_include_directories(${PROJECT_NAME}_etree_analysis PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/etree_analysis)

# Link rxmesh_dev_helpers
target_link_libraries(${PROJECT_NAME}_etree_analysis PUBLIC rxmesh_dev_helpers)

# Configure RPATH for STRUMPACK dependencies if enabled
if(${PROJECT_NAME}_WITH_STRUMPACK)
    set_target_properties(${PROJECT_NAME}_etree_analysis PROPERTIES
            BUILD_RPATH "${STRUMPACK_RPATH_DIRS}"
            INSTALL_RPATH "${STRUMPACK_RPATH_DIRS}"
            INSTALL_RPATH_USE_LINK_PATH TRUE
            LINK_FLAGS "-Wl,--disable-new-dtags"
    )
endif()

# Common compile options for all benchmarks
foreach(tgt ${PROJECT_NAME}_cudss_benchmark ${PROJECT_NAME}_ordering_benchmark ${PROJECT_NAME}_etree_analysis)
  set_property(TARGET ${tgt} PROPERTY INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF)
  target_compile_options(${tgt} PUBLIC
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-O0;-Xptxas=-O0;-lineinfo;-Xcompiler=-O0,-g3>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0;-g3;-fno-inline>
  )
endforeach()

