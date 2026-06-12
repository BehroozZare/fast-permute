# -----------------------------------------------------------------------------
# Helper to configure common example executable settings.
# -----------------------------------------------------------------------------
function(configure_example_target target_name)
    target_link_libraries(${target_name} PRIVATE homa CLI11::CLI11)
    if(TARGET libigl_utils)
        target_link_libraries(${target_name} PRIVATE libigl_utils)
    endif()

    if(_homa_needs_cuda)
        set_property(TARGET ${target_name} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
        set_property(TARGET ${target_name} PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
    endif()

    set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_DEBUG OFF)
    target_compile_options(${target_name} PRIVATE
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/Od;/Zi>
        $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-O0;-g3;-fno-inline>
        $<$<AND:$<BOOL:${_homa_needs_cuda}>,$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:MSVC>>:-G;-O0;-lineinfo;-Xcompiler=/Od,/Zi>
        $<$<AND:$<BOOL:${_homa_needs_cuda}>,$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:GNU>>:-G;-O0;-Xptxas=-O0;-lineinfo;-Xcompiler=-O0,-g3>
    )

    if(COMMAND homa_configure_mkl_runtime)
        homa_configure_mkl_runtime(${target_name})
    endif()
    if(COMMAND homa_configure_cudss_runtime)
        homa_configure_cudss_runtime(${target_name})
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Example executables
# -----------------------------------------------------------------------------
if(HOMA_WITH_CHOLMOD)
    add_executable(cholmod_example cholmod_example.cpp)
    configure_example_target(cholmod_example)
endif()

if(HOMA_WITH_MKL)
    add_executable(mkl_example mkl_example.cpp)
    configure_example_target(mkl_example)
endif()

if(HOMA_WITH_CUDSS)
    add_executable(cudss_example cudss_example.cpp)
    configure_example_target(cudss_example)
endif()

add_executable(matrix_example matrix_example.cpp)
configure_example_target(matrix_example)
