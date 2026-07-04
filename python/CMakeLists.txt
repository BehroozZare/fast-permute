find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

include(${HOMA_SOURCE_DIR}/cmake/recipes/nanobind.cmake)

nanobind_add_module(_homa
    STABLE_ABI
    NB_STATIC
    src/homa_bindings.cpp
)

target_link_libraries(_homa PRIVATE homa homa_flags)

if(_homa_needs_cuda)
    set_target_properties(_homa PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if(COMMAND homa_configure_runtime)
    homa_configure_runtime(_homa)
endif()

install(TARGETS _homa LIBRARY DESTINATION homa)

if(WIN32)
    install(DIRECTORY "$<TARGET_FILE_DIR:_homa>/"
            DESTINATION homa
            FILES_MATCHING PATTERN "*.dll")
endif()
