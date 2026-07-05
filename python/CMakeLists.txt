find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)

include(${HOMA_SOURCE_DIR}/cmake/recipes/nanobind.cmake)

nanobind_add_module(_homapy
    STABLE_ABI
    NB_STATIC
    src/homapy_bindings.cpp
)

target_link_libraries(_homapy PRIVATE homa homa_flags)

if(_homa_needs_cuda)
    set_target_properties(_homapy PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if(COMMAND homa_configure_runtime)
    homa_configure_runtime(_homapy)
endif()

install(TARGETS _homapy LIBRARY DESTINATION homapy)

if(WIN32)
    install(DIRECTORY "$<TARGET_FILE_DIR:_homapy>/"
            DESTINATION homapy
            FILES_MATCHING PATTERN "*.dll")
endif()
