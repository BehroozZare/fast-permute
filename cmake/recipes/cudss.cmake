include_guard(GLOBAL)

function(homa_configure_cudss_runtime target_name)
    if(WIN32 AND TARGET cudss)
        get_target_property(_homa_cudss_target_type cudss TYPE)
        get_target_property(_homa_cudss_runtime_file cudss homa_cudss_runtime_file)

        if(_homa_cudss_runtime_file)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_homa_cudss_runtime_file}"
                        "$<TARGET_FILE_DIR:${target_name}>"
                VERBATIM)
        elseif(NOT _homa_cudss_target_type STREQUAL "INTERFACE_LIBRARY")
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "$<TARGET_FILE:cudss>"
                        "$<TARGET_FILE_DIR:${target_name}>"
                VERBATIM)
        endif()
    endif()
endfunction()

if(TARGET cudss)
    return()
endif()

if(CMAKE_CUDA_COMPILER_VERSION)
    string(REGEX MATCH "^[0-9]+" HOMA_CUDSS_CUDA_VERSION_MAJOR "${CMAKE_CUDA_COMPILER_VERSION}")
elseif(CUDAToolkit_VERSION)
    string(REGEX MATCH "^[0-9]+" HOMA_CUDSS_CUDA_VERSION_MAJOR "${CUDAToolkit_VERSION}")
endif()

message(STATUS "Configuring cuDSS for Homa examples")
if(HOMA_CUDSS_CUDA_VERSION_MAJOR)
    message(STATUS "Detected CUDA toolkit version: ${CMAKE_CUDA_COMPILER_VERSION} (major: ${HOMA_CUDSS_CUDA_VERSION_MAJOR})")
endif()

set(_homa_cudss_config_paths)

if(WIN32)
    file(GLOB _homa_cudss_install_roots LIST_DIRECTORIES true
        "C:/Program Files/NVIDIA cuDSS/v*")
    if(_homa_cudss_install_roots)
        list(SORT _homa_cudss_install_roots COMPARE NATURAL ORDER DESCENDING)
    endif()

    foreach(_homa_cudss_install_root IN LISTS _homa_cudss_install_roots)
        if(HOMA_CUDSS_CUDA_VERSION_MAJOR)
            list(APPEND _homa_cudss_config_paths
                "${_homa_cudss_install_root}/lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}/cmake/cudss")
        endif()
        file(GLOB _homa_cudss_root_configs LIST_DIRECTORIES true
            "${_homa_cudss_install_root}/lib/*/cmake/cudss")
        list(APPEND _homa_cudss_config_paths ${_homa_cudss_root_configs})
    endforeach()
elseif(UNIX AND NOT APPLE)
    if(HOMA_CUDSS_CUDA_VERSION_MAJOR)
        list(APPEND _homa_cudss_config_paths
            "/usr/lib/x86_64-linux-gnu/libcudss/${HOMA_CUDSS_CUDA_VERSION_MAJOR}/cmake/cudss"
            "/usr/lib/aarch64-linux-gnu/libcudss/${HOMA_CUDSS_CUDA_VERSION_MAJOR}/cmake/cudss"
            "/usr/local/cuda-${HOMA_CUDSS_CUDA_VERSION_MAJOR}/lib64/cmake/cudss")
    endif()
endif()

list(APPEND _homa_cudss_config_paths ${CMAKE_PREFIX_PATH})
if(_homa_cudss_config_paths)
    list(REMOVE_DUPLICATES _homa_cudss_config_paths)
endif()

if(NOT cudss_DIR OR
   (NOT EXISTS "${cudss_DIR}/cudss-config.cmake" AND
    NOT EXISTS "${cudss_DIR}/cudssConfig.cmake"))
    foreach(_homa_cudss_config_path IN LISTS _homa_cudss_config_paths)
        if(EXISTS "${_homa_cudss_config_path}/cudss-config.cmake" OR
           EXISTS "${_homa_cudss_config_path}/cudssConfig.cmake")
            set(cudss_DIR "${_homa_cudss_config_path}" CACHE PATH "Path to cuDSS CMake config" FORCE)
            message(STATUS "Auto-detected cuDSS CMake config at: ${cudss_DIR}")
            break()
        endif()
    endforeach()
endif()

find_package(cudss CONFIG QUIET)

if(cudss_FOUND AND TARGET cudss::cudss AND NOT TARGET cudss)
    add_library(cudss INTERFACE IMPORTED GLOBAL)
    target_link_libraries(cudss INTERFACE cudss::cudss)
endif()

if(NOT cudss_FOUND)
    set(_homa_cudss_roots)
    if(CUDSS_ROOT)
        list(APPEND _homa_cudss_roots "${CUDSS_ROOT}")
    endif()
    if(DEFINED ENV{CUDSSROOT})
        list(APPEND _homa_cudss_roots "$ENV{CUDSSROOT}")
    endif()
    list(APPEND _homa_cudss_roots ${CMAKE_PREFIX_PATH})

    if(WIN32)
        list(APPEND _homa_cudss_roots ${_homa_cudss_install_roots})
    elseif(UNIX AND NOT APPLE)
        list(APPEND _homa_cudss_roots
            "/usr"
            "/usr/local")
        if(HOMA_CUDSS_CUDA_VERSION_MAJOR)
            list(APPEND _homa_cudss_roots
            "/usr/lib/x86_64-linux-gnu/libcudss/${HOMA_CUDSS_CUDA_VERSION_MAJOR}"
            "/usr/lib/aarch64-linux-gnu/libcudss/${HOMA_CUDSS_CUDA_VERSION_MAJOR}"
            "/usr/local/cuda-${HOMA_CUDSS_CUDA_VERSION_MAJOR}")
        endif()
    endif()

    if(_homa_cudss_roots)
        list(REMOVE_DUPLICATES _homa_cudss_roots)
    endif()

    find_path(CUDSS_INCLUDES
        NAMES cudss.h
        HINTS ${_homa_cudss_roots}
        PATH_SUFFIXES
            .
            include
            include/cudss
            include/libcudss
            include/libcudss/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
            libcudss/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
            Library/include)

    find_library(CUDSS_LIBRARY
        NAMES cudss libcudss
        HINTS ${_homa_cudss_roots}
        PATH_SUFFIXES
            lib
            lib64
            lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
            lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}/lib
            Library/lib
            Library/bin)

    find_library(CUDSS_MTLAYER_LIBRARY
        NAMES cudss_mtlayer_gomp libcudss_mtlayer_gomp
        HINTS ${_homa_cudss_roots}
        PATH_SUFFIXES
            lib
            lib64
            lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
            lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}/lib
            Library/lib
            Library/bin)

    find_library(CUDSS_COMMLAYER_LIBRARY
        NAMES cudss_commlayer_nccl libcudss_commlayer_nccl
        HINTS ${_homa_cudss_roots}
        PATH_SUFFIXES
            lib
            lib64
            lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
            lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}/lib
            Library/lib
            Library/bin)

    if(WIN32)
        find_file(CUDSS_RUNTIME_LIBRARY
            NAMES cudss.dll cudss64_0.dll cudss64_1.dll
            HINTS ${_homa_cudss_roots}
            PATH_SUFFIXES
                bin
                lib
                lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
                lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}/bin
                Library/bin
                Library/lib)
    endif()

    if(CUDSS_INCLUDES AND EXISTS "${CUDSS_INCLUDES}/cudss.h")
        file(READ "${CUDSS_INCLUDES}/cudss.h" _homa_cudss_version_header)
        string(REGEX MATCH "define[ \t]+CUDSS_VER_MAJOR[ \t]+([0-9]+)" _homa_cudss_major_match "${_homa_cudss_version_header}")
        set(_homa_cudss_major "${CMAKE_MATCH_1}")
        string(REGEX MATCH "define[ \t]+CUDSS_VER_MINOR[ \t]+([0-9]+)" _homa_cudss_minor_match "${_homa_cudss_version_header}")
        set(_homa_cudss_minor "${CMAKE_MATCH_1}")
        string(REGEX MATCH "define[ \t]+CUDSS_VER_PATCH[ \t]+([0-9]+)" _homa_cudss_patch_match "${_homa_cudss_version_header}")
        set(_homa_cudss_patch "${CMAKE_MATCH_1}")

        if(NOT _homa_cudss_major)
            string(REGEX MATCH "CUDSS_VERSION_MAJOR[ \t]+([0-9]+)" _homa_cudss_major_match "${_homa_cudss_version_header}")
            set(_homa_cudss_major "${CMAKE_MATCH_1}")
            string(REGEX MATCH "CUDSS_VERSION_MINOR[ \t]+([0-9]+)" _homa_cudss_minor_match "${_homa_cudss_version_header}")
            set(_homa_cudss_minor "${CMAKE_MATCH_1}")
            string(REGEX MATCH "CUDSS_VERSION_PATCH[ \t]+([0-9]+)" _homa_cudss_patch_match "${_homa_cudss_version_header}")
            set(_homa_cudss_patch "${CMAKE_MATCH_1}")
        endif()

        if(_homa_cudss_major)
            set(cudss_VERSION "${_homa_cudss_major}.${_homa_cudss_minor}.${_homa_cudss_patch}")
        endif()
    endif()

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(cudss DEFAULT_MSG CUDSS_INCLUDES CUDSS_LIBRARY)

    if(cudss_FOUND)
        set(_homa_cudss_libraries "${CUDSS_LIBRARY}")
        if(CUDSS_MTLAYER_LIBRARY)
            list(APPEND _homa_cudss_libraries "${CUDSS_MTLAYER_LIBRARY}")
        endif()
        if(CUDSS_COMMLAYER_LIBRARY)
            list(APPEND _homa_cudss_libraries "${CUDSS_COMMLAYER_LIBRARY}")
        endif()

        if(NOT TARGET cudss::cudss)
            add_library(cudss::cudss INTERFACE IMPORTED)
            set_target_properties(cudss::cudss PROPERTIES
                INTERFACE_INCLUDE_DIRECTORIES "${CUDSS_INCLUDES}"
                INTERFACE_LINK_LIBRARIES "${_homa_cudss_libraries}")
        endif()
    endif()

    if(TARGET cudss::cudss AND NOT TARGET cudss)
        add_library(cudss INTERFACE IMPORTED GLOBAL)
        target_link_libraries(cudss INTERFACE cudss::cudss)
        if(CUDSS_RUNTIME_LIBRARY)
            set_property(TARGET cudss PROPERTY homa_cudss_runtime_file "${CUDSS_RUNTIME_LIBRARY}")
        endif()
    endif()
endif()

if(NOT cudss_FOUND)
    message(FATAL_ERROR "cuDSS was not found. Set cudss_DIR, CUDSS_ROOT, CUDSSROOT, or CMAKE_PREFIX_PATH.")
endif()

if(NOT TARGET cudss)
    message(FATAL_ERROR "cuDSS was found, but no cudss target was exported.")
endif()

if(cudss_VERSION)
    message(STATUS "Found cuDSS version ${cudss_VERSION}")
else()
    message(STATUS "Found cuDSS")
endif()

unset(_homa_cudss_config_path)
unset(_homa_cudss_config_paths)
unset(_homa_cudss_install_root)
unset(_homa_cudss_install_roots)
unset(_homa_cudss_libraries)
unset(_homa_cudss_roots)
unset(_homa_cudss_root_configs)
unset(_homa_cudss_runtime_file)
unset(_homa_cudss_target_type)
unset(_homa_cudss_version_header)
