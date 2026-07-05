include_guard(GLOBAL)

function(homa_configure_cudss_runtime target_name)
    if(NOT TARGET cudss)
        return()
    endif()

    get_property(_runtime_dirs TARGET cudss PROPERTY homa_cudss_runtime_dirs)
    get_property(_runtime_files TARGET cudss PROPERTY homa_cudss_runtime_files)
    get_property(_runtime_file TARGET cudss PROPERTY homa_cudss_runtime_file)

    if(_runtime_file)
        list(APPEND _runtime_files "${_runtime_file}")
        get_filename_component(_runtime_file_dir "${_runtime_file}" DIRECTORY)
        list(APPEND _runtime_dirs "${_runtime_file_dir}")
    endif()

    if(_runtime_dirs)
        list(REMOVE_DUPLICATES _runtime_dirs)
    endif()
    if(_runtime_files)
        list(REMOVE_DUPLICATES _runtime_files)
    endif()

    if(UNIX AND NOT APPLE AND _runtime_dirs)
        set_property(TARGET ${target_name} APPEND PROPERTY BUILD_RPATH ${_runtime_dirs})
        set_property(TARGET ${target_name} APPEND PROPERTY INSTALL_RPATH ${_runtime_dirs})
    endif()

    if(WIN32)
        set(_runtime_dlls)

        foreach(_dll IN LISTS _runtime_files)
            if(EXISTS "${_dll}" AND _dll MATCHES "\\.dll$")
                list(APPEND _runtime_dlls "${_dll}")
            endif()
        endforeach()

        foreach(_dir IN LISTS _runtime_dirs)
            if(EXISTS "${_dir}")
                file(GLOB _cudss_runtime_dlls "${_dir}/*.dll")
                list(APPEND _runtime_dlls ${_cudss_runtime_dlls})
            endif()
        endforeach()

        if(_runtime_dlls)
            list(REMOVE_DUPLICATES _runtime_dlls)
        endif()

        foreach(_dll IN LISTS _runtime_dlls)
            add_custom_command(TARGET ${target_name} POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_dll}"
                        "$<TARGET_FILE_DIR:${target_name}>"
                VERBATIM)
        endforeach()
    endif()
endfunction()

if(TARGET cudss)
    return()
endif()

if(NOT DEFINED HOMA_CUDSS_ALLOW_DOWNLOAD)
    set(HOMA_CUDSS_ALLOW_DOWNLOAD OFF)
endif()
if(NOT DEFINED HOMA_CUDSS_VERSION)
    set(HOMA_CUDSS_VERSION "0.7.1.6")
endif()

if(CMAKE_CUDA_COMPILER_VERSION)
    string(REGEX MATCH "^[0-9]+" HOMA_CUDSS_CUDA_VERSION_MAJOR "${CMAKE_CUDA_COMPILER_VERSION}")
elseif(CUDAToolkit_VERSION)
    string(REGEX MATCH "^[0-9]+" HOMA_CUDSS_CUDA_VERSION_MAJOR "${CUDAToolkit_VERSION}")
endif()

message(STATUS "Configuring cuDSS for Homa")
if(HOMA_CUDSS_CUDA_VERSION_MAJOR)
    message(STATUS "Detected CUDA toolkit version: ${CMAKE_CUDA_COMPILER_VERSION} (major: ${HOMA_CUDSS_CUDA_VERSION_MAJOR})")
endif()

function(_homa_cudss_clear_manual_cache)
    unset(CUDSS_INCLUDES CACHE)
    unset(CUDSS_LIBRARY CACHE)
    unset(CUDSS_MTLAYER_LIBRARY CACHE)
    unset(CUDSS_RUNTIME_LIBRARY CACHE)
    unset(CUDSS_MTLAYER_RUNTIME_LIBRARY CACHE)
    unset(CUDSS_NCCL_RUNTIME_LIBRARY CACHE)
    unset(CUDSS_OPENMPI_RUNTIME_LIBRARY CACHE)
    unset(CUDSS_INCLUDES)
    unset(CUDSS_LIBRARY)
    unset(CUDSS_MTLAYER_LIBRARY)
    unset(CUDSS_RUNTIME_LIBRARY)
    unset(CUDSS_MTLAYER_RUNTIME_LIBRARY)
    unset(CUDSS_NCCL_RUNTIME_LIBRARY)
    unset(CUDSS_OPENMPI_RUNTIME_LIBRARY)
endfunction()

function(_homa_cudss_download_wheel)
    if(NOT HOMA_CUDSS_VERSION STREQUAL "0.7.1.6")
        message(FATAL_ERROR "cuDSS wheel fallback currently supports HOMA_CUDSS_VERSION=0.7.1.6 only.")
    endif()

    if(WIN32)
        set(_platform_tag "win_amd64")
        set(_wheel_hash "6cbe6f0c404259243faecf2ca13dfe282655d6296b97d5188995166190728435")
        set(_wheel_url "https://files.pythonhosted.org/packages/4d/67/13a8eb884b6cb314fffd8b9e4c60bcb18c8141e1a6ec9b51a95c84c87bde/nvidia_cudss_cu12-0.7.1.6-py3-none-win_amd64.whl")
    elseif(UNIX AND NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$")
        set(_platform_tag "manylinux_2_17_x86_64")
        set(_wheel_hash "8b7837d216e871bb9ac174300a62eb998344f0b5faa03cf8f93d37c28985b325")
        set(_wheel_url "https://files.pythonhosted.org/packages/b9/3b/bc2da119af8328b7096bced5632d3b1045dad9b40cb2f3c64717c535e89f/nvidia_cudss_cu12-0.7.1.6-py3-none-manylinux_2_17_x86_64.whl")
    else()
        message(FATAL_ERROR "cuDSS wheel fallback is only supported on Windows x64 and Linux x86_64.")
    endif()

    set(_wheel_name "nvidia_cudss_cu12-${HOMA_CUDSS_VERSION}-py3-none-${_platform_tag}.whl")
    set(_archive "${HOMA_BINARY_DIR}/_deps/homa_cudss/${_wheel_name}")
    set(_extract_dir "${HOMA_BINARY_DIR}/_deps/homa_cudss/nvidia_cudss_cu12")
    set(_marker "${_extract_dir}/.homa_extracted_${HOMA_CUDSS_VERSION}_${_platform_tag}")

    if(NOT EXISTS "${_marker}")
        file(MAKE_DIRECTORY "${HOMA_BINARY_DIR}/_deps/homa_cudss")
        message(STATUS "Downloading NVIDIA cuDSS wheel: ${_wheel_name}")
        file(DOWNLOAD "${_wheel_url}" "${_archive}"
            EXPECTED_HASH SHA256=${_wheel_hash}
            SHOW_PROGRESS
            STATUS _download_status
            LOG _download_log)
        list(GET _download_status 0 _download_code)
        if(NOT _download_code EQUAL 0)
            message(FATAL_ERROR "Failed to download ${_wheel_name}: ${_download_status}\n${_download_log}")
        endif()

        file(REMOVE_RECURSE "${_extract_dir}")
        file(MAKE_DIRECTORY "${_extract_dir}")
        file(ARCHIVE_EXTRACT INPUT "${_archive}" DESTINATION "${_extract_dir}")
        file(WRITE "${_marker}" "")
    endif()

    set(HOMA_CUDSS_DOWNLOAD_ROOTS
        "${_extract_dir}/nvidia/cu12"
        "${_extract_dir}"
        PARENT_SCOPE)
endfunction()

function(_homa_cudss_read_version includes)
    if(includes AND EXISTS "${includes}/cudss.h")
        file(READ "${includes}/cudss.h" _homa_cudss_version_header)
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
            set(cudss_VERSION "${_homa_cudss_major}.${_homa_cudss_minor}.${_homa_cudss_patch}" PARENT_SCOPE)
        endif()
    endif()
endfunction()

function(_homa_cudss_apply_imported_runtime wrapper_target imported_target)
    set(_homa_cudss_runtime_files)
    set(_homa_cudss_runtime_dirs)

    foreach(_config IN ITEMS "" NOCONFIG DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
        if(_config STREQUAL "")
            set(_location_property IMPORTED_LOCATION)
        else()
            set(_location_property IMPORTED_LOCATION_${_config})
        endif()

        get_target_property(_runtime_file ${imported_target} ${_location_property})
        if(_runtime_file AND EXISTS "${_runtime_file}")
            list(APPEND _homa_cudss_runtime_files "${_runtime_file}")
            get_filename_component(_runtime_dir "${_runtime_file}" DIRECTORY)
            list(APPEND _homa_cudss_runtime_dirs "${_runtime_dir}")
        endif()
    endforeach()

    if(_homa_cudss_runtime_files)
        list(REMOVE_DUPLICATES _homa_cudss_runtime_files)
        set_property(TARGET ${wrapper_target} PROPERTY
            homa_cudss_runtime_files "${_homa_cudss_runtime_files}")
    endif()
    if(_homa_cudss_runtime_dirs)
        list(REMOVE_DUPLICATES _homa_cudss_runtime_dirs)
        set_property(TARGET ${wrapper_target} PROPERTY
            homa_cudss_runtime_dirs "${_homa_cudss_runtime_dirs}")
    endif()
endfunction()

function(_homa_cudss_make_target include_dir libraries runtime_files runtime_dirs)
    set(_homa_cudss_link_libraries "${libraries}")
    set(_homa_cudss_runtime_files_arg "${runtime_files}")
    set(_homa_cudss_primary_library)
    set(_homa_cudss_primary_runtime)

    if(_homa_cudss_link_libraries)
        list(GET _homa_cudss_link_libraries 0 _homa_cudss_primary_library)
    endif()
    if(_homa_cudss_runtime_files_arg)
        list(GET _homa_cudss_runtime_files_arg 0 _homa_cudss_primary_runtime)
    endif()

    if(NOT TARGET cudss::cudss)
        if(WIN32 AND _homa_cudss_primary_library AND _homa_cudss_primary_runtime)
            add_library(cudss::cudss SHARED IMPORTED GLOBAL)
            set_target_properties(cudss::cudss PROPERTIES
                IMPORTED_IMPLIB "${_homa_cudss_primary_library}"
                IMPORTED_LOCATION "${_homa_cudss_primary_runtime}")
        elseif(WIN32 AND _homa_cudss_primary_library)
            add_library(cudss::cudss UNKNOWN IMPORTED GLOBAL)
            set_target_properties(cudss::cudss PROPERTIES
                IMPORTED_LOCATION "${_homa_cudss_primary_library}")
        elseif(_homa_cudss_primary_library)
            add_library(cudss::cudss SHARED IMPORTED GLOBAL)
            set_target_properties(cudss::cudss PROPERTIES
                IMPORTED_LOCATION "${_homa_cudss_primary_library}")
        else()
            add_library(cudss::cudss INTERFACE IMPORTED GLOBAL)
        endif()

        set_target_properties(cudss::cudss PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${include_dir}")

        set(_homa_cudss_extra_libraries "${_homa_cudss_link_libraries}")
        if(_homa_cudss_extra_libraries)
            list(REMOVE_AT _homa_cudss_extra_libraries 0)
            if(_homa_cudss_extra_libraries)
                target_link_libraries(cudss::cudss INTERFACE ${_homa_cudss_extra_libraries})
            endif()
        endif()
    endif()

    if(NOT TARGET cudss)
        add_library(cudss INTERFACE IMPORTED GLOBAL)
        target_link_libraries(cudss INTERFACE cudss::cudss)
    endif()

    if(runtime_files)
        set_property(TARGET cudss PROPERTY homa_cudss_runtime_files "${runtime_files}")
    endif()
    if(runtime_dirs)
        set_property(TARGET cudss PROPERTY homa_cudss_runtime_dirs "${runtime_dirs}")
    endif()
endfunction()

function(_homa_cudss_find_manual out_found out_roots)
    set(_roots ${${out_roots}})
    if(_roots)
        list(REMOVE_DUPLICATES _roots)
    endif()

    _homa_cudss_clear_manual_cache()

    find_path(CUDSS_INCLUDES
        NAMES cudss.h
        HINTS ${_roots}
        PATH_SUFFIXES
            .
            include
            include/cudss
            include/libcudss
            include/libcudss/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
            libcudss/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
            Library/include
            nvidia/cu12/include)

    find_library(CUDSS_LIBRARY
        NAMES cudss libcudss libcudss.so.0
        HINTS ${_roots}
        PATH_SUFFIXES
            lib
            lib64
            bin
            lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
            lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}/lib
            Library/lib
            Library/bin
            nvidia/cu12/lib
            nvidia/cu12/bin)

    find_library(CUDSS_MTLAYER_LIBRARY
        NAMES
            cudss_mtlayer_gomp
            libcudss_mtlayer_gomp
            libcudss_mtlayer_gomp.so.0
            cudss_mtlayer_vcomp140
        HINTS ${_roots}
        PATH_SUFFIXES
            lib
            lib64
            bin
            lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
            lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}/lib
            Library/lib
            Library/bin
            nvidia/cu12/lib
            nvidia/cu12/bin)

    set(_homa_cudss_runtime_files)
    set(_homa_cudss_runtime_dirs)

    if(WIN32)
        find_file(CUDSS_RUNTIME_LIBRARY
            NAMES cudss.dll cudss64_0.dll cudss64_1.dll
            HINTS ${_roots}
            PATH_SUFFIXES
                bin
                lib
                lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
                lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}/bin
                Library/bin
                Library/lib
                nvidia/cu12/bin
                nvidia/cu12/lib)

        find_file(CUDSS_MTLAYER_RUNTIME_LIBRARY
            NAMES cudss_mtlayer_vcomp140.dll cudss_mtlayer_vcomp14064_0.dll
            HINTS ${_roots}
            PATH_SUFFIXES
                bin
                lib
                Library/bin
                Library/lib
                nvidia/cu12/bin
                nvidia/cu12/lib)
    elseif(UNIX AND NOT APPLE)
        find_file(CUDSS_RUNTIME_LIBRARY
            NAMES libcudss.so.0 libcudss.so
            HINTS ${_roots}
            PATH_SUFFIXES
                lib
                lib64
                lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
                lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}/lib
                nvidia/cu12/lib)

        find_file(CUDSS_MTLAYER_RUNTIME_LIBRARY
            NAMES libcudss_mtlayer_gomp.so.0 libcudss_mtlayer_gomp.so
            HINTS ${_roots}
            PATH_SUFFIXES
                lib
                lib64
                lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}
                lib/${HOMA_CUDSS_CUDA_VERSION_MAJOR}/lib
                nvidia/cu12/lib)

        find_file(CUDSS_NCCL_RUNTIME_LIBRARY
            NAMES libcudss_commlayer_nccl.so.0 libcudss_commlayer_nccl.so
            HINTS ${_roots}
            PATH_SUFFIXES lib lib64 nvidia/cu12/lib)

        find_file(CUDSS_OPENMPI_RUNTIME_LIBRARY
            NAMES libcudss_commlayer_openmpi.so.0 libcudss_commlayer_openmpi.so
            HINTS ${_roots}
            PATH_SUFFIXES lib lib64 nvidia/cu12/lib)
    endif()

    foreach(_runtime_file IN ITEMS
        "${CUDSS_RUNTIME_LIBRARY}"
        "${CUDSS_MTLAYER_RUNTIME_LIBRARY}"
        "${CUDSS_NCCL_RUNTIME_LIBRARY}"
        "${CUDSS_OPENMPI_RUNTIME_LIBRARY}")
        if(_runtime_file AND EXISTS "${_runtime_file}")
            list(APPEND _homa_cudss_runtime_files "${_runtime_file}")
            get_filename_component(_runtime_dir "${_runtime_file}" DIRECTORY)
            list(APPEND _homa_cudss_runtime_dirs "${_runtime_dir}")
        endif()
    endforeach()

    if(CUDSS_LIBRARY)
        get_filename_component(_library_dir "${CUDSS_LIBRARY}" DIRECTORY)
        list(APPEND _homa_cudss_runtime_dirs "${_library_dir}")
    endif()

    if(_homa_cudss_runtime_files)
        list(REMOVE_DUPLICATES _homa_cudss_runtime_files)
    endif()
    if(_homa_cudss_runtime_dirs)
        list(REMOVE_DUPLICATES _homa_cudss_runtime_dirs)
    endif()

    if(CUDSS_INCLUDES AND CUDSS_LIBRARY)
        _homa_cudss_read_version("${CUDSS_INCLUDES}")
        set(_homa_cudss_libraries "${CUDSS_LIBRARY}")
        if(CUDSS_MTLAYER_LIBRARY)
            list(APPEND _homa_cudss_libraries "${CUDSS_MTLAYER_LIBRARY}")
        endif()

        _homa_cudss_make_target(
            "${CUDSS_INCLUDES}"
            "${_homa_cudss_libraries}"
            "${_homa_cudss_runtime_files}"
            "${_homa_cudss_runtime_dirs}")
        set(${out_found} TRUE PARENT_SCOPE)
    else()
        set(${out_found} FALSE PARENT_SCOPE)
    endif()
endfunction()

set(_homa_cudss_config_paths)
set(_homa_cudss_install_roots)

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

if(cudss_FOUND AND TARGET cudss)
    _homa_cudss_apply_imported_runtime(cudss cudss)
elseif(cudss_FOUND AND TARGET cudss::cudss AND NOT TARGET cudss)
    add_library(cudss INTERFACE IMPORTED GLOBAL)
    target_link_libraries(cudss INTERFACE cudss::cudss)
    _homa_cudss_apply_imported_runtime(cudss cudss::cudss)
endif()

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
    list(APPEND _homa_cudss_roots "/usr" "/usr/local")
    if(HOMA_CUDSS_CUDA_VERSION_MAJOR)
        list(APPEND _homa_cudss_roots
            "/usr/lib/x86_64-linux-gnu/libcudss/${HOMA_CUDSS_CUDA_VERSION_MAJOR}"
            "/usr/lib/aarch64-linux-gnu/libcudss/${HOMA_CUDSS_CUDA_VERSION_MAJOR}"
            "/usr/local/cuda-${HOMA_CUDSS_CUDA_VERSION_MAJOR}")
    endif()
endif()

set(_homa_cudss_searched_roots ${_homa_cudss_roots})

if(NOT TARGET cudss)
    _homa_cudss_find_manual(_homa_cudss_found _homa_cudss_roots)
else()
    set(_homa_cudss_found TRUE)
endif()

if(NOT _homa_cudss_found AND HOMA_CUDSS_ALLOW_DOWNLOAD)
    _homa_cudss_download_wheel()
    list(APPEND _homa_cudss_searched_roots ${HOMA_CUDSS_DOWNLOAD_ROOTS})
    _homa_cudss_find_manual(_homa_cudss_found HOMA_CUDSS_DOWNLOAD_ROOTS)
endif()

if(NOT _homa_cudss_found)
    if(_homa_cudss_searched_roots)
        list(REMOVE_DUPLICATES _homa_cudss_searched_roots)
        string(REPLACE ";" "\n  " _homa_cudss_searched_roots_text "${_homa_cudss_searched_roots}")
        set(_homa_cudss_searched_roots_text "  ${_homa_cudss_searched_roots_text}")
    else()
        set(_homa_cudss_searched_roots_text "  <none>")
    endif()
    message(FATAL_ERROR "cuDSS was not found. Set cudss_DIR, CUDSS_ROOT, CUDSSROOT, CMAKE_PREFIX_PATH, or enable HOMA_CUDSS_ALLOW_DOWNLOAD.\nSearched roots:\n${_homa_cudss_searched_roots_text}")
endif()

if(NOT TARGET cudss)
    message(FATAL_ERROR "cuDSS was found, but no cudss target was exported.")
endif()

set(cudss_FOUND TRUE CACHE BOOL "cuDSS found" FORCE)

if(cudss_VERSION)
    message(STATUS "Found cuDSS version ${cudss_VERSION}")
else()
    message(STATUS "Found cuDSS")
endif()

unset(_homa_cudss_config_path)
unset(_homa_cudss_config_paths)
unset(_homa_cudss_download_roots)
unset(_homa_cudss_found)
unset(_homa_cudss_install_root)
unset(_homa_cudss_install_roots)
unset(_homa_cudss_root_configs)
unset(_homa_cudss_roots)
unset(_homa_cudss_searched_roots)
unset(_homa_cudss_searched_roots_text)
