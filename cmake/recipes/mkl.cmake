include_guard(GLOBAL)

function(homa_configure_mkl_runtime target_name)
    if(NOT TARGET Homa::MKL)
        return()
    endif()

    get_property(_runtime_dirs TARGET Homa::MKL PROPERTY homa_mkl_runtime_dirs)

    if(UNIX AND NOT APPLE AND _runtime_dirs)
        set_property(TARGET ${target_name} APPEND PROPERTY BUILD_RPATH ${_runtime_dirs})
    endif()

    if(WIN32 AND _runtime_dirs)
        foreach(_dir IN LISTS _runtime_dirs)
            if(EXISTS "${_dir}")
                file(GLOB _mkl_runtime_dlls "${_dir}/*.dll")
                foreach(_dll IN LISTS _mkl_runtime_dlls)
                    add_custom_command(TARGET ${target_name} POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                                "${_dll}"
                                "$<TARGET_FILE_DIR:${target_name}>"
                        VERBATIM)
                endforeach()
            endif()
        endforeach()
    endif()
endfunction()

if(TARGET Homa::MKL)
    return()
endif()

if(NOT DEFINED HOMA_MKL_ALLOW_DOWNLOAD)
    set(HOMA_MKL_ALLOW_DOWNLOAD OFF)
endif()
if(NOT DEFINED HOMA_MKL_VERSION)
    set(HOMA_MKL_VERSION "2025.3.0")
endif()
if(NOT DEFINED HOMA_MKL_INTERFACE)
    set(HOMA_MKL_INTERFACE "lp64")
endif()
if(NOT DEFINED HOMA_MKL_THREADING)
    set(HOMA_MKL_THREADING "intel")
endif()
if(NOT DEFINED HOMA_MKL_LINKING)
    set(HOMA_MKL_LINKING "sdl")
endif()

set(_HOMA_MKL_SUPPORTED_INTERFACES lp64)
set(_HOMA_MKL_SUPPORTED_THREADING intel sequential)
set(_HOMA_MKL_SUPPORTED_LINKING sdl)

if(NOT HOMA_MKL_INTERFACE IN_LIST _HOMA_MKL_SUPPORTED_INTERFACES)
    message(FATAL_ERROR "HOMA_MKL_INTERFACE=${HOMA_MKL_INTERFACE} is not supported yet. Homa uses int CSR indices and requires LP64 MKL.")
endif()
if(NOT HOMA_MKL_THREADING IN_LIST _HOMA_MKL_SUPPORTED_THREADING)
    message(FATAL_ERROR "HOMA_MKL_THREADING=${HOMA_MKL_THREADING} is not supported yet. Use intel or sequential for the MKL PARDISO example.")
endif()
if(NOT HOMA_MKL_LINKING IN_LIST _HOMA_MKL_SUPPORTED_LINKING)
    message(FATAL_ERROR "HOMA_MKL_LINKING=${HOMA_MKL_LINKING} is not supported yet. Use sdl to link against mkl_rt.")
endif()

function(_homa_mkl_apply_config target_name)
    target_compile_definitions(${target_name} INTERFACE
        HOMA_MKL_INTERFACE_LAYER=MKL_INTERFACE_LP64)

    if(HOMA_MKL_THREADING STREQUAL "intel")
        target_compile_definitions(${target_name} INTERFACE
            HOMA_MKL_THREADING_LAYER=MKL_THREADING_INTEL)
    elseif(HOMA_MKL_THREADING STREQUAL "sequential")
        target_compile_definitions(${target_name} INTERFACE
            HOMA_MKL_THREADING_LAYER=MKL_THREADING_SEQUENTIAL)
    endif()
endfunction()

function(_homa_mkl_make_target mkl_include_dir mkl_link_library mkl_runtime_file)
    if(TARGET MKL::MKL)
        add_library(Homa::MKL INTERFACE IMPORTED GLOBAL)
        target_link_libraries(Homa::MKL INTERFACE MKL::MKL)
        _homa_mkl_apply_config(Homa::MKL)
        return()
    endif()

    if(NOT TARGET MKL::rt)
        add_library(MKL::rt SHARED IMPORTED GLOBAL)
        if(WIN32)
            set_target_properties(MKL::rt PROPERTIES
                IMPORTED_IMPLIB "${mkl_link_library}")
            if(mkl_runtime_file)
                set_target_properties(MKL::rt PROPERTIES
                    IMPORTED_LOCATION "${mkl_runtime_file}")
            else()
                set_target_properties(MKL::rt PROPERTIES
                    IMPORTED_LOCATION "${mkl_link_library}")
            endif()
        else()
            set_target_properties(MKL::rt PROPERTIES
                IMPORTED_LOCATION "${mkl_link_library}")
        endif()
    endif()

    add_library(MKL::MKL INTERFACE IMPORTED GLOBAL)
    set_target_properties(MKL::MKL PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${mkl_include_dir}")
    target_link_libraries(MKL::MKL INTERFACE MKL::rt)

    if(UNIX AND NOT APPLE)
        find_package(Threads REQUIRED)
        target_link_libraries(MKL::MKL INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
        find_library(HOMA_MKL_M_LIBRARY m)
        if(HOMA_MKL_M_LIBRARY)
            target_link_libraries(MKL::MKL INTERFACE ${HOMA_MKL_M_LIBRARY})
        endif()
    endif()

    add_library(Homa::MKL INTERFACE IMPORTED GLOBAL)
    target_link_libraries(Homa::MKL INTERFACE MKL::MKL)
    _homa_mkl_apply_config(Homa::MKL)
endfunction()

function(_homa_mkl_download_wheel package_name platform_tag md5 blake2)
    string(REPLACE "-" "_" wheel_package_name "${package_name}")
    string(SUBSTRING "${blake2}" 0 2 digest_a)
    string(SUBSTRING "${blake2}" 2 2 digest_b)
    string(SUBSTRING "${blake2}" 4 -1 digest_c)

    set(wheel_name "${wheel_package_name}-${HOMA_MKL_VERSION}-py2.py3-none-${platform_tag}.whl")
    set(url "https://files.pythonhosted.org/packages/${digest_a}/${digest_b}/${digest_c}/${wheel_name}")
    set(archive "${CMAKE_BINARY_DIR}/_deps/homa_mkl/${wheel_name}")
    set(extract_dir "${CMAKE_BINARY_DIR}/_deps/homa_mkl/${wheel_package_name}")
    set(marker "${extract_dir}/.homa_extracted")

    if(NOT EXISTS "${marker}")
        file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps/homa_mkl")
        message(STATUS "Downloading Intel MKL wheel: ${wheel_name}")
        file(DOWNLOAD "${url}" "${archive}"
            EXPECTED_MD5 "${md5}"
            SHOW_PROGRESS
            STATUS download_status
            LOG download_log)
        list(GET download_status 0 download_code)
        if(NOT download_code EQUAL 0)
            message(FATAL_ERROR "Failed to download ${wheel_name}: ${download_status}\n${download_log}")
        endif()

        file(REMOVE_RECURSE "${extract_dir}")
        file(MAKE_DIRECTORY "${extract_dir}")
        file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${extract_dir}")
        file(WRITE "${marker}" "")
    endif()

    set(_wheel_roots
        "${extract_dir}"
        "${extract_dir}/${wheel_package_name}-${HOMA_MKL_VERSION}.data/data"
        "${extract_dir}/${wheel_package_name}-${HOMA_MKL_VERSION}.data/data/Library")

    file(GLOB _wheel_data_roots LIST_DIRECTORIES true "${extract_dir}/*.data/data")
    foreach(_wheel_data_root IN LISTS _wheel_data_roots)
        list(APPEND _wheel_roots "${_wheel_data_root}")
        if(EXISTS "${_wheel_data_root}/Library")
            list(APPEND _wheel_roots "${_wheel_data_root}/Library")
        endif()
    endforeach()

    set(_all_wheel_roots ${HOMA_MKL_WHEEL_ROOTS} ${_wheel_roots})
    list(REMOVE_DUPLICATES _all_wheel_roots)
    set(HOMA_MKL_WHEEL_ROOTS ${_all_wheel_roots} PARENT_SCOPE)
endfunction()

function(_homa_mkl_download_openmp_fallback)
    if(NOT HOMA_MKL_VERSION STREQUAL "2025.3.0")
        message(FATAL_ERROR "Intel OpenMP wheel fallback currently supports HOMA_MKL_VERSION=2025.3.0 only.")
    endif()

    unset(HOMA_MKL_WHEEL_ROOTS)

    if(WIN32)
        _homa_mkl_download_wheel(intel-openmp win_amd64
            f27905466e68655d78fc0e3b3d5bebbe
            796905addedd727061b61a85b4fd989754edb628b5be1cd5d161727f98cf4d86)
    elseif(UNIX AND NOT APPLE)
        _homa_mkl_download_wheel(intel-openmp manylinux_2_28_x86_64
            81ed5f7b1a8632891fb221ef3d7d14e2
            6a8768241d0b532f0e41d5b2928b640b00f9bb48945df9921df1d878f42c1d38)
    else()
        message(FATAL_ERROR "Intel OpenMP wheel fallback is not supported on this platform.")
    endif()

    set(HOMA_MKL_OPENMP_DOWNLOAD_ROOTS "${HOMA_MKL_WHEEL_ROOTS}" PARENT_SCOPE)
endfunction()

function(_homa_mkl_download_fallback)
    if(NOT HOMA_MKL_VERSION STREQUAL "2025.3.0")
        message(FATAL_ERROR "MKL wheel fallback currently supports HOMA_MKL_VERSION=2025.3.0 only.")
    endif()

    unset(HOMA_MKL_WHEEL_ROOTS)
    if(WIN32)
        set(platform_tag win_amd64)
        _homa_mkl_download_wheel(mkl ${platform_tag}
            f0b81e2fcde05df3beec309d2123f1b6
            750eb863a60889643e21d5d8d6b619f913703257878979b1df22d344a56276d0)
        _homa_mkl_download_wheel(mkl-devel ${platform_tag}
            293713bf10ab6651a9e549e2dbdc80ab
            5ed6a0f24cf3193610debc8dceddcbac7d1510f68dbffe8f171ed5510c1e4f5e)
        _homa_mkl_download_wheel(mkl-include ${platform_tag}
            41b5c0ec6c5e884ffd16095559269c77
            b9d70445e80511e70b653291daaac068712a780ee0e00bb791d047eed0053c4b)
    elseif(UNIX AND NOT APPLE)
        set(platform_tag manylinux_2_28_x86_64)
        _homa_mkl_download_wheel(mkl ${platform_tag}
            25f1ec8059dea2063a55328d19340e94
            6db4ef531295ed33b929c6c5214421eeebe370f1be22536b6956b4aaf18fdbc5)
        _homa_mkl_download_wheel(mkl-devel ${platform_tag}
            098dc06c7a48911ea0deb2c7e6b81369
            9d000e64ede5280b6fba32837e54e56a63219f3add42ed0ff101c9d2f6c5959f)
        _homa_mkl_download_wheel(mkl-include ${platform_tag}
            808dee3759adae2c0a839798989e302f
            d3e75f8f78044b421f859f1a9f6aa4cc957c4733bee3e716a1d54b79797f241c)
    else()
        message(FATAL_ERROR "Intel MKL wheel fallback is not supported on this platform.")
    endif()

    if(HOMA_MKL_THREADING STREQUAL "intel")
        _homa_mkl_download_openmp_fallback()
        list(APPEND HOMA_MKL_WHEEL_ROOTS ${HOMA_MKL_OPENMP_DOWNLOAD_ROOTS})
        list(REMOVE_DUPLICATES HOMA_MKL_WHEEL_ROOTS)
    endif()

    set(HOMA_MKL_DOWNLOAD_ROOTS "${HOMA_MKL_WHEEL_ROOTS}" PARENT_SCOPE)
endfunction()

function(_homa_mkl_find_manual out_found out_roots)
    set(_roots ${${out_roots}})
    if(_roots)
        list(REMOVE_DUPLICATES _roots)
    endif()

    unset(MKL_INCLUDE_DIRS CACHE)
    unset(HOMA_MKL_RT_LIBRARY CACHE)
    unset(HOMA_MKL_RT_RUNTIME CACHE)
    unset(MKL_INCLUDE_DIRS)
    unset(HOMA_MKL_RT_LIBRARY)
    unset(HOMA_MKL_RT_RUNTIME)

    find_path(MKL_INCLUDE_DIRS
        NAMES mkl.h
        HINTS ${_roots}
        PATH_SUFFIXES
            include
            Library/include
            data/include
            data/Library/include)

    find_library(HOMA_MKL_RT_LIBRARY
        NAMES mkl_rt mkl_rt_dll libmkl_rt.so.2
        HINTS ${_roots}
        PATH_SUFFIXES
            lib
            lib/intel64
            Library/lib
            Library/bin
            data/lib
            data/bin
            data/Library/lib
            data/Library/bin)

    if(WIN32)
        find_file(HOMA_MKL_RT_RUNTIME
            NAMES mkl_rt.2.dll mkl_rt.dll
            HINTS ${_roots}
            PATH_SUFFIXES
                bin
                Library/bin
                Library/lib
                redist/intel64
                data/bin
                data/Library/bin
                data/Library/lib)
    elseif(UNIX AND NOT APPLE)
        find_file(HOMA_MKL_RT_RUNTIME
            NAMES libmkl_rt.so.2 libmkl_rt.so
            HINTS ${_roots}
            PATH_SUFFIXES
                lib
                lib/intel64
                data/lib
                data/Library/lib)
    endif()

    if(MKL_INCLUDE_DIRS AND HOMA_MKL_RT_LIBRARY)
        _homa_mkl_make_target("${MKL_INCLUDE_DIRS}" "${HOMA_MKL_RT_LIBRARY}" "${HOMA_MKL_RT_RUNTIME}")
        if(HOMA_MKL_RT_RUNTIME)
            get_filename_component(_homa_mkl_runtime_dir "${HOMA_MKL_RT_RUNTIME}" DIRECTORY)
        else()
            get_filename_component(_homa_mkl_runtime_dir "${HOMA_MKL_RT_LIBRARY}" DIRECTORY)
        endif()
        set(MKL_LIBRARIES MKL::MKL CACHE STRING "Intel MKL libraries" FORCE)
        set(MKL_INCLUDE_DIRS "${MKL_INCLUDE_DIRS}" CACHE PATH "Intel MKL include directory" FORCE)
        set(HOMA_MKL_RUNTIME_DIRS "${_homa_mkl_runtime_dir}" PARENT_SCOPE)
        set(${out_found} TRUE PARENT_SCOPE)
    else()
        set(${out_found} FALSE PARENT_SCOPE)
    endif()
endfunction()

function(_homa_mkl_find_mkl_runtime_dir out_runtime_dirs out_roots)
    set(_runtime_dirs ${${out_runtime_dirs}})
    set(_roots ${${out_roots}})
    if(_roots)
        list(REMOVE_DUPLICATES _roots)
    endif()

    unset(HOMA_MKL_RT_RUNTIME CACHE)
    unset(HOMA_MKL_RT_RUNTIME)

    if(WIN32)
        find_file(HOMA_MKL_RT_RUNTIME
            NAMES mkl_rt.2.dll mkl_rt.dll
            HINTS ${_roots}
            PATH_SUFFIXES
                bin
                Library/bin
                Library/lib
                redist/intel64
                data/bin
                data/Library/bin
                data/Library/lib)
    elseif(UNIX AND NOT APPLE)
        find_file(HOMA_MKL_RT_RUNTIME
            NAMES libmkl_rt.so.2 libmkl_rt.so
            HINTS ${_roots}
            PATH_SUFFIXES
                lib
                lib/intel64
                data/lib
                data/Library/lib)
    endif()

    if(HOMA_MKL_RT_RUNTIME)
        get_filename_component(_homa_mkl_runtime_dir "${HOMA_MKL_RT_RUNTIME}" DIRECTORY)
        list(APPEND _runtime_dirs "${_homa_mkl_runtime_dir}")
        list(REMOVE_DUPLICATES _runtime_dirs)
    endif()

    set(${out_runtime_dirs} "${_runtime_dirs}" PARENT_SCOPE)
endfunction()

function(_homa_mkl_find_threading_runtime out_found out_runtime_dirs out_roots)
    set(_runtime_dirs ${${out_runtime_dirs}})

    if(NOT HOMA_MKL_THREADING STREQUAL "intel")
        set(${out_found} TRUE PARENT_SCOPE)
        set(${out_runtime_dirs} "${_runtime_dirs}" PARENT_SCOPE)
        return()
    endif()

    set(_roots ${${out_roots}})
    if(DEFINED ENV{ONEAPI_ROOT})
        list(APPEND _roots "$ENV{ONEAPI_ROOT}/compiler/latest")
    endif()
    if(WIN32)
        list(APPEND _roots
            "C:/Program Files (x86)/Intel/oneAPI/compiler/latest"
            "C:/Program Files/Intel/oneAPI/compiler/latest")
    elseif(UNIX AND NOT APPLE)
        list(APPEND _roots "/opt/intel/oneapi/compiler/latest")
    endif()
    list(APPEND _roots ${CMAKE_PREFIX_PATH})
    if(_roots)
        list(REMOVE_DUPLICATES _roots)
    endif()

    unset(HOMA_MKL_OPENMP_RUNTIME CACHE)
    unset(HOMA_MKL_OPENMP_RUNTIME)

    if(WIN32)
        find_file(HOMA_MKL_OPENMP_RUNTIME
            NAMES libiomp5md.dll
            HINTS ${_roots}
            PATH_SUFFIXES
                bin
                Library/bin
                redist/intel64/compiler
                data/Library/bin)
    elseif(UNIX AND NOT APPLE)
        find_file(HOMA_MKL_OPENMP_RUNTIME
            NAMES libiomp5.so
            HINTS ${_roots}
            PATH_SUFFIXES
                lib
                lib/intel64
                Library/lib
                data/lib
                data/Library/lib)
    endif()

    if(HOMA_MKL_OPENMP_RUNTIME)
        get_filename_component(_homa_mkl_openmp_runtime_dir "${HOMA_MKL_OPENMP_RUNTIME}" DIRECTORY)
        list(APPEND _runtime_dirs "${_homa_mkl_openmp_runtime_dir}")
        list(REMOVE_DUPLICATES _runtime_dirs)
        set(${out_found} TRUE PARENT_SCOPE)
        set(${out_runtime_dirs} "${_runtime_dirs}" PARENT_SCOPE)
    else()
        set(${out_found} FALSE PARENT_SCOPE)
        set(${out_runtime_dirs} "${_runtime_dirs}" PARENT_SCOPE)
    endif()
endfunction()

message(STATUS "Configuring Intel MKL for Homa examples")

set(_homa_mkl_roots)
if(MKL_ROOT)
    list(APPEND _homa_mkl_roots "${MKL_ROOT}")
endif()
if(DEFINED ENV{MKLROOT})
    list(APPEND _homa_mkl_roots "$ENV{MKLROOT}")
endif()
if(DEFINED ENV{ONEAPI_ROOT})
    list(APPEND _homa_mkl_roots "$ENV{ONEAPI_ROOT}/mkl/latest")
endif()
if(WIN32)
    list(APPEND _homa_mkl_roots
        "C:/Program Files (x86)/Intel/oneAPI/mkl/latest"
        "C:/Program Files/Intel/oneAPI/mkl/latest")
elseif(UNIX AND NOT APPLE)
    list(APPEND _homa_mkl_roots "/opt/intel/oneapi/mkl/latest")
endif()
list(APPEND _homa_mkl_roots ${CMAKE_PREFIX_PATH})
set(_homa_mkl_searched_roots ${_homa_mkl_roots})

find_package(MKL CONFIG QUIET)
if(TARGET MKL::MKL)
    add_library(Homa::MKL INTERFACE IMPORTED GLOBAL)
    target_link_libraries(Homa::MKL INTERFACE MKL::MKL)
    _homa_mkl_apply_config(Homa::MKL)

    unset(HOMA_MKL_RUNTIME_DIRS)
    _homa_mkl_find_mkl_runtime_dir(HOMA_MKL_RUNTIME_DIRS _homa_mkl_roots)
    _homa_mkl_find_threading_runtime(_homa_mkl_threading_found HOMA_MKL_RUNTIME_DIRS _homa_mkl_roots)
    if(NOT _homa_mkl_threading_found AND HOMA_MKL_ALLOW_DOWNLOAD)
        _homa_mkl_download_openmp_fallback()
        list(APPEND _homa_mkl_roots ${HOMA_MKL_OPENMP_DOWNLOAD_ROOTS})
        list(APPEND _homa_mkl_searched_roots ${HOMA_MKL_OPENMP_DOWNLOAD_ROOTS})
        _homa_mkl_find_threading_runtime(_homa_mkl_threading_found HOMA_MKL_RUNTIME_DIRS _homa_mkl_roots)
    endif()

    if(_homa_mkl_threading_found)
        if(HOMA_MKL_RUNTIME_DIRS)
            set_property(TARGET Homa::MKL PROPERTY homa_mkl_runtime_dirs "${HOMA_MKL_RUNTIME_DIRS}")
        endif()
        set(MKL_FOUND TRUE CACHE BOOL "Intel MKL found" FORCE)
        message(STATUS "Intel MKL configured: MKL::MKL")
        return()
    endif()

    if(_homa_mkl_searched_roots)
        list(REMOVE_DUPLICATES _homa_mkl_searched_roots)
        string(REPLACE ";" "\n  " _homa_mkl_searched_roots_text "${_homa_mkl_searched_roots}")
        set(_homa_mkl_searched_roots_text "  ${_homa_mkl_searched_roots_text}")
    else()
        set(_homa_mkl_searched_roots_text "  <none>")
    endif()
    message(FATAL_ERROR "Intel MKL was found, but HOMA_MKL_THREADING=intel requires the Intel OpenMP runtime (libiomp5md.dll/libiomp5.so). Install Intel OpenMP, set CMAKE_PREFIX_PATH to it, or enable HOMA_MKL_ALLOW_DOWNLOAD.\nSearched roots:\n${_homa_mkl_searched_roots_text}")
endif()

_homa_mkl_find_manual(_homa_mkl_found _homa_mkl_roots)
set(_homa_mkl_threading_found TRUE)
if(_homa_mkl_found)
    _homa_mkl_find_threading_runtime(_homa_mkl_threading_found HOMA_MKL_RUNTIME_DIRS _homa_mkl_roots)
    if(NOT _homa_mkl_threading_found AND HOMA_MKL_ALLOW_DOWNLOAD)
        _homa_mkl_download_openmp_fallback()
        list(APPEND _homa_mkl_roots ${HOMA_MKL_OPENMP_DOWNLOAD_ROOTS})
        list(APPEND _homa_mkl_searched_roots ${HOMA_MKL_OPENMP_DOWNLOAD_ROOTS})
        _homa_mkl_find_threading_runtime(_homa_mkl_threading_found HOMA_MKL_RUNTIME_DIRS _homa_mkl_roots)
    endif()
endif()

if(_homa_mkl_found AND _homa_mkl_threading_found AND HOMA_MKL_RUNTIME_DIRS)
    set_property(TARGET Homa::MKL PROPERTY homa_mkl_runtime_dirs "${HOMA_MKL_RUNTIME_DIRS}")
endif()

if(NOT _homa_mkl_found AND HOMA_MKL_ALLOW_DOWNLOAD)
    _homa_mkl_download_fallback()
    list(APPEND _homa_mkl_searched_roots ${HOMA_MKL_DOWNLOAD_ROOTS})
    _homa_mkl_find_manual(_homa_mkl_found HOMA_MKL_DOWNLOAD_ROOTS)
    if(_homa_mkl_found)
        _homa_mkl_find_threading_runtime(_homa_mkl_threading_found HOMA_MKL_RUNTIME_DIRS HOMA_MKL_DOWNLOAD_ROOTS)
        set_property(TARGET Homa::MKL PROPERTY homa_mkl_downloaded TRUE)
        if(_homa_mkl_threading_found AND HOMA_MKL_RUNTIME_DIRS)
            set_property(TARGET Homa::MKL PROPERTY homa_mkl_runtime_dirs "${HOMA_MKL_RUNTIME_DIRS}")
        endif()
    endif()
endif()

if(NOT _homa_mkl_found)
    if(_homa_mkl_searched_roots)
        list(REMOVE_DUPLICATES _homa_mkl_searched_roots)
        string(REPLACE ";" "\n  " _homa_mkl_searched_roots_text "${_homa_mkl_searched_roots}")
        set(_homa_mkl_searched_roots_text "  ${_homa_mkl_searched_roots_text}")
    else()
        set(_homa_mkl_searched_roots_text "  <none>")
    endif()
    message(FATAL_ERROR "Intel MKL was not found. Set MKL_ROOT/MKLROOT/CMAKE_PREFIX_PATH or enable HOMA_MKL_ALLOW_DOWNLOAD.\nSearched roots:\n${_homa_mkl_searched_roots_text}")
endif()

if(NOT _homa_mkl_threading_found)
    if(_homa_mkl_searched_roots)
        list(REMOVE_DUPLICATES _homa_mkl_searched_roots)
        string(REPLACE ";" "\n  " _homa_mkl_searched_roots_text "${_homa_mkl_searched_roots}")
        set(_homa_mkl_searched_roots_text "  ${_homa_mkl_searched_roots_text}")
    else()
        set(_homa_mkl_searched_roots_text "  <none>")
    endif()
    message(FATAL_ERROR "Intel MKL was found, but HOMA_MKL_THREADING=intel requires the Intel OpenMP runtime (libiomp5md.dll/libiomp5.so). Install Intel OpenMP, set CMAKE_PREFIX_PATH to it, or enable HOMA_MKL_ALLOW_DOWNLOAD.\nSearched roots:\n${_homa_mkl_searched_roots_text}")
endif()

set(MKL_FOUND TRUE CACHE BOOL "Intel MKL found" FORCE)
message(STATUS "Intel MKL configured: ${HOMA_MKL_RT_LIBRARY}")
