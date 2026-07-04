if(TARGET nanobind-static)
    return()
endif()

if(NOT DEFINED Python_EXECUTABLE)
    find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

#prefer a nanobind that comes with the active Python environment
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE _homa_nanobind_cmake_dir
    RESULT_VARIABLE _homa_nanobind_query
    ERROR_QUIET)

if(_homa_nanobind_query EQUAL 0 AND EXISTS "${_homa_nanobind_cmake_dir}")
    message(STATUS "Using nanobind from Python env: ${_homa_nanobind_cmake_dir}")
    list(APPEND CMAKE_PREFIX_PATH "${_homa_nanobind_cmake_dir}")
    find_package(nanobind CONFIG REQUIRED)
else()
    #fall back to fetching the source
    message(STATUS "nanobind not found in Python env; fetching source")
    include(FetchContent)
    FetchContent_Declare(nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG        v2.2.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nanobind)
endif()
