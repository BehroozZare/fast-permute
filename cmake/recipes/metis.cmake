# metis.cmake

if(TARGET metis)
    return()
endif()

include(FetchContent)

# IMPORTANT: GKlib relies on per-thread setjmp buffers declared with GCC's
# `__thread` storage class. Its auto-detection (`check_thread_storage.c`) only
# probes the GCC keyword, so on MSVC it silently disables TLS by defining
# `-D__thread=`. The resulting `metis.lib` is then not safe to call from
# multiple OpenMP threads — every thread races on a shared jmp_buf. We need
# real TLS so our parallel decompose loop can call METIS concurrently.
#
# Fix it at the source via PATCH_COMMAND: rewrite the TLS probe so the MSVC
# path tests `__declspec(thread)` (which works on MSVC) and prepend a portable
# TLS shim to the few GKlib files that use `__thread`. After the patch GKlib
# detects TLS support correctly and `__thread` expands to `__declspec(thread)`,
# so the per-thread state really is per-thread on Windows.
set(_homa_metis_patcher "${CMAKE_CURRENT_LIST_DIR}/metis_tls_patch.cmake")

FetchContent_Declare(
    metis
    GIT_REPOSITORY https://github.com/scivision/METIS.git
    GIT_TAG d4a3aac2a3a0efc18e1de24ae97302ed510f43c7
    PATCH_COMMAND ${CMAKE_COMMAND} -P "${_homa_metis_patcher}"
    UPDATE_DISCONNECTED TRUE
)
FetchContent_MakeAvailable(metis)

# Optionally set IDXTYPEWIDTH and REALTYPEWIDTH
set(IDXTYPEWIDTH 32 CACHE STRING "Width of integer type for METIS")
set(REALTYPEWIDTH 32 CACHE STRING "Width of real type for METIS")

# METIS is added by FetchContent_MakeAvailable

# Include directories and definitions
target_include_directories(metis INTERFACE
    $<BUILD_INTERFACE:${metis_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_definitions(metis INTERFACE
    IDXTYPEWIDTH=${IDXTYPEWIDTH}
    REALTYPEWIDTH=${REALTYPEWIDTH}
)

# Install rules
install(DIRECTORY ${metis_SOURCE_DIR}/include/ DESTINATION include)
install(TARGETS metis EXPORT MetisTargets)
install(EXPORT MetisTargets DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/metis)