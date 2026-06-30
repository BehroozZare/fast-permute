# Runs from inside the populated METIS source directory (cwd is set there by
# FetchContent_Declare PATCH_COMMAND). Idempotent: re-running is a no-op.

# 1) Rewrite the TLS probe so CMake's MSVC check_thread_storage test passes
#    (the upstream probe only knows about the GCC `__thread` keyword).
set(_probe "GKlib/conf/check_thread_storage.c")
set(_probe_contents
"#if defined(_MSC_VER)
extern __declspec(thread) int x;
#else
extern __thread int x;
#endif

int main(int argc, char **argv) {
  return 0;
}
")
if(EXISTS "${_probe}")
    file(WRITE "${_probe}" "${_probe_contents}")
endif()

# 2) Also patch GKlib's CMake plumbing: when the TLS probe is missing from the
#    cache, the upstream logic always adds `-D__thread=`, which wipes out our
#    storage class even if the probe later succeeds. Force the probe to be
#    cached as TRUE on MSVC so the empty define is never emitted.
set(_gklib_system "GKlib/GKlibSystem.cmake")
if(EXISTS "${_gklib_system}")
    file(READ "${_gklib_system}" _gklib_contents)
    if(NOT _gklib_contents MATCHES "HOMA_TLS_OVERRIDE")
        string(REPLACE
            "  if(NOT HAVE_THREADLOCALSTORAGE)"
            "  # HOMA_TLS_OVERRIDE: MSVC supports TLS via __declspec(thread);\n  # we map __thread to it below so always treat TLS as available.\n  set(HAVE_THREADLOCALSTORAGE TRUE)\n  if(NOT HAVE_THREADLOCALSTORAGE)"
            _gklib_contents "${_gklib_contents}")
        file(WRITE "${_gklib_system}" "${_gklib_contents}")
    endif()
endif()

# 3) Prepend a TLS shim to GKlib files that use the GCC `__thread` keyword so
#    they expand to MSVC's `__declspec(thread)` on Windows. After this both
#    the declarations in gk_externs.h and the definitions in error.c / memory.c
#    end up in the real .tls section of the resulting .lib.
set(_shim
"#if defined(_MSC_VER) && !defined(HOMA_TLS_SHIM)
# define HOMA_TLS_SHIM
# undef __thread
# define __thread __declspec(thread)
#endif
")

function(_homa_patch_tls _path)
    if(NOT EXISTS "${_path}")
        return()
    endif()
    file(READ "${_path}" _content)
    if(_content MATCHES "HOMA_TLS_SHIM")
        return()
    endif()
    file(WRITE "${_path}" "${_shim}${_content}")
endfunction()

_homa_patch_tls("GKlib/gk_externs.h")
_homa_patch_tls("GKlib/error.c")
_homa_patch_tls("GKlib/memory.c")
