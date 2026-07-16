from __future__ import annotations

import ctypes
import glob
import os
import sys

_DLL_HANDLES = []


def _setup_linux() -> None:
    if sys.platform != "linux":
        return

    package_dir = os.path.dirname(os.path.abspath(__file__))
    libs_dir = os.path.abspath(os.path.join(package_dir, os.pardir, "homapy.libs"))
    if not os.path.isdir(libs_dir):
        return

    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    mode |= getattr(os, "RTLD_NOW", 0)

    def load(pattern: str) -> None:
        for path in sorted(glob.glob(os.path.join(libs_dir, pattern))):
            try:
                _DLL_HANDLES.append(ctypes.CDLL(path, mode=mode))
                return
            except OSError:
                pass

    for pattern in (
        "libiomp5*.so*",
        "libmkl_core.so*",
        "libmkl_intel_lp64.so*",
        "libmkl_intel_thread.so*",
        "libmkl_sequential.so*",
        "libmkl_def.so*",
        "libmkl_avx*.so*",
        "libmkl_mc*.so*",
        "libmkl_rt.so*",
    ):
        load(pattern)


def setup() -> None:
    _setup_linux()

    if sys.platform != "win32":
        return
    if not hasattr(os, "add_dll_directory"):
        return

    candidates = []

    candidates.append(os.path.dirname(os.path.abspath(__file__)))

    for env_var in ("CUDA_PATH", "CUDA_HOME"):
        cuda = os.environ.get(env_var)
        if cuda:
            candidates.append(os.path.join(cuda, "bin"))

    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        candidates.append(os.path.join(prefix, "Library", "bin"))
        candidates.append(os.path.join(prefix, "bin"))

    seen = set()
    for directory in candidates:
        if not directory or directory in seen:
            continue
        seen.add(directory)
        if os.path.isdir(directory):
            try:
                _DLL_HANDLES.append(os.add_dll_directory(directory))
            except OSError:
                pass
