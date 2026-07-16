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

    candidates = glob.glob(os.path.join(libs_dir, "libiomp5*.so*"))
    if not candidates:
        return

    def sort_key(path: str) -> tuple[int, str]:
        basename = os.path.basename(path)
        return (1 if basename == "libiomp5.so" else 0, basename)

    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    mode |= getattr(os, "RTLD_NOW", 0)

    for path in sorted(candidates, key=sort_key):
        try:
            _DLL_HANDLES.append(ctypes.CDLL(path, mode=mode))
            return
        except OSError:
            pass


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
