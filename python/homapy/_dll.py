from __future__ import annotations

import os
import sys

_DLL_HANDLES = []


def setup() -> None:
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
