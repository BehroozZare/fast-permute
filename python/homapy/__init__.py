"""Homa: fast fill-reducing sparse ordering + SPD direct solvers.

Thin Python wrapper over the C++ ``LinSysSolver`` (CHOLMOD / MKL / cuDSS) and
the HOMA ordering. The API mirrors the C++ one:

    import scipy.sparse as sp
    import homapy

    solver = homapy.Solver(backend="cholmod", dtype="float64")
    solver.set_matrix(A)                 # A: SPD CSR (scipy) or cupyx CSR (device)
    solver.ordering(homapy.Options(patch_size=512))   # omit to use backend default
    solver.analyze_pattern()
    solver.factorize()
    x = solver.solve(b)

Refactorization (same sparsity pattern, new values): mutate the matrix values
and call ``refactorize(A)``::

    A.setdiag(A.diagonal() + 0.5)
    solver.refactorize(A)
    x2 = solver.solve(b2)

For GPU-resident matrices (cupyx.scipy.sparse CSR), the device pointers are used
directly (zero-copy); ``solve`` returns a cupy array.
"""

from __future__ import annotations

import numpy as _np

from . import _dll as _dll

_dll.setup()

from . import _homapy  # noqa: E402  (must import after _dll.setup on Windows)

__all__ = [
    "Solver",
    "Options",
    "compute_ordering",
    "spsolve",
    "Backend",
    "has_cholmod",
    "has_mkl",
    "has_cudss",
    "has_cuda",
]

Backend = _homapy.Backend


# --------------------------------------------------------------------------- #
# capability flags
# --------------------------------------------------------------------------- #
def has_cholmod() -> bool:
    return bool(_homapy.HAS_CHOLMOD)


def has_mkl() -> bool:
    return bool(_homapy.HAS_MKL)


def has_cudss() -> bool:
    return bool(_homapy.HAS_CUDSS)


def has_cuda() -> bool:
    return bool(_homapy.HAS_CUDA)


# --------------------------------------------------------------------------- #
# enum / option helpers
# --------------------------------------------------------------------------- #
_BACKENDS = {
    "cholmod": Backend.CPU_CHOLMOD,
    "cpu_cholmod": Backend.CPU_CHOLMOD,
    "mkl": Backend.CPU_MKL,
    "pardiso": Backend.CPU_MKL,
    "cpu_mkl": Backend.CPU_MKL,
    "cudss": Backend.GPU_CUDSS,
    "gpu_cudss": Backend.GPU_CUDSS,
}

_SEPARATOR = {
    "auto": _homapy.SeparatorMethod.AUTO,
    "quotient": _homapy.SeparatorMethod.QUOTIENT,
    "direct": _homapy.SeparatorMethod.DIRECT_METIS,
    "direct_metis": _homapy.SeparatorMethod.DIRECT_METIS,
    "metis": _homapy.SeparatorMethod.DIRECT_METIS,
}

_LOCAL = {
    "amd": _homapy.LocalMethod.AMD,
    "metis": _homapy.LocalMethod.METIS,
    "none": _homapy.LocalMethod.NONE,
}

_PATCH = {
    "greedy": _homapy.PatchMethod.GREEDY,
    "metis": _homapy.PatchMethod.METIS,
    "lloyd": _homapy.PatchMethod.LLOYD,
}

_LLOYD_SEED = {
    "random": _homapy.LloydSeedMethod.RANDOM,
    "morton": _homapy.LloydSeedMethod.MORTON,
    "fps": _homapy.LloydSeedMethod.FPS,
}

_ENUM_MAPS = {
    "separator_method": _SEPARATOR,
    "local_method": _LOCAL,
    "patch_method": _PATCH,
    "lloyd_seed_method": _LLOYD_SEED,
}


def _resolve(name, value):
    """Map a string to the corresponding enum, pass enums through."""
    mapping = _ENUM_MAPS[name]
    if isinstance(value, str):
        key = value.lower()
        if key not in mapping:
            raise ValueError(
                f"invalid {name}={value!r}; expected one of {sorted(mapping)}")
        return mapping[key]
    return value


def _backend_enum(backend):
    if isinstance(backend, Backend):
        result = backend
    else:
        key = str(backend).lower()
        if key not in _BACKENDS:
            raise ValueError(
                f"unknown backend {backend!r}; expected one of "
                f"{sorted(set(_BACKENDS))}")
        result = _BACKENDS[key]
    _check_backend_available(result)
    return result


def _check_backend_available(backend):
    if backend == Backend.GPU_CUDSS and not has_cudss():
        raise RuntimeError("cuDSS backend was not built (HOMA_WITH_CUDSS=OFF)")
    if backend == Backend.CPU_CHOLMOD and not has_cholmod():
        raise RuntimeError("CHOLMOD backend was not built (HOMA_WITH_CHOLMOD=OFF)")
    if backend == Backend.CPU_MKL and not has_mkl():
        raise RuntimeError("MKL backend was not built (HOMA_WITH_MKL=OFF)")


def Options(**kwargs):
    """Build a HOMA ordering ``Options`` object from keyword arguments.

    Enum fields accept friendly strings, e.g.
    ``Options(patch_size=512, separator_method="auto", local_method="amd")``.
    """
    opts = _homapy.Options()
    for key, value in kwargs.items():
        if not hasattr(opts, key):
            raise TypeError(f"Options got an unexpected argument {key!r}")
        if key in _ENUM_MAPS:
            value = _resolve(key, value)
        setattr(opts, key, value)
    return opts


# --------------------------------------------------------------------------- #
# matrix helpers
# --------------------------------------------------------------------------- #
def _is_device(array) -> bool:
    return array is not None and hasattr(array, "__cuda_array_interface__")


def _dtype_for(dtype) -> "_np.dtype":
    dt = _np.dtype(dtype)
    if dt not in (_np.dtype(_np.float32), _np.dtype(_np.float64)):
        raise TypeError("dtype must be float32 or float64")
    return dt


def _host_csr(A, dtype, backend=None):
    import scipy.sparse as sp

    if not sp.issparse(A):
        raise TypeError(
            "set_matrix expects a scipy.sparse (host) or cupyx.scipy.sparse "
            "(device) CSR matrix")
    A = A.tocsr()
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square")

    if backend == Backend.CPU_MKL:
        lower = sp.tril(A, k=-1).nnz
        upper = sp.triu(A, k=1).nnz
        A = A.T.tocsr() if lower and not upper else sp.triu(A).tocsr()

    indptr = _np.ascontiguousarray(A.indptr, dtype=_np.int32)
    indices = _np.ascontiguousarray(A.indices, dtype=_np.int32)
    values = _np.ascontiguousarray(A.data, dtype=dtype)
    return indptr, indices, values, int(A.shape[0]), int(indices.size)


def _device_csr(A, dtype):
    import cupy as cp
    import cupyx.scipy.sparse as csp

    if not csp.isspmatrix_csr(A):
        A = csp.csr_matrix(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError("matrix must be square")
    indptr = cp.ascontiguousarray(A.indptr, dtype=cp.int32)
    indices = cp.ascontiguousarray(A.indices, dtype=cp.int32)
    values = cp.ascontiguousarray(A.data, dtype=dtype)
    return indptr, indices, values, int(A.shape[0]), int(indices.size)


def _array_ptr(array) -> int:
    if _is_device(array):
        return int(array.data.ptr)
    return int(array.ctypes.data)


def _array_equal(left, right) -> bool:
    if left.shape != right.shape:
        return False
    if _is_device(left) or _is_device(right):
        import cupy as cp

        result = cp.array_equal(left, right)
        if hasattr(result, "get"):
            result = result.get()
        return bool(result)
    return bool(_np.array_equal(left, right))


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
class Solver:
    """SPD direct solver with a HOMA (or backend-default) ordering.

    Parameters
    ----------
    backend : {"cholmod", "mkl", "cudss"} or Backend
    dtype   : {"float64", "float32"}
    """

    def __init__(self, backend="cudss", dtype="float32"):
        self._backend = _backend_enum(backend)
        self._dtype = _dtype_for(dtype)
        cls = _homapy._SolverD if self._dtype == _np.float64 else _homapy._SolverF
        self._solver = cls(self._backend)
        self._n = None
        self._is_device = False
        # Borrowed arrays are kept alive here (the C++ solver holds raw pointers).
        self._indptr = None
        self._indices = None
        self._values = None

    @property
    def values(self):
        """The value buffer the solver borrows.

        Overwrite it in place (``solver.values[:] = ...``) and call
        ``factorize()`` to refactorize with the same sparsity pattern. It is a
        numpy array for host matrices and a cupy array for device matrices.

        Note: with the cuDSS backend and a *host* matrix the values are copied
        to the device at ``set_matrix`` time, so use a device (cupy) matrix for
        zero-copy refactorization.
        """
        return self._values

    @property
    def n(self):
        return self._n

    def set_matrix(self, A):
        self._is_device = _is_device(getattr(A, "data", None))
        if self._is_device:
            indptr, indices, values, n, nnz = _device_csr(A, self._dtype)
            self._solver.set_matrix_device(
                int(indptr.data.ptr),
                int(indices.data.ptr),
                int(values.data.ptr),
                n,
                nnz,
            )
        else:
            indptr, indices, values, n, nnz = _host_csr(
                A, self._dtype, self._backend)
            self._solver.set_matrix_host(indptr, indices, values, n)        
        self._indptr, self._indices, self._values = indptr, indices, values
        self._n = n
        return self

    def ordering(self, options=None, **kwargs):
        """Run the HOMA ordering. Skip this call to use the backend default."""
        if options is None:
            options = Options(**kwargs)
        elif kwargs:
            raise TypeError("pass either an Options object or keyword arguments")
        self._solver.ordering(options)
        return self

    def analyze_pattern(self):
        self._solver.analyze_pattern()
        return self

    def factorize(self):
        self._solver.factorize()
        return self

    def refactorize(self, A=None):
        """Update numeric values for the same sparsity pattern and factorize."""
        if A is None:
            return self.factorize()
        if self._values is None:
            raise RuntimeError("refactorize called before set_matrix")

        is_device = _is_device(getattr(A, "data", None))
        if is_device != self._is_device:
            raise TypeError("refactorize matrix must use the same memory location")
        if self._backend == Backend.GPU_CUDSS and not is_device:
            raise RuntimeError(
                "cuDSS refactorize(A) requires a cupyx CSR matrix; host "
                "matrices are copied to device during set_matrix"
            )

        if is_device:
            indptr, indices, values, n, nnz = _device_csr(A, self._dtype)
        else:
            indptr, indices, values, n, nnz = _host_csr(
                A, self._dtype, self._backend)

        if n != self._n or nnz != int(self._indices.shape[0]):
            raise ValueError("refactorize requires the same sparsity pattern")
        if not _array_equal(indptr, self._indptr) or not _array_equal(
            indices, self._indices
        ):
            raise ValueError("refactorize requires the same sparsity pattern")

        if _array_ptr(values) != _array_ptr(self._values):
            self._values[:] = values
        return self.factorize()

    def solve(self, b):
        if self._is_device:
            return self._solve_device(b)
        return self._solve_host(b)

    def reset(self):
        self._solver.reset()
        self._indptr = self._indices = self._values = None
        self._n = None
        return self

    # -- internals ---------------------------------------------------------- #
    def _solve_host(self, b):
        b = _np.asarray(b, dtype=self._dtype)
        if b.ndim == 1:
            rhs = _np.ascontiguousarray(b)
            out = _np.empty(self._n, dtype=self._dtype)
        else:
            rhs = _np.asfortranarray(b)          # packed column-major
            out = _np.empty((self._n, b.shape[1]), dtype=self._dtype, order="F")
        self._solver.solve_host(rhs, out)
        return out

    def _solve_device(self, b):
        import cupy as cp

        b = cp.asarray(b, dtype=self._dtype)
        if b.ndim == 1:
            rhs = cp.ascontiguousarray(b)
            out = cp.empty(self._n, dtype=self._dtype)
            k = 1
        else:
            rhs = cp.asfortranarray(b)
            k = int(b.shape[1])
            out = cp.empty((self._n, k), dtype=self._dtype, order="F")
        self._solver.solve_device(
            int(rhs.data.ptr), int(out.data.ptr), self._n, k)
        cp.cuda.runtime.deviceSynchronize()
        return out


# --------------------------------------------------------------------------- #
# free functions
# --------------------------------------------------------------------------- #
def compute_ordering(A, options=None, **kwargs):
    """Compute a HOMA fill-reducing ordering for the pattern of ``A``.

    Returns ``(perm, etree)`` as numpy int32 arrays. Ordering is CPU-only, so a
    device matrix's pattern is copied to the host first.
    """
    if options is None:
        options = Options(**kwargs)
    elif kwargs:
        raise TypeError("pass either an Options object or keyword arguments")

    if _is_device(getattr(A, "data", None)):
        A = A.get()  # cupyx CSR -> scipy CSR (host)

    import scipy.sparse as sp

    A = sp.csr_matrix(A)
    indptr = _np.ascontiguousarray(A.indptr, dtype=_np.int32)
    indices = _np.ascontiguousarray(A.indices, dtype=_np.int32)
    return _homapy.compute_ordering(indptr, indices, int(A.shape[0]), options)


def _infer_dtype(A, b):
    for obj in (getattr(A, "data", None), b):
        dt = getattr(obj, "dtype", None)
        if dt is not None and _np.dtype(dt) == _np.dtype(_np.float32):
            return "float32"
    return "float64"


def spsolve(A, b, backend="cudss", dtype=None, reorder=True, **ordering_kwargs):
    """One-shot solve of ``A x = b`` for an SPD matrix ``A``.

    ``reorder=True`` uses the HOMA ordering (with any ``ordering_kwargs``);
    ``reorder=False`` uses the backend's own default ordering.
    """
    if dtype is None:
        dtype = _infer_dtype(A, b)
    solver = Solver(backend=backend, dtype=dtype)
    solver.set_matrix(A)
    if reorder:
        solver.ordering(**ordering_kwargs)
    solver.analyze_pattern()
    solver.factorize()
    return solver.solve(b)
