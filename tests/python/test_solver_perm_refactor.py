from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np
import pytest
import scipy.sparse as sp

import homapy


SIDE = 50


@dataclass(frozen=True)
class OrderingCase:
    use_homa: bool
    options: dict[str, object]


ORDERING_CASES = [
    pytest.param(OrderingCase(False, {}), id="default"),
    pytest.param(
        OrderingCase(
            True,
            {
                "separator_method": "auto",
                "local_method": "amd",
                "patch_method": "greedy",
            },
        ),
        id="homa-auto-amd-greedy",
    ),
    pytest.param(
        OrderingCase(
            True,
            {
                "separator_method": "quotient",
                "local_method": "none",
                "patch_method": "greedy",
            },
        ),
        id="homa-quotient-none-greedy",
    ),
    pytest.param(
        OrderingCase(
            True,
            {
                "separator_method": "direct_metis",
                "local_method": "amd",
                "patch_method": "metis",
            },
        ),
        id="homa-direct-amd-metis",
    ),
    pytest.param(
        OrderingCase(
            True,
            {
                "separator_method": "auto",
                "local_method": "metis",
                "patch_method": "greedy",
            },
        ),
        id="homa-auto-metis-greedy",
    ),
]


def _cupy_device_available() -> bool:
    if importlib.util.find_spec("cupy") is None:
        return False
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _backend_params() -> list[pytest.ParameterSet | str]:
    params: list[pytest.ParameterSet | str] = []
    if homapy.has_cholmod():
        params.append(pytest.param("cholmod", id="cholmod"))
    if homapy.has_mkl():
        params.append(pytest.param("mkl", id="mkl"))
    if homapy.has_cudss():
        mark = ()
        if not _cupy_device_available():
            mark = pytest.mark.skip(reason="cuDSS test needs cupy and a CUDA device")
        params.append(pytest.param("cudss", marks=mark, id="cudss"))
    return params


BACKENDS = _backend_params()
if not BACKENDS:
    pytest.skip("no Homa solver backends are available", allow_module_level=True)


def _make_matrix(dtype: np.dtype, shift: float) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for r in range(SIDE):
        for c in range(SIDE):
            i = r * SIDE + c
            rows.append(i)
            cols.append(i)
            data.append(5.0 + shift + i * 0.01)

            if r + 1 < SIDE:
                j = i + SIDE
                rows.extend((i, j))
                cols.extend((j, i))
                data.extend((-1.0, -1.0))

            if c + 1 < SIDE:
                j = i + 1
                rows.extend((i, j))
                cols.extend((j, i))
                data.extend((-1.0, -1.0))

    n = SIDE * SIDE
    return sp.coo_matrix(
        (np.asarray(data, dtype=dtype), (rows, cols)), shape=(n, n)
    ).tocsr()


def _solver_matrix(backend: str, matrix: sp.csr_matrix) -> sp.csr_matrix:
    if backend == "mkl":
        return sp.tril(matrix).tocsr()
    return matrix.tocsr()


def _stored_value_matrix(backend: str, matrix: sp.csr_matrix) -> sp.csr_matrix:
    if backend == "mkl":
        return sp.triu(matrix).tocsr()
    return _solver_matrix(backend, matrix)


def _to_solver_matrix(backend: str, matrix: sp.csr_matrix):
    matrix = _solver_matrix(backend, matrix)
    if backend != "cudss":
        return matrix

    import cupyx.scipy.sparse as csp

    return csp.csr_matrix(matrix)


def _rhs(backend: str, rhs: np.ndarray):
    if backend != "cudss":
        return rhs

    import cupy as cp

    return cp.asarray(rhs)


def _assign_refactor_values(
    solver: homapy.Solver,
    backend: str,
    matrix: sp.csr_matrix,
) -> None:
    values = _stored_value_matrix(backend, matrix).data
    if backend == "cudss":
        import cupy as cp

        values = cp.asarray(values)
    solver.values[:] = values


def _assert_solve(
    backend: str,
    matrix: sp.csr_matrix,
    rhs,
    solution,
    tol: float,
) -> None:
    if backend == "cudss":
        import cupy as cp
        import cupyx.scipy.sparse as csp

        matrix = csp.csr_matrix(matrix)
        rel = float(cp.linalg.norm(matrix @ solution - rhs) / cp.linalg.norm(rhs))
    else:
        rel = float(np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs))

    assert rel < tol


def _assert_valid_permutation(perm: np.ndarray, n: int) -> None:
    assert len(perm) == n
    assert np.array_equal(np.sort(perm), np.arange(n, dtype=perm.dtype))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "dtype,tol",
    [
        pytest.param(np.float32, 1e-3, id="float32"),
        pytest.param(np.float64, 1e-8, id="float64"),
    ],
)
@pytest.mark.parametrize("case", ORDERING_CASES)
def test_solver_permutation_refactorization(
    backend: str,
    dtype: np.dtype,
    tol: float,
    case: OrderingCase,
) -> None:
    a0 = _make_matrix(dtype, 0.0)
    a1 = _make_matrix(dtype, 0.5)
    x_true = np.linspace(1.0, 2.0, a0.shape[0], dtype=dtype)
    b0 = a0 @ x_true
    b1 = a1 @ x_true

    solver = homapy.Solver(backend=backend, dtype=np.dtype(dtype).name)
    solver.set_matrix(_to_solver_matrix(backend, a0))

    if case.use_homa:
        perm, _ = homapy.compute_ordering(a0, **case.options)
        _assert_valid_permutation(perm, a0.shape[0])
        solver.ordering(**case.options)

    solver.analyze_pattern()
    solver.factorize()
    rhs0 = _rhs(backend, b0)
    _assert_solve(backend, a0, rhs0, solver.solve(rhs0), tol)

    _assign_refactor_values(solver, backend, a1)
    solver.factorize()
    rhs1 = _rhs(backend, b1)
    _assert_solve(backend, a1, rhs1, solver.solve(rhs1), tol)
