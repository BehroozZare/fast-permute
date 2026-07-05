from __future__ import annotations

import argparse

import numpy as np
import scipy.sparse as sp

import homapy


SIDE = 50
DTYPE = np.dtype(np.float64)


def make_spd_grid(side: int) -> sp.csr_matrix:
    main = np.full(side, 2.0, dtype=DTYPE)
    off = -np.ones(side - 1, dtype=DTYPE)
    t = sp.diags((off, main, off), (-1, 0, 1), shape=(side, side), format="csr")
    eye = sp.eye(side, dtype=DTYPE, format="csr")
    return (sp.kron(eye, t) + sp.kron(t, eye) + sp.eye(side * side)).tocsr()


def to_device(A: sp.csr_matrix, b: np.ndarray, device: str):
    if device == "cpu":
        return A, b

    if not homapy.has_cudss():
        raise SystemExit("GPU example requires a homapy build with cuDSS.")

    try:
        import cupy as cp
        import cupyx.scipy.sparse as csp
    except ImportError as exc:
        raise SystemExit("GPU example requires cupy and cupyx. Simply install via `conda install conda-forge::cupy`.") from exc

    try:
        device_count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:
        raise SystemExit(f"CUDA device check failed: {exc}") from exc

    if device_count == 0:
        raise SystemExit("GPU example requires a visible CUDA device.")

    return csp.csr_matrix(A), cp.asarray(b)


def to_backend_array(x: np.ndarray, device: str):
    if device == "cpu":
        return x

    import cupy as cp

    return cp.asarray(x)


def relative_residual(A, x, b, device: str) -> float:
    if device == "gpu":
        import cupy as cp

        return float(cp.linalg.norm(A @ x - b) / cp.linalg.norm(b))
    return float(np.linalg.norm(A @ x - b) / np.linalg.norm(b))


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve an SPD system with homapy.")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()

    backend = "mkl" if args.device == "cpu" else "cudss"

    # Build the system.
    A_cpu = make_spd_grid(SIDE)
    x_true = np.linspace(1.0, 2.0, A_cpu.shape[0], dtype=DTYPE)
    b_cpu = A_cpu @ x_true
    A, b = to_device(A_cpu, b_cpu, args.device)
    x_ref = to_backend_array(x_true, args.device)

    # Solve the system.
    solver = homapy.Solver(backend=backend, dtype=DTYPE.name)
    solver.set_matrix(A)
    solver.ordering(separator_method="auto", local_method="amd", patch_method="greedy")
    solver.analyze_pattern()
    solver.factorize()
    x = solver.solve(b)
    rel = relative_residual(A, x, b, args.device)

    # Update A in place and refactorize without repeating ordering/analysis.
    A.setdiag(A.diagonal() + 0.5)
    b = A @ x_ref
    solver.refactorize(A)
    x = solver.solve(b)
    rel_updated = relative_residual(A, x, b, args.device)

    print(
        f"\n device={args.device}, backend={backend}, "
        f"n={A_cpu.shape[0]}, relative_residual={rel:.3e}, "
        f"updated_relative_residual={rel_updated:.3e}"
    )


if __name__ == "__main__":
    main()
