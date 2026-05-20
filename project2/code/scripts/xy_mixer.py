"""
xy_mixer.py — Hamming-weight-preserving XY ring mixer for QAOA.

The standard transverse-field mixer H_M = sum_i X_i drives the search
across *all* 2^n bitstrings; the budget constraint sum(x) = K is enforced
only softly through the QUBO penalty A (sum x - K)^2. The XY ring mixer

    H_M^{XY} = (1/2) sum_{i=0}^{n-1} (X_i X_{i+1} + Y_i Y_{i+1})        (1)

(indices mod n) commutes with the total-magnetisation operator
N = sum_i (I - Z_i)/2 = sum_i x_i, and therefore preserves Hamming weight
under U_M(beta) = exp(-i beta H_M^{XY}). Combined with a Dicke initial
state |D^n_K> (equal superposition over all weight-K bitstrings), the
QAOA dynamics stays inside the K-constrained subspace exactly — no soft
penalty, no infeasible bitstrings in the output distribution.

This module builds H_M^{XY} as a sparse matrix and provides a one-shot
applicator using `scipy.sparse.linalg.expm_multiply` so the QAOA loop
can do `psi <- exp(-i beta H_M^{XY}) psi` without ever materialising the
dense exponential.

Conventions match `ising.py`:
- n qubits, dim = 2^n
- qubit 0 is the most significant bit of the basis index
- ring topology: pairs (0, 1), (1, 2), ..., (n-2, n-1), (n-1, 0)

For small n (n <= ~12) the sparse matrix has O(n * 2^(n-1)) non-zeros
and expm_multiply runs in seconds. This is intentionally a research-grade
implementation — not optimised for n >> 14.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply


def _xy_pair_indices(n: int, i: int, j: int):
    """Indices and data for (1/2)(X_i X_j + Y_i Y_j) on n qubits.

    The operator acts non-trivially only on basis states where bits i and
    j differ, and swaps them — i.e. it is a "ring hopping" term. Concretely,

        (X_i X_j + Y_i Y_j) |...01...> = 2 |...10...>   (and the other way)

    so the matrix has entries of +1 (after the (1/2) prefactor) between
    every pair of basis indices that differ only in bits i and j.
    """
    dim = 1 << n
    bit_i = 1 << (n - 1 - i)
    bit_j = 1 << (n - 1 - j)
    rows, cols = [], []
    for k in range(dim):
        bi = (k >> (n - 1 - i)) & 1
        bj = (k >> (n - 1 - j)) & 1
        if bi != bj:
            # Hop: flip both bits at once.
            k2 = k ^ bit_i ^ bit_j
            rows.append(k)
            cols.append(k2)
    data = np.ones(len(rows), dtype=float)
    return rows, cols, data


def build_HM_xy(n: int, periodic: bool = True) -> sp.csr_matrix:
    """Sparse XY ring mixer H_M^{XY} = (1/2) sum_{<i,j>} (X_i X_j + Y_i Y_j).

    Parameters
    ----------
    n : int
        Number of qubits.
    periodic : bool, default True
        If True, include the (n-1, 0) wrap-around pair (ring topology).
        If False, the topology is a line — slightly larger basin coverage
        but doesn't preserve weight as symmetrically across qubits.

    Returns
    -------
    csr_matrix of shape (2^n, 2^n).
    """
    dim = 1 << n
    rows_all, cols_all, data_all = [], [], []
    pairs = [(i, (i + 1) % n) for i in range(n if periodic else n - 1)]
    if not periodic:
        pairs = [(i, i + 1) for i in range(n - 1)]
    for (i, j) in pairs:
        r, c, d = _xy_pair_indices(n, i, j)
        rows_all.extend(r); cols_all.extend(c); data_all.extend(d)
    H = sp.csr_matrix((data_all, (rows_all, cols_all)), shape=(dim, dim))
    # Symmetrise (the per-pair builder above already produces symmetric
    # contributions but be defensive — XY ring is Hermitian) and scale by 1/2.
    H = 0.5 * H
    return H.tocsr()


def number_operator_diag(n: int) -> np.ndarray:
    """Return the diagonal of N = sum_i (I - Z_i)/2 (= total Hamming weight).

    Used for the commutator sanity check `[H_M^{XY}, N] = 0` — if that
    commutator is non-zero on any vector, the mixer leaks out of the
    constrained subspace and the whole point of the XY ring is lost.
    """
    dim = 1 << n
    return np.array([bin(k).count('1') for k in range(dim)], dtype=float)


def apply_xy_mixer(psi: np.ndarray, beta: float,
                   H_M_xy: sp.csr_matrix) -> np.ndarray:
    """Apply U_M(beta) = exp(-i beta H_M^{XY}) to |psi> via expm_multiply.

    `expm_multiply` evaluates exp(-i beta H) @ psi without forming the
    dense exponential — O((nnz(H) + dim) * Krylov_iters) per call,
    typically ~5-30 ms for n <= 10. Beyond n ~ 14 this gets slow and
    you'd want a Trotterised XY mixer instead.
    """
    return expm_multiply(-1j * beta * H_M_xy, psi)
