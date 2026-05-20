"""
ising.py — QUBO ↔ Ising mapping and Hamiltonian builders.

The cost we minimise:
    C(x) = -mu^T x + lam x^T Sigma x + A (sum x - K)^2,   x in {0,1}^n.

Pipeline:
    Portfolio (mu, Sigma, lam, A, K)
        -> QUBO Q (upper-triangular n x n, plus offset)
        -> Ising (h, J, const) via x_i = (1 - z_i)/2
        -> H_C diagonal (over the 2^n computational basis)
        -> H_M = sum_i X_i (sparse, for the mixer)

Conventions
-----------
- Qubit 0 is the most significant bit of the basis index.
- Spin z_i = +1 if x_i = 0, -1 if x_i = 1.
- All builders are pure functions; they take problem parameters and return
  numpy arrays / sparse matrices. No state, no plotting.

A round-trip sanity check (ising_energy(z) == qubo_cost(x)) lives in
tests/test_ising.py.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp


# ── QUBO construction ────────────────────────────────────────────────────
def build_qubo(mu: np.ndarray, Sigma: np.ndarray,
               lam: float, A: float, K: int):
    """Return (Q, offset) for the mean-variance cardinality QUBO.

    Q is upper-triangular: Q[i,i] is the linear coefficient, Q[i,j] (i<j)
    is the symmetric pair coefficient. The constant offset comes from
    expanding A (sum x - K)^2 = A K^2 - 2 A K sum x_i + A (sum x_i)^2 and
    folding A x_i^2 = A x_i into the diagonal.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, i] = -mu[i] + lam * Sigma[i, i] + A * (1 - 2 * K)
        for j in range(i + 1, n):
            Q[i, j] = 2 * lam * Sigma[i, j] + 2 * A
    offset = A * K ** 2
    return Q, offset


def qubo_cost(x: np.ndarray, Q: np.ndarray, offset: float) -> float:
    """Evaluate the QUBO cost x^T Q x + offset for a binary vector x."""
    x = np.asarray(x, dtype=float)
    return float(np.einsum("i,ij,j->", x, Q, x) + offset)


# ── QUBO -> Ising ────────────────────────────────────────────────────────
def qubo_to_ising(Q: np.ndarray, offset: float = 0.0):
    """Substitute x_i = (1 - z_i)/2 and collect linear, quadratic, constant.

    Returns
    -------
    h     : (n,)        linear Z coefficients
    J     : (n, n)      upper-triangular ZZ coefficients
    const : float       additive scalar
    """
    n = Q.shape[0]
    h = np.zeros(n)
    J = np.zeros((n, n))
    c = float(offset)

    c += np.sum(np.diag(Q)) / 2
    h -= np.diag(Q) / 2
    for i in range(n):
        for j in range(i + 1, n):
            qij = Q[i, j]
            c    += qij / 4
            h[i] -= qij / 4
            h[j] -= qij / 4
            J[i, j] = qij / 4
    return h, J, c


def ising_energy(z: np.ndarray, h: np.ndarray, J: np.ndarray,
                 const: float = 0.0) -> float:
    """Classical Ising energy H(z) = sum h_i z_i + sum_{i<j} J_ij z_i z_j + const."""
    z = np.asarray(z, dtype=float)
    e = float(h @ z) + const
    n = z.size
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j]:
                e += J[i, j] * z[i] * z[j]
    return e


# ── Hamiltonian builders ─────────────────────────────────────────────────
def build_HC_diag(h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Return the diagonal of H_C = sum h_i Z_i + sum J_ij Z_i Z_j.

    H_C is diagonal in the computational basis, so we store just the 2^n
    diagonal entries. Element k is the Ising energy of the bitstring whose
    binary expansion is k (qubit 0 = MSB).
    """
    n = h.size
    dim = 1 << n
    spins = np.empty((dim, n))
    for k in range(dim):
        for i in range(n):
            spins[k, i] = 1 - 2 * ((k >> (n - 1 - i)) & 1)
    diag = spins @ h
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j]:
                diag += J[i, j] * spins[:, i] * spins[:, j]
    return diag


def build_HM(n: int) -> sp.csr_matrix:
    """Sparse H_M = sum_i X_i on n qubits.

    Each X_i is a permutation matrix that flips bit i of the basis index;
    summing them gives a 2^n x 2^n sparse matrix with at most n non-zeros
    per row.
    """
    dim = 1 << n
    rows, cols = [], []
    for k in range(dim):
        for i in range(n):
            rows.append(k)
            cols.append(k ^ (1 << (n - 1 - i)))
    data = np.ones(len(rows))
    return sp.csr_matrix((data, (rows, cols)), shape=(dim, dim))


# ── Convenience: end-to-end from a Portfolio ─────────────────────────────
def from_portfolio(pf):
    """Build (Q, offset, h, J, const, H_C_diag) for a Portfolio object.

    `pf` only needs the attributes mu, Sigma, lam, A, K.
    """
    Q, off = build_qubo(pf.mu, pf.Sigma, pf.lam, pf.A, pf.K)
    h, J, c = qubo_to_ising(Q, off)
    HC = build_HC_diag(h, J)
    return Q, off, h, J, c, HC
