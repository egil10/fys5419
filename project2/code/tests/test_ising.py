"""
test_ising.py — QUBO <-> Ising round-trip and H_C diagonal correctness.

These tests catch the silent sign-flip bugs that eat a weekend. Run with:

    pytest project2/code/tests -q
"""
from itertools import product
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.ising     import build_qubo, qubo_to_ising, build_HC_diag, ising_energy
from scripts.portfolio import Portfolio


def _small_problem(n=4, K=2, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 0.1, size=n)
    A = rng.normal(0.0, 0.05, size=(n, n))
    Sigma = A @ A.T + 0.01 * np.eye(n)
    return Portfolio(mu, Sigma, lam=2.0, A=0.5, K=K)


def test_qubo_cost_matches_portfolio_cost():
    """x^T Q x + offset must equal the closed-form C(x) for every bitstring."""
    pf = _small_problem(n=5, K=2)
    Q, off = build_qubo(pf.mu, pf.Sigma, pf.lam, pf.A, pf.K)
    for bits in product([0, 1], repeat=pf.n):
        x = np.array(bits, dtype=float)
        lhs = float(x @ Q @ x + off)
        rhs = pf.cost(x)
        assert np.isclose(lhs, rhs), f"x={bits}: {lhs} != {rhs}"


def test_qubo_to_ising_round_trip():
    """Substituting z_i = 1 - 2 x_i must reproduce the QUBO cost."""
    pf = _small_problem(n=4, K=2)
    Q, off = build_qubo(pf.mu, pf.Sigma, pf.lam, pf.A, pf.K)
    h, J, c = qubo_to_ising(Q, off)
    for bits in product([0, 1], repeat=pf.n):
        x = np.array(bits, dtype=float)
        z = 1 - 2 * x
        qubo = float(x @ Q @ x + off)
        ising = ising_energy(z, h, J, c)
        assert np.isclose(qubo, ising), f"x={bits}: QUBO={qubo}, Ising={ising}"


def test_HC_diag_matches_ising_energies():
    """H_C diagonal entry k = Ising energy of bitstring whose binary expansion is k."""
    pf = _small_problem(n=4, K=2)
    Q, off = build_qubo(pf.mu, pf.Sigma, pf.lam, pf.A, pf.K)
    h, J, _ = qubo_to_ising(Q, off)
    diag = build_HC_diag(h, J)  # NOTE: no const added
    for k in range(1 << pf.n):
        z = np.array([1 - 2 * ((k >> (pf.n - 1 - i)) & 1) for i in range(pf.n)])
        e = ising_energy(z, h, J, 0.0)  # diag excludes const
        assert np.isclose(diag[k], e), f"k={k}: diag={diag[k]}, e={e}"
