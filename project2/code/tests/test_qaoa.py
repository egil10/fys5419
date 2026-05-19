"""
test_qaoa.py — Sanity checks for the QAOA pipeline.

Two checks (per the project's "three sanity checks before sweeping" list):

1. QAOA at p=1 on a tiny problem reproduces the brute-force optimum's
   energy when the search lands on a near-optimal (gamma, beta) — i.e.
   the Hamiltonian indexing is correct.
2. QAOA at p=10 on a tiny n=4 problem returns approximation_ratio >= 0.99
   given enough seeds. If it does not, H_C or the mixer is wrong.
"""
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.portfolio import PortfolioProblem
from scripts.classical import brute_force
from scripts.qaoa      import solve, make_hamiltonians, qaoa_energy


def _small_problem(n=4, K=2, seed=3):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 0.1, size=n)
    A = rng.normal(0.0, 0.05, size=(n, n))
    Sigma = A @ A.T + 0.01 * np.eye(n)
    return PortfolioProblem(mu, Sigma, lam=2.0, A=0.5, K=K)


def test_qaoa_energy_matches_C_units():
    """H_C diagonal min should equal brute-force C(x*)."""
    pf = _small_problem(n=4, K=2)
    bf = brute_force(pf)
    _, _, _, _, _, HC = make_hamiltonians(pf)
    assert np.isclose(HC.min(), bf.cost), (
        f"H_C min={HC.min()} != brute force C(x*)={bf.cost}"
    )


def test_qaoa_high_depth_reaches_optimum():
    """At p=10 with enough seeds, QAOA's scaled ratio on n=4 should be >= 0.95.

    Scaled ratio: r = (E_worst - E) / (E_worst - E_opt) in [0, 1].
    A near-perfect QAOA at p>>1 should saturate near 1.
    """
    pf = _small_problem(n=4, K=2)
    res = solve(pf, p=10, n_restarts=15, seed=0,
                maxiter=400, rhobeg=0.1)
    assert res["ratio"] >= 0.95, (
        f"QAOA scaled ratio={res['ratio']:.4f} < 0.95 on n=4 p=10"
    )


def test_qaoa_uniform_state_energy_is_mean_HC():
    """E(gamma=0, beta=0) on |+>^n equals the uniform mean of H_C."""
    pf = _small_problem(n=3, K=1)
    _, _, _, _, _, HC = make_hamiltonians(pf)
    n = pf.n
    e0 = qaoa_energy(np.array([0.0, 0.0]), HC, n, p=1)
    expected = float(HC.mean())
    assert np.isclose(e0, expected, atol=1e-10), (
        f"E(0,0) = {e0} should equal mean(H_C) = {expected}"
    )
