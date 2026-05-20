"""
test_classical.py — Sanity checks for the classical baselines.

- brute_force always returns a feasible (sum x = K) bitstring.
- All baselines return binary x in {0, 1}^n.
- brute_force cost <= every other solver's cost (it is the ground truth).
"""
from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.classical import (
    brute_force, greedy_top_k, markowitz_round, simulated_annealing, run_all,
)
from scripts.portfolio import PortfolioProblem


def _small_problem(n=5, K=2, seed=1):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 0.1, size=n)
    A = rng.normal(0.0, 0.05, size=(n, n))
    Sigma = A @ A.T + 0.01 * np.eye(n)
    return PortfolioProblem(mu, Sigma, lam=2.0, A=0.5, K=K)


def test_brute_force_feasible():
    pf = _small_problem()
    bf = brute_force(pf)
    assert bf.feasible
    assert bf.x.sum() == pf.K
    assert set(np.unique(bf.x)).issubset({0, 1})


def test_brute_force_is_lower_bound():
    pf = _small_problem(n=6, K=3, seed=7)
    bf = brute_force(pf)
    for solver in (
        greedy_top_k(pf, score='sharpe'),
        markowitz_round(pf),
        simulated_annealing(pf, n_sweeps=200, seed=2),
    ):
        if solver.feasible:
            assert bf.cost <= solver.cost + 1e-12, (
                f"{solver.name} found cost {solver.cost} < brute force {bf.cost}"
            )


def test_run_all_keys():
    pf = _small_problem()
    results = run_all(pf)
    expected = {"brute_force", "greedy_sharpe", "greedy_mu",
                "markowitz_round", "simulated_annealing"}
    assert set(results.keys()) == expected
