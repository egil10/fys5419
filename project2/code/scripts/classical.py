"""
classical.py — Classical baselines for the cardinality portfolio QUBO.

Every solver has the same signature:
    solve(problem, **kwargs) -> SolverResult

Where `problem` is a `Portfolio` (or anything with mu, Sigma, lam, A, K,
n, tickers, and .cost(x)). The uniform `SolverResult` makes the comparison
notebook trivial.

Solvers
-------
- brute_force(pf)            -- ground truth; only callable for n <= ~20
- greedy_top_k(pf, score)    -- pick the K assets with best score
- simulated_annealing(pf)    -- strongest classical baseline
- markowitz_round(pf)        -- continuous Markowitz, round to top-K weights

All return the *binary* vector x that satisfies the budget K (or attempts to;
SA may violate). Cost is evaluated through pf.cost so the comparison stays
on the one true objective defined in portfolio.py.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from time import perf_counter
import numpy as np


# ── Unified result type ──────────────────────────────────────────────────
@dataclass
class SolverResult:
    """Uniform output for every classical (and quantum) solver."""
    name:      str
    x:         np.ndarray              # selected bitstring (binary)
    cost:      float                   # pf.cost(x) — the one true objective
    runtime:   float                   # seconds
    n_evals:   int                     # cost-function evaluations used
    feasible:  bool                    # sum(x) == K
    metadata:  dict = field(default_factory=dict)

    @property
    def bitstring(self) -> str:
        return "".join(map(str, self.x.astype(int)))

    def tickers(self, pf) -> list:
        return [pf.tickers[i] for i in range(pf.n) if self.x[i]]


# ── Brute force ──────────────────────────────────────────────────────────
def brute_force(pf) -> SolverResult:
    """Enumerate all 2^n bitstrings, return the cheapest feasible one."""
    t0 = perf_counter()
    best_x = np.zeros(pf.n, dtype=int)
    best_c = np.inf
    n_evals = 0
    for bits in product([0, 1], repeat=pf.n):
        x = np.array(bits, dtype=int)
        n_evals += 1
        if x.sum() != pf.K:
            continue
        c = pf.cost(x)
        if c < best_c:
            best_c, best_x = c, x
    return SolverResult(
        name="brute_force",
        x=best_x,
        cost=float(best_c),
        runtime=perf_counter() - t0,
        n_evals=n_evals,
        feasible=True,
    )


# ── Greedy top-K ─────────────────────────────────────────────────────────
def greedy_top_k(pf, score: str = "sharpe") -> SolverResult:
    """Pick the K assets with the highest per-asset score.

    score : {"sharpe", "mu", "neg_var"}  asset ranking criterion.
    """
    t0 = perf_counter()
    if score == "sharpe":
        s = pf.mu / np.sqrt(np.diag(pf.Sigma))
    elif score == "mu":
        s = pf.mu.copy()
    elif score == "neg_var":
        s = -np.diag(pf.Sigma)
    else:
        raise ValueError(f"unknown score: {score!r}")

    top = np.argsort(-s)[:pf.K]
    x = np.zeros(pf.n, dtype=int)
    x[top] = 1
    return SolverResult(
        name=f"greedy_{score}",
        x=x,
        cost=pf.cost(x),
        runtime=perf_counter() - t0,
        n_evals=1,
        feasible=bool(x.sum() == pf.K),
        metadata={"score": score},
    )


# ── Simulated annealing ──────────────────────────────────────────────────
def simulated_annealing(
    pf,
    n_sweeps: int = 2_000,
    T0: float = 1.0,
    Tf: float = 1e-3,
    seed: int = 0,
) -> SolverResult:
    """Single-spin-flip Metropolis with geometric temperature schedule.

    The penalty term A (sum x - K)^2 in pf.cost handles the budget softly;
    a swap-style move would enforce K exactly but blurs the comparison.
    """
    rng = np.random.default_rng(seed)
    t0 = perf_counter()

    x = rng.integers(0, 2, size=pf.n)
    e = pf.cost(x)
    best_x, best_e = x.copy(), e

    schedule = T0 * (Tf / T0) ** (np.arange(n_sweeps) / max(n_sweeps - 1, 1))
    n_evals = 1

    for T in schedule:
        for _ in range(pf.n):
            i = rng.integers(pf.n)
            x[i] ^= 1
            e_new = pf.cost(x)
            n_evals += 1
            d = e_new - e
            if d < 0 or rng.random() < np.exp(-d / max(T, 1e-12)):
                e = e_new
                if e < best_e:
                    best_e, best_x = e, x.copy()
            else:
                x[i] ^= 1

    return SolverResult(
        name="simulated_annealing",
        x=best_x,
        cost=float(best_e),
        runtime=perf_counter() - t0,
        n_evals=n_evals,
        feasible=bool(best_x.sum() == pf.K),
        metadata={"n_sweeps": n_sweeps, "T0": T0, "Tf": Tf, "seed": seed},
    )


# ── Markowitz-then-round ─────────────────────────────────────────────────
def markowitz_round(pf) -> SolverResult:
    """Solve continuous Markowitz (no budget penalty) then take top-K weights.

    Continuous problem: minimise -mu^T w + lam w^T Sigma w  s.t. sum w = 1.
    Closed form: w* = (Sigma^{-1} (mu + nu 1)) / (2 lam),
        nu chosen so that 1^T w* = 1.
    Then round: x_i = 1 iff w*_i is in the top K.
    """
    t0 = perf_counter()
    Sinv = np.linalg.inv(pf.Sigma)
    ones = np.ones(pf.n)
    # 1^T Sigma^{-1} (mu + nu 1) = 2 lam  ->  nu = (2 lam - 1^T Sinv mu) / (1^T Sinv 1)
    nu = (2 * pf.lam - ones @ Sinv @ pf.mu) / (ones @ Sinv @ ones)
    w = Sinv @ (pf.mu + nu * ones) / (2 * pf.lam)

    top = np.argsort(-w)[:pf.K]
    x = np.zeros(pf.n, dtype=int)
    x[top] = 1
    return SolverResult(
        name="markowitz_round",
        x=x,
        cost=pf.cost(x),
        runtime=perf_counter() - t0,
        n_evals=1,
        feasible=bool(x.sum() == pf.K),
        metadata={"weights": w},
    )


# ── Roll-up convenience ──────────────────────────────────────────────────
def run_all(pf) -> dict[str, SolverResult]:
    """Run every classical baseline and return them keyed by name."""
    results = {
        "brute_force":         brute_force(pf),
        "greedy_sharpe":       greedy_top_k(pf, score="sharpe"),
        "greedy_mu":           greedy_top_k(pf, score="mu"),
        "markowitz_round":     markowitz_round(pf),
        "simulated_annealing": simulated_annealing(pf),
    }
    return results
