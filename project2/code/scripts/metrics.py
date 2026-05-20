"""
metrics.py — Comparison metrics for QAOA vs classical baselines.

Tiny but isolated so plots and tables pull from one place.
"""
from __future__ import annotations
import numpy as np


def approximation_ratio(energy: float, ground_state: float,
                        worst: float | None = None) -> float:
    """Approximation ratio r = E / E_0 when both are negative cost values.

    If a `worst` (highest) energy is supplied, returns the scaled ratio
        r = (E - worst) / (E_0 - worst)  in [0, 1],
    which is the conventional definition for QAOA on Max-Cut and friends.
    """
    if worst is None:
        return energy / ground_state
    return (energy - worst) / (ground_state - worst)


def scaled_ratio(energy: float, e_opt: float, e_worst: float) -> float:
    """Scaled approximation ratio r = (E_worst - E) / (E_worst - E_opt) in [0, 1].

    Same convention used by `qaoa.approximation_ratio`. r = 1 at the true
    ground state, r = 0 at the worst basis state, and an unbiased uniform
    distribution gives (E_worst - mean) / (E_worst - E_opt). Stable when
    E_opt is small in magnitude relative to the Hamiltonian's spread —
    use this instead of E / E_opt for mixed-sign cost functions where the
    unscaled ratio explodes.

    Output is clipped to [0, 1] to suppress float-precision artefacts that
    would otherwise produce ratios like 1 + 1e-16 for exact-optimum solvers
    (brute force, SA at the optimum). A reader who sees r > 1 should not be
    led to think a solver "beat the optimum".
    """
    if e_worst == e_opt:
        return 1.0
    r = (e_worst - energy) / (e_worst - e_opt)
    return max(0.0, min(1.0, r))


def gap(energy: float, optimum: float) -> float:
    """Relative gap (E - E_opt) / |E_opt|. Smaller is better; 0 = optimal.

    Convention-agnostic alternative to approximation_ratio when the sign
    of the optimum is awkward (mixed-sign cost functions).
    """
    return (energy - optimum) / abs(optimum) if optimum != 0 else float("inf")


def prob_optimal(probs: np.ndarray, x_opt: np.ndarray) -> float:
    """Probability the optimiser samples the brute-force optimum.

    Conventions: qubit 0 is MSB, so the basis index of bitstring x is
    sum_i x[i] * 2^(n-1-i).
    """
    x_opt = np.asarray(x_opt, dtype=int)
    n = x_opt.size
    idx = int(sum(int(b) << (n - 1 - i) for i, b in enumerate(x_opt)))
    return float(probs[idx])


def prob_feasible(probs: np.ndarray, n: int, K: int) -> float:
    """Total probability mass on bitstrings with Hamming weight K."""
    mask = np.array([bin(k).count("1") == K for k in range(1 << n)])
    return float(probs[mask].sum())


def sharpe(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    """Annualised-style Sharpe ratio of the equal-weighted portfolio in x.

    Caller should pass annualised mu and Sigma if an annualised Sharpe is
    wanted; the function itself just computes mu_p / sigma_p.
    """
    x = np.asarray(x, dtype=float)
    if x.sum() == 0:
        return 0.0
    w = x / x.sum()
    mu_p = w @ mu
    sig_p = np.sqrt(w @ Sigma @ w)
    return float(mu_p / sig_p) if sig_p > 0 else 0.0


def hit_rate(samples: np.ndarray, x_opt: np.ndarray) -> float:
    """Fraction of bitstring samples that equal the optimum (Monte Carlo)."""
    x_opt = np.asarray(x_opt, dtype=int)
    return float(np.mean(np.all(samples == x_opt[None, :], axis=1)))
