"""
optimize.py — Classical outer-loop wrappers for QAOA training.

Single entry point: `multi_start_minimize`, which runs a SciPy minimiser
from many random initialisations and returns the best result plus the
full trajectory. QAOA's parameter landscape is non-convex enough that
single-start results are not really publishable; multi-start is mandatory.

Method shortcuts: "COBYLA" (gradient-free, default) and "SPSA" (a tiny
SPSA implementation, useful when COBYLA stalls or for noisy objectives).
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable
import numpy as np
from scipy.optimize import minimize


@dataclass
class _SimpleResult:
    """Lightweight scipy.optimize.OptimizeResult stand-in for SPSA."""
    x:    np.ndarray
    fun:  float
    nfev: int


@dataclass
class OptimizeResult:
    fun:        float                      # best objective value
    x:          np.ndarray                 # best parameters
    n_restarts: int
    runtime:    float                      # total seconds
    history:    list = field(default_factory=list)
    # history entries: dict(restart, x0, fun, x, nfev)


def multi_start_minimize(
    objective: Callable[[np.ndarray], float],
    n_params: int,
    n_restarts: int = 10,
    method: str = "COBYLA",
    bounds: tuple[float, float] = (0.0, 2 * np.pi),
    seed: int = 42,
    maxiter: int = 2000,
    rhobeg: float = 0.5,
    verbose: bool = False,
    parallel: bool = True,
    n_jobs: int | None = None,
) -> OptimizeResult:
    """Run `method` from `n_restarts` random inits in [bounds[0], bounds[1]]^n_params.

    Restarts run in parallel via a `ThreadPoolExecutor` (default).
    `scipy.optimize.minimize` releases the GIL inside its C-level loop and
    `qaoa_energy` spends most of its time in NumPy ops (also GIL-releasing),
    so threads give a real wall-clock speed-up — roughly 1.8x on a 2-core
    Colab CPU, 4-6x on an 8-core laptop. Pass `parallel=False` for the
    sequential path (e.g. when debugging with `verbose=True`).
    """
    rng = np.random.default_rng(seed)
    lo, hi = bounds
    t0 = perf_counter()

    x0s = [rng.uniform(lo, hi, n_params) for _ in range(n_restarts)]

    def _run_one(args):
        r, x0 = args
        if method.upper() == "SPSA":
            res = _spsa(objective, x0, maxiter=maxiter, seed=seed + r)
        else:
            res = minimize(
                objective, x0, method=method,
                options={"maxiter": maxiter, "rhobeg": rhobeg},
            )
        return r, x0, float(res.fun), np.asarray(res.x), getattr(res, "nfev", None)

    if parallel and n_restarts > 1:
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_run_one, enumerate(x0s)))
    else:
        results = [_run_one((i, x0)) for i, x0 in enumerate(x0s)]

    # Sort by restart index so the trajectory stays reproducible across runs.
    results.sort(key=lambda r: r[0])

    best_fun = np.inf
    best_x: np.ndarray = np.zeros(n_params)
    history = []
    for r, x0, f_val, x, nfev in results:
        history.append({
            "restart": r, "x0": x0, "fun": f_val, "x": x, "nfev": nfev,
        })
        if verbose:
            print(f"  restart {r+1:2d}: f = {f_val:.6f}")
        if f_val < best_fun:
            best_fun = f_val
            best_x = x

    return OptimizeResult(
        fun=best_fun,
        x=best_x,
        n_restarts=n_restarts,
        runtime=perf_counter() - t0,
        history=history,
    )


# ── Minimal SPSA implementation ──────────────────────────────────────────
def _spsa(objective, x0, maxiter=200, a=0.1, c=0.1, alpha=0.602, gamma=0.101,
          seed=0):
    """Simultaneous Perturbation Stochastic Approximation.

    Useful for noisy objectives (e.g. shot-based QAOA energy estimates),
    where COBYLA's deterministic line search is fragile. Returns an object
    with .x, .fun, .nfev to match the scipy.optimize.OptimizeResult API.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x0, dtype=float).copy()
    n = x.size
    A = 0.10 * maxiter
    best_f = objective(x)
    best_x = x.copy()
    nfev = 1
    for k in range(1, maxiter + 1):
        ak = a / (k + A) ** alpha
        ck = c / k ** gamma
        delta = rng.choice([-1.0, 1.0], size=n)
        f_plus  = objective(x + ck * delta); nfev += 1
        f_minus = objective(x - ck * delta); nfev += 1
        g = (f_plus - f_minus) / (2 * ck) * (1.0 / delta)
        x = x - ak * g
        f = objective(x); nfev += 1
        if f < best_f:
            best_f, best_x = f, x.copy()

    return _SimpleResult(x=best_x, fun=float(best_f), nfev=nfev)
