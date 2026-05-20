"""
mip.py — Exact MIP baseline via PuLP (CBC solver).

This is the industry baseline for cardinality-constrained mean-variance
portfolio selection: a mixed-integer quadratic program with a hard
cardinality constraint. We solve the *same* QUBO that QAOA tackles —
no relaxation, no surrogate — using PuLP's bundled CBC backend, then
report cost, runtime, and the MIP optimality gap.

The variance term `x^T Sigma x` is quadratic in binary variables. We
linearise it by introducing one variable per pair via the standard
McCormick / Glover linearisation:

    z_ij = x_i x_j        x_i, x_j in {0,1}
    <=>
    z_ij <= x_i,  z_ij <= x_j,  z_ij >= x_i + x_j - 1,  z_ij >= 0

then x^T Sigma x = sum_i Sigma_ii x_i + 2 sum_{i<j} Sigma_ij z_ij.
The cardinality constraint sum x_i = K replaces the QUBO's soft
A (sum x - K)^2 penalty — feasibility is exact.

Same `SolverResult` shape as `scripts.classical` so the comparison
table in 02_classical / 09_compare gets a one-line addition.

PuLP / CBC are required; install via `pip install pulp` (which bundles
the CBC binary). If PuLP isn't on the path the import raises a clear
error rather than silently returning a junk result.
"""
from __future__ import annotations
from time import perf_counter
import numpy as np

from scripts.classical import SolverResult


def mip_exact(pf, time_limit: float | None = 60.0,
              msg: bool = False) -> SolverResult:
    """Solve the cardinality-constrained mean-variance MIP to (proven) optimality.

    Parameters
    ----------
    pf : PortfolioProblem
        Source of (mu, Sigma, lam, K). The QUBO's A (budget penalty) is
        ignored because the MIP imposes sum(x) == K as a hard constraint.
    time_limit : float or None
        Seconds before CBC bails. None means no limit. Default 60 s is
        plenty for n <= ~30 with K << n.
    msg : bool
        Forwarded to CBC's `msg` flag — silence the solver by default.

    Returns
    -------
    SolverResult with .cost == pf.cost(x), so the row is directly
    comparable to brute_force, greedy, SA, QAOA. Extra fields in
    .metadata: 'mip_status', 'mip_gap', 'time_limit_s'.
    """
    try:
        import pulp  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "scripts.mip requires PuLP — `pip install pulp` and retry. "
            "PuLP ships a CBC binary, so no separate solver install is needed."
        ) from e

    t0 = perf_counter()

    n = pf.n
    K = int(pf.K)
    mu = np.asarray(pf.mu, dtype=float)
    Sigma = np.asarray(pf.Sigma, dtype=float)
    lam = float(pf.lam)

    prob = pulp.LpProblem("portfolio_mip", pulp.LpMinimize)
    x = [pulp.LpVariable(f"x_{i}", cat=pulp.LpBinary) for i in range(n)]

    # z_ij = x_i * x_j for i < j (McCormick linearisation).
    z = {}
    for i in range(n):
        for j in range(i + 1, n):
            z_ij = pulp.LpVariable(f"z_{i}_{j}", lowBound=0, upBound=1)
            z[(i, j)] = z_ij
            prob += z_ij <= x[i]
            prob += z_ij <= x[j]
            prob += z_ij >= x[i] + x[j] - 1

    # Cardinality constraint (exact — the QUBO's A-penalty is unnecessary here).
    prob += pulp.lpSum(x) == K

    # Objective: -mu^T x + lam * (sum Sigma_ii x_i + 2 sum_{i<j} Sigma_ij z_ij).
    linear = pulp.lpSum(-mu[i] * x[i] for i in range(n))
    diag   = pulp.lpSum(lam * Sigma[i, i] * x[i] for i in range(n))
    off    = pulp.lpSum(2 * lam * Sigma[i, j] * z[(i, j)]
                        for i, j in z)
    prob += linear + diag + off

    solver = pulp.PULP_CBC_CMD(msg=msg,
                               timeLimit=time_limit if time_limit else None)
    status = prob.solve(solver)
    runtime = perf_counter() - t0

    x_val = np.array([int(round(pulp.value(xi))) for xi in x], dtype=int)
    status_name = pulp.LpStatus[status]
    # CBC reports a relative MIP gap via solver attributes when available;
    # falling back to NaN if not (e.g. on time-out).
    try:
        mip_gap = float(getattr(solver, "actualGap", float("nan")))
    except Exception:
        mip_gap = float("nan")

    return SolverResult(
        name="mip_cbc",
        x=x_val,
        cost=pf.cost(x_val),
        runtime=runtime,
        n_evals=1,  # MIP is one call; n_evals is bookkeeping only.
        feasible=bool(x_val.sum() == K),
        metadata={"mip_status": status_name,
                  "mip_gap":    mip_gap,
                  "time_limit_s": time_limit},
    )
