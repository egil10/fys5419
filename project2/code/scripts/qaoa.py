"""
qaoa.py — Pure-NumPy QAOA for the portfolio QUBO.

Functional API: every step is a free function. State is kept in plain
arrays; no class is required to run the pipeline.

Pipeline
--------
    problem            -> make_hamiltonians  (Q, h, J, const, H_C_diag)
    (gammas, betas)    -> qaoa_statevector   |psi> in 2^n
    psi                -> qaoa_energy        <psi|H_C|psi>
    psi                -> sample_probs       |psi|^2
    probs              -> decode_top_k       top bitstrings with C(x)

End-to-end convenience: `solve(problem, p, n_restarts)` wraps the lot
and uses `scripts.optimize.multi_start_minimize` (COBYLA + restarts).

Implementation notes
--------------------
- H_C is diagonal in the computational basis, so it is stored as a
  length-2^n array and applied by elementwise multiplication.
- exp(-i beta sum_i X_i) factorises into a product of single-qubit X
  rotations (X_i commute). Applied as (2,..,2)-tensor reshapes for
  O(n * 2^n) per layer — much cheaper than diagonalising H_M.
- Qubit 0 is the most significant bit of the basis index (matches the
  convention in `ising.py`).
"""
from __future__ import annotations
import numpy as np

from scripts.portfolio import PortfolioProblem, eval_cost
from scripts.ising     import from_portfolio


# ── Hamiltonian builders ──────────────────────────────────────────────────
def make_hamiltonians(problem: PortfolioProblem):
    """Return (Q, offset, h, J, const, H_C_diag) for `problem`.

    `H_C_diag[k]` is the *full* portfolio cost C(x_k) — i.e. the Ising
    energy plus the constant offset from the QUBO -> Ising substitution.
    This keeps QAOA's energy and ground-state values in the same units as
    `pf.cost(x)`, so `<psi|H_C|psi>` is directly comparable to classical
    solver costs and `approximation_ratio` is unit-correct.
    """
    Q, off, h, J, c, HC = from_portfolio(problem)
    return Q, off, h, J, c, HC + c


# ── Single-qubit X-mixer (factorised exp(-i beta sum X_i)) ───────────────
def _apply_X_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply exp(-i beta sum_i X_i) = prod_i exp(-i beta X_i) to |psi>.

    Each factor is a 2x2 rotation [[cos b, -i sin b], [-i sin b, cos b]].
    """
    cos = np.cos(beta)
    isin = -1j * np.sin(beta)
    psi = psi.reshape((2,) * n)
    for i in range(n):
        psi = np.moveaxis(psi, i, 0)
        new = np.empty_like(psi)
        new[0] = cos * psi[0] + isin * psi[1]
        new[1] = isin * psi[0] + cos * psi[1]
        psi = np.moveaxis(new, 0, i)
    return psi.reshape(-1)


# ── QAOA ansatz / energy ─────────────────────────────────────────────────
def qaoa_statevector(gammas, betas, H_C_diag: np.ndarray, n: int) -> np.ndarray:
    """Apply p QAOA layers to |+>^n.

    |psi> = prod_k exp(-i beta_k H_M) exp(-i gamma_k H_C) |+>^n.
    """
    dim = 1 << n
    psi = np.ones(dim, dtype=complex) / np.sqrt(dim)
    for gamma, beta in zip(gammas, betas):
        psi = np.exp(-1j * gamma * H_C_diag) * psi
        psi = _apply_X_mixer(psi, float(beta), n)
    return psi


def qaoa_energy(params: np.ndarray, H_C_diag: np.ndarray, n: int, p: int) -> float:
    """Objective for the classical outer loop: <psi|H_C|psi>."""
    gammas, betas = params[:p], params[p:]
    psi = qaoa_statevector(gammas, betas, H_C_diag, n)
    return float(np.real(psi.conj() @ (H_C_diag * psi)))


# ── Measurement ──────────────────────────────────────────────────────────
def sample_probs(psi: np.ndarray) -> np.ndarray:
    """|psi|^2 over the computational basis."""
    return np.abs(psi) ** 2


def decode_top_k(probs: np.ndarray, problem: PortfolioProblem,
                 k: int = 5) -> list[dict]:
    """Top-k most probable bitstrings, each with its portfolio cost C(x)."""
    n = problem.n
    ranked = np.argsort(probs)[::-1][:k]
    out = []
    for idx in ranked:
        x = np.array([(int(idx) >> (n - 1 - i)) & 1 for i in range(n)],
                     dtype=int)
        out.append({
            "bitstring": "".join(map(str, x)),
            "x":         x,
            "prob":      float(probs[idx]),
            "cost":      eval_cost(x, problem),
            "budget":    int(x.sum()),
        })
    return out


# ── Diagnostics ──────────────────────────────────────────────────────────
def ground_state_energy(H_C_diag: np.ndarray) -> float:
    """Exact lowest eigenvalue of H_C (diagonal -> just its min)."""
    return float(H_C_diag.min())


def approximation_ratio(energy: float, H_C_diag: np.ndarray) -> float:
    """Scaled QAOA approximation ratio in [0, 1].

        r = (E_worst - E) / (E_worst - E_opt),

    so r = 1 when QAOA reaches the ground state, r = 0 when it lands on
    the highest-cost state, and a uniform initial state gives the value
    (E_worst - mean) / (E_worst - E_opt). This is the standard
    convention (matches Farhi et al. on Max-Cut) and stays interpretable
    when E_opt is small in magnitude relative to the spread.
    """
    e_opt   = float(H_C_diag.min())
    e_worst = float(H_C_diag.max())
    if e_worst == e_opt:
        return 1.0
    return (e_worst - energy) / (e_worst - e_opt)


# ── End-to-end convenience ───────────────────────────────────────────────
def solve(problem: PortfolioProblem,
          p: int = 1,
          n_restarts: int = 10,
          seed: int = 42,
          method: str = "COBYLA",
          maxiter: int = 200,
          rhobeg: float = 0.1,
          verbose: bool = False) -> dict:
    """Build Hamiltonians, train QAOA from `n_restarts` random inits, return result.

    Defaults match the project methodology: 10 multi-start seeds,
    gamma_k ~ U[0, 2*pi], beta_k ~ U[0, pi], COBYLA with maxiter=200,
    rhobeg=0.1. Single-seed QAOA results are not meaningful — the
    landscape is non-convex enough that multi-start is mandatory.

    The returned dict has everything downstream notebooks need:
        p, energy, ratio, ground_state_energy, gammas, betas, psi, probs,
        history (full multi-start trajectory), runtime.
    """
    # Local import keeps qaoa.py importable even if optimize.py isn't on the path.
    from scripts.optimize import multi_start_minimize

    _, _, _, _, _, HC = make_hamiltonians(problem)
    n = problem.n

    objective = lambda params: qaoa_energy(params, HC, n, p)
    # gamma in [0, 2*pi] for the first p entries, beta in [0, pi] for the last p.
    # multi_start_minimize uses a single bounds tuple, so we sample uniformly in
    # [0, 2*pi] and rely on the X-mixer's 2*pi-periodicity in beta — equivalent
    # up to a global phase.
    opt = multi_start_minimize(
        objective, n_params=2 * p, n_restarts=n_restarts,
        method=method, seed=seed, verbose=verbose,
        maxiter=maxiter, rhobeg=rhobeg,
    )

    gammas, betas = opt.x[:p], opt.x[p:]
    psi   = qaoa_statevector(gammas, betas, HC, n)
    probs = sample_probs(psi)

    return {
        "p":                   p,
        "energy":              opt.fun,
        "ratio":               approximation_ratio(opt.fun, HC),  # scaled, in [0, 1]
        "ground_state_energy": ground_state_energy(HC),
        "worst_energy":        float(HC.max()),
        "gammas":              gammas,
        "betas":               betas,
        "psi":                 psi,
        "probs":               probs,
        "history":             opt.history,
        "runtime":             opt.runtime,
        "n_restarts":          opt.n_restarts,
    }


# ── Warm-started solve (Zhou et al. 2018) ────────────────────────────────
def solve_warm(problem: PortfolioProblem,
               p: int,
               prev: tuple[np.ndarray, np.ndarray] | None = None,
               n_restarts: int = 10,
               seed: int = 42,
               method: str = "COBYLA",
               maxiter: int = 200,
               rhobeg: float = 0.05,
               verbose: bool = False) -> dict:
    """QAOA with warm-start init from the previous-depth optimum.

    Implements the Zhou-Wang-Choi-Pichler-Lukin (2018) prescription:
    at depth p, seed the optimiser with the depth-(p-1) optimal angles
    extended by one fresh layer with small random angles. This routinely
    "fixes" non-monotonic depth sweeps that random-restart suffers from,
    so comparing warm-start vs random tells us whether non-monotonicity
    is a restart-budget pathology (fixed by warm starts) or a structural
    ansatz limit (warm starts don't help).

    The warm init is augmented with `n_restarts - 1` small random
    perturbations around it, then COBYLA picks the best — same restart
    budget across protocols.

    `prev=None` falls back to a pure random multi-start at this depth
    (so the p=1 entry of the sweep is well-defined).
    """
    from scripts.optimize import multi_start_minimize, OptimizeResult

    _, _, _, _, _, HC = make_hamiltonians(problem)
    n = problem.n
    objective = lambda params: qaoa_energy(params, HC, n, p)
    rng = np.random.default_rng(seed)

    if prev is None or p == 1:
        # Bootstrap: no prior depth to copy from. Use a random multi-start
        # with the same restart budget so this call is comparable.
        opt = multi_start_minimize(
            objective, n_params=2 * p, n_restarts=n_restarts,
            method=method, seed=seed, verbose=verbose,
            maxiter=maxiter, rhobeg=rhobeg,
        )
    else:
        from scipy.optimize import minimize
        prev_g, prev_b = prev
        if len(prev_g) != p - 1 or len(prev_b) != p - 1:
            raise ValueError(
                f"`prev` has {len(prev_g)}+{len(prev_b)} angles but expected "
                f"{(p - 1) * 2} for warm-starting depth {p}."
            )

        # Anchor init: previous gammas/betas + one fresh small random layer.
        def _anchor():
            g_new = rng.uniform(0.0, 0.2)        # small fresh-layer angles
            b_new = rng.uniform(0.0, 0.2)
            return np.concatenate([prev_g, [g_new], prev_b, [b_new]])

        # Run n_restarts: first the noise-free anchor, then perturbations of it.
        from time import perf_counter
        t0 = perf_counter()
        history = []
        best_fun = np.inf
        best_x: np.ndarray = _anchor()  # initialised; overwritten on first restart
        for r in range(n_restarts):
            if r == 0:
                x0 = _anchor()
            else:
                jitter = rng.normal(0.0, 0.05, size=2 * p)
                x0 = _anchor() + jitter
            res = minimize(objective, x0, method=method,
                           options={"maxiter": maxiter, "rhobeg": rhobeg})
            history.append({"restart": r, "x0": x0, "fun": float(res.fun),
                            "x": np.asarray(res.x),
                            "nfev": getattr(res, "nfev", None)})
            if verbose:
                print(f"  warm restart {r+1:2d}: f = {float(res.fun):.6f}")
            if float(res.fun) < best_fun:
                best_fun = float(res.fun)
                best_x   = np.asarray(res.x)
        opt = OptimizeResult(fun=best_fun, x=best_x, n_restarts=n_restarts,
                             runtime=perf_counter() - t0, history=history)

    gammas, betas = opt.x[:p], opt.x[p:]
    psi   = qaoa_statevector(gammas, betas, HC, n)
    probs = sample_probs(psi)

    return {
        "p":                   p,
        "energy":              opt.fun,
        "ratio":               approximation_ratio(opt.fun, HC),
        "ground_state_energy": ground_state_energy(HC),
        "worst_energy":        float(HC.max()),
        "gammas":              gammas,
        "betas":               betas,
        "psi":                 psi,
        "probs":               probs,
        "history":             opt.history,
        "runtime":             opt.runtime,
        "n_restarts":          opt.n_restarts,
        "protocol":            "warm_start",
    }


# ── XY-ring mixer + Dicke-state QAOA ─────────────────────────────────────
def qaoa_xy_statevector(gammas, betas, H_C_diag, H_M_xy, dicke_psi, n):
    """Apply p layers of |psi> -> exp(-i beta H_M^XY) exp(-i gamma H_C) |psi>.

    Starts from the Dicke state |D^n_K> instead of |+>^n. Because the XY
    ring mixer commutes with N = sum_i x_i, the resulting state stays in
    the weight-K subspace throughout the dynamics — exact budget
    feasibility, no soft penalty needed.
    """
    from scripts.xy_mixer import apply_xy_mixer

    psi = dicke_psi.astype(complex).copy()
    for gamma, beta in zip(gammas, betas):
        psi = np.exp(-1j * gamma * H_C_diag) * psi
        psi = apply_xy_mixer(psi, float(beta), H_M_xy)
    return psi


def qaoa_xy_energy(params, H_C_diag, H_M_xy, dicke_psi, n, p) -> float:
    """Objective for the XY-mixer outer loop: <psi|H_C|psi>."""
    gammas, betas = params[:p], params[p:]
    psi = qaoa_xy_statevector(gammas, betas, H_C_diag, H_M_xy, dicke_psi, n)
    return float(np.real(psi.conj() @ (H_C_diag * psi)))


def solve_xy(problem: PortfolioProblem,
             p: int = 1,
             n_restarts: int = 10,
             seed: int = 42,
             method: str = "COBYLA",
             maxiter: int = 200,
             rhobeg: float = 0.1,
             verbose: bool = False) -> dict:
    """XY-ring + Dicke QAOA — the constraint-preserving variant.

    Initial state: |D^n_K> (Dicke state).
    Mixer: H_M^{XY} = (1/2) sum_{<i,j>} (X_i X_j + Y_i Y_j) on a ring.

    The constraint sum(x) = K is preserved exactly: P(feasible) = 1
    identically, all returned bitstrings have Hamming weight K. The
    portfolio penalty A is effectively irrelevant for this solver — we
    still evaluate the QUBO cost (so the comparison vs the standard
    X-mixer is apples-to-apples) but the optimiser only navigates
    *between* feasible bitstrings.

    Returns the same dict shape as `solve` plus 'protocol' = 'xy_dicke',
    so downstream notebooks can stack records from both protocols and
    plot them together.
    """
    from scripts.optimize import multi_start_minimize
    from scripts.xy_mixer import build_HM_xy
    from scripts.dicke     import dicke_state

    _, _, _, _, _, HC = make_hamiltonians(problem)
    n = problem.n
    H_M_xy   = build_HM_xy(n, periodic=True)
    dicke_psi = dicke_state(n, problem.K)

    objective = lambda params: qaoa_xy_energy(params, HC, H_M_xy, dicke_psi, n, p)
    opt = multi_start_minimize(
        objective, n_params=2 * p, n_restarts=n_restarts,
        method=method, seed=seed, verbose=verbose,
        maxiter=maxiter, rhobeg=rhobeg,
    )

    gammas, betas = opt.x[:p], opt.x[p:]
    psi   = qaoa_xy_statevector(gammas, betas, HC, H_M_xy, dicke_psi, n)
    probs = sample_probs(psi)

    return {
        "p":                   p,
        "energy":              opt.fun,
        "ratio":               approximation_ratio(opt.fun, HC),
        "ground_state_energy": ground_state_energy(HC),
        "worst_energy":        float(HC.max()),
        "gammas":              gammas,
        "betas":               betas,
        "psi":                 psi,
        "probs":               probs,
        "history":             opt.history,
        "runtime":             opt.runtime,
        "n_restarts":          opt.n_restarts,
        "protocol":            "xy_dicke",
    }
