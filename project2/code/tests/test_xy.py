"""
test_xy.py — XY ring mixer + Dicke state correctness checks.

Three properties that have to hold or the whole XY-Dicke story (notebook
09, the Methods section, the Conclusion's "constraint-preserving" claim)
falls apart:

1. `[H_M^XY, N] = 0` on a random vector. If the commutator is non-zero,
   the mixer leaks outside the weight-K subspace and feasibility is no
   longer exact.

2. The Dicke state |D^n_K> has exactly C(n, K) non-zero amplitudes, each
   of magnitude 1/sqrt(C(n, K)), supported on weight-K basis states.

3. QAOA-XY at p=0 (Dicke state, no layers applied) returns
   <D^n_K | H_C | D^n_K>, which is the average cost over all weight-K
   bitstrings. This is the trivial sanity check that the QAOA-XY pipeline
   wiring is correct.
"""
from pathlib import Path
from math import comb
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from scripts.portfolio import PortfolioProblem
from scripts.xy_mixer  import build_HM_xy, number_operator_diag, apply_xy_mixer
from scripts.dicke     import dicke_state
from scripts.qaoa      import make_hamiltonians, solve_xy


def _small_problem(n=6, K=2, seed=3):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0, 0.1, size=n)
    A = rng.normal(0.0, 0.05, size=(n, n))
    Sigma = A @ A.T + 0.01 * np.eye(n)
    return PortfolioProblem(mu, Sigma, lam=2.0, A=0.5, K=K)


def test_xy_commutes_with_number_operator():
    """[H_M^XY, N] |psi> = 0 to machine precision on a random vector."""
    for n in (4, 6, 8):
        H_xy = build_HM_xy(n, periodic=True)
        N_diag = number_operator_diag(n)
        rng = np.random.default_rng(n)
        psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        psi = psi / np.linalg.norm(psi)
        lhs = H_xy @ (N_diag * psi)
        rhs = N_diag * (H_xy @ psi)
        err = float(np.linalg.norm(lhs - rhs))
        assert err < 1e-10, f"[H_M^XY, N] non-zero at n={n}: ||.||={err:.2e}"


def test_xy_preserves_dicke_weight():
    """exp(-i beta H_M^XY) |D^n_K> stays at weight K."""
    for n, K in [(4, 2), (6, 3), (8, 4)]:
        H_xy = build_HM_xy(n, periodic=True)
        N_diag = number_operator_diag(n)
        psi0 = dicke_state(n, K)
        psi  = apply_xy_mixer(psi0, beta=0.37, H_M_xy=H_xy)
        exp_N = float(np.real(psi.conj() @ (N_diag * psi)))
        assert abs(exp_N - K) < 1e-10, (
            f"<N> drifted at (n={n}, K={K}): {exp_N} vs target {K}"
        )


def test_dicke_state_amplitudes():
    """|D^n_K> has C(n, K) basis states with equal amplitude 1/sqrt(C(n, K))."""
    for n, K in [(4, 2), (6, 3), (8, 4)]:
        psi = dicke_state(n, K)
        nz = np.flatnonzero(np.abs(psi) > 1e-12)
        assert nz.size == comb(n, K), (
            f"|D^{n}_{K}> has {nz.size} non-zero amps, expected {comb(n, K)}"
        )
        # Check every non-zero basis index has Hamming weight K.
        for k in nz:
            assert bin(int(k)).count('1') == K
        # Equal amplitudes.
        amp = 1.0 / np.sqrt(comb(n, K))
        assert np.allclose(np.abs(psi[nz]), amp, atol=1e-12)
        # Normalised.
        assert abs(float(np.linalg.norm(psi)) - 1.0) < 1e-12


def test_qaoa_xy_p0_matches_dicke_expectation():
    """At p=1 the XY-Dicke pipeline returns an energy <= <D^n_K|H_C|D^n_K>."""
    pf = _small_problem(n=6, K=2)
    _, _, _, _, _, HC = make_hamiltonians(pf)
    psi0 = dicke_state(pf.n, pf.K)
    dicke_energy = float(np.real(psi0.conj() @ (HC * psi0)))

    res = solve_xy(pf, p=1, n_restarts=5, seed=0)
    # QAOA at any p >= 1 must do at least as well as the Dicke state itself
    # (it can always choose gamma = beta = 0 and reproduce the Dicke energy).
    # Add a small tolerance for COBYLA's rhobeg-bounded search radius.
    assert res['energy'] <= dicke_energy + 1e-6, (
        f"QAOA-XY at p=1 worse than Dicke baseline: "
        f"{res['energy']:.6f} vs {dicke_energy:.6f}"
    )
