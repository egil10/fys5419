"""
qaoa.py — QAOA portfolio optimiser.

Built on the lecture notebook by Morten Hjorth-Jensen (FYS5419, Spring 2026).

Pipeline:
    portfolio (mu, Sigma, lam, A, K)
        → QUBO         (Q matrix)
        → Ising        (h, J, c)
        → Hamiltonian  (H_C diagonal, H_M dense)
        → QAOA ansatz  (statevector simulation)
        → COBYLA optimisation with random restarts
        → measurement probabilities

Usage
-----
    from scripts.snp import SNP
    from scripts.portfolio import Portfolio
    from scripts.qaoa import QAOA

    snp = SNP(["AAPL", "MSFT", "AMZN", "GOOG"], "2020-01-01", "2023-12-31").cached_fetch()
    pf = Portfolio(snp.mu, snp.Sigma, lam=2.0, A=0.5, K=2, tickers=snp.tickers)
    qaoa = QAOA(pf)

    result = qaoa.optimise(p=2, n_restarts=15)
    print(qaoa.decode(result["probs"], top_k=5))
"""
import numpy as np
from scipy.optimize import minimize
from itertools import product

# ── Single-qubit Pauli matrices ───────────────────────────────────────
_I2 = np.eye(2, dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_X = np.array([[0, 1], [1,  0]], dtype=complex)


def _pauli_kron(P, qubit, n_qubits):
    """Embed single-qubit gate P on `qubit` in n-qubit tensor space (qubit 0 = MSB)."""
    ops = [_I2] * n_qubits
    ops[qubit] = P
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


class QAOA:
    """
    Quantum Approximate Optimisation Algorithm for mean-variance portfolios.

    Parameters
    ----------
    portfolio : Portfolio
        Portfolio object exposing mu, Sigma, lam, A, K, n, and optionally tickers.
    seed : int, optional
        Random seed for restart initialisation. Defaults to 42.
    """

    def __init__(self, portfolio, seed=42):
        self.pf = portfolio
        self.n = portfolio.n
        self.dim = 2 ** self.n
        self.rng = np.random.default_rng(seed)

        # Build Q, then (h, J, c), then H_C diagonal and H_M
        self._Q, self._offset = self._build_qubo()
        self.h, self.J, self.c = self._qubo_to_ising()
        self._HC_diag = self._build_HC_diag()
        self._HM_eigvals, self._HM_eigvecs = self._diagonalise_HM()
        self._psi0 = np.ones(self.dim, dtype=complex) / np.sqrt(self.dim)

    # ── QUBO and Ising construction ───────────────────────────────────
    def _build_qubo(self):
        """Build QUBO matrix (upper triangular) and constant offset."""
        pf = self.pf
        n = self.n
        Q = np.zeros((n, n))
        for i in range(n):
            Q[i, i] = -pf.mu[i] + pf.lam * pf.Sigma[i, i] + pf.A * (1 - 2 * pf.K)
            for j in range(i + 1, n):
                Q[i, j] = 2 * pf.lam * pf.Sigma[i, j] + 2 * pf.A
        return Q, pf.A * pf.K ** 2

    def _qubo_to_ising(self):
        """Convert QUBO to Ising coefficients using x_i = (1 - z_i)/2."""
        Q, c = self._Q, self._offset
        n = self.n
        h = np.zeros(n)
        J = np.zeros((n, n))
        for i in range(n):
            c -= -Q[i, i] / 2
            c += Q[i, i] / 2
            h[i] -= Q[i, i] / 2
        # diagonal contribution to constant
        c = self._offset + np.sum(np.diag(Q)) / 2
        h = -np.diag(Q) / 2
        # off-diagonal contribution
        for i in range(n):
            for j in range(i + 1, n):
                c += Q[i, j] / 4
                h[i] -= Q[i, j] / 4
                h[j] -= Q[i, j] / 4
                J[i, j] = Q[i, j] / 4
        return h, J, c

    # ── Hamiltonian construction ──────────────────────────────────────
    def _build_HC_diag(self):
        """Diagonal of H_C = Σ h_i Z_i + Σ J_ij Z_i Z_j (kept as 1D array)."""
        n, dim = self.n, self.dim
        # bit i of basis index k → spin z_i = +1 if bit=0, -1 if bit=1
        # (qubit 0 = MSB by our convention)
        spins = np.empty((dim, n))
        for k in range(dim):
            for i in range(n):
                spins[k, i] = 1 - 2 * ((k >> (n - 1 - i)) & 1)
        diag = spins @ self.h
        for i in range(n):
            for j in range(i + 1, n):
                if abs(self.J[i, j]) > 1e-14:
                    diag += self.J[i, j] * spins[:, i] * spins[:, j]
        return diag

    def _diagonalise_HM(self):
        """Build H_M = Σ X_i and return its eigendecomposition (cached for speed)."""
        HM = np.zeros((self.dim, self.dim), dtype=complex)
        for i in range(self.n):
            HM += _pauli_kron(_X, i, self.n)
        eigvals, eigvecs = np.linalg.eigh(HM)
        return eigvals, eigvecs

    # ── QAOA circuit ──────────────────────────────────────────────────
    def statevector(self, gammas, betas):
        """
        Apply p QAOA layers to |+>^n.

        |ψ> = Π_k exp(-i β_k H_M) exp(-i γ_k H_C) |+>^n

        H_C action is elementwise (diagonal); H_M action uses the cached
        eigendecomposition for O(dim^2) per layer instead of O(dim^3).
        """
        psi = self._psi0.copy()
        for gamma, beta in zip(gammas, betas):
            psi = np.exp(-1j * gamma * self._HC_diag) * psi
            psi = self._HM_eigvecs @ (
                np.exp(-1j * beta * self._HM_eigvals)
                * (self._HM_eigvecs.conj().T @ psi)
            )
        return psi

    def energy(self, params, p):
        """E(γ,β) = <ψ|H_C|ψ> for the QAOA ansatz at depth p."""
        gammas, betas = params[:p], params[p:]
        psi = self.statevector(gammas, betas)
        return float(np.real(psi.conj() @ (self._HC_diag * psi)))

    # ── Optimisation ──────────────────────────────────────────────────
    def optimise(self, p=1, n_restarts=15, verbose=False):
        """
        Find optimal QAOA parameters via COBYLA with random restarts.

        Parameters
        ----------
        p : int
            Circuit depth (number of QAOA layers).
        n_restarts : int
            Number of independent random initialisations.
        verbose : bool
            Print energy of each restart if True.

        Returns
        -------
        dict with keys: 'energy', 'gammas', 'betas', 'psi', 'probs', 'p'.
        """
        best_energy = np.inf
        best_x = None

        for r in range(n_restarts):
            x0 = self.rng.uniform(0, 2 * np.pi, 2 * p)
            res = minimize(
                lambda params: self.energy(params, p),
                x0, method="COBYLA",
                options={"maxiter": 2000, "rhobeg": 0.5},
            )
            if verbose:
                print(f"  restart {r+1:2d}: E = {res.fun:.6f}")
            if res.fun < best_energy:
                best_energy = res.fun
                best_x = res.x

        gammas, betas = best_x[:p], best_x[p:]
        psi = self.statevector(gammas, betas)
        return {
            "energy": best_energy,
            "gammas": gammas,
            "betas":  betas,
            "psi":    psi,
            "probs":  np.abs(psi) ** 2,
            "p":      p,
        }

    # ── Decoding ──────────────────────────────────────────────────────
    def decode(self, probs, top_k=5):
        """
        Return the top-k most probable bitstrings with their portfolio costs.

        Returns
        -------
        list of dicts, each with: bitstring, x, prob, E_ising, C_finance, budget.
        """
        ranked = np.argsort(probs)[::-1][:top_k]
        out = []
        for idx in ranked:
            x = np.array([(idx >> (self.n - 1 - i)) & 1 for i in range(self.n)])
            z = 1 - 2 * x
            E_ising = float(self.h @ z)
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    E_ising += self.J[i, j] * z[i] * z[j]
            out.append({
                "bitstring": "".join(map(str, x)),
                "x":         x,
                "prob":      float(probs[idx]),
                "E_ising":   E_ising,
                "C_finance": E_ising + self.c,
                "budget":    int(x.sum()),
            })
        return out

    # ── Diagnostics ───────────────────────────────────────────────────
    def ground_state_energy(self):
        """Exact lowest eigenvalue of H_C (since H_C is diagonal, just its min)."""
        return float(self._HC_diag.min())

    def approximation_ratio(self, energy):
        """E / E_0. Equals 1 if QAOA reaches the ground state."""
        return energy / self.ground_state_energy()

    def __repr__(self):
        return f"QAOA(n={self.n}, K={self.pf.K}, lam={self.pf.lam}, A={self.pf.A})"