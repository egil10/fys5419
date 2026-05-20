"""
analysis.py — QAOA diagnostics: parameter landscape and statistical physics.

Two classes:

    Landscape       2D scan of QAOA energy E(gamma, beta) at p=1, with
                    optional overlay of optimisation results.

    Thermodynamics  Classical spin-glass picture: thermal energy <H>_beta
                    as a function of inverse temperature, and the histogram
                    of Ising energies across all 2^n configurations. QAOA
                    energies overlay as horizontal lines (the zero-T limit).

Both wrap the new functional QAOA API (`scripts.qaoa.make_hamiltonians`,
`scripts.qaoa.qaoa_energy`). No state besides the cached Hamiltonian and
the scan grid.

Usage
-----
    from scripts.portfolio import PortfolioProblem
    from scripts.qaoa      import solve
    from scripts.analysis  import Landscape, Thermodynamics

    pf = PortfolioProblem(mu, Sigma, lam=2.0, A=0.5, K=2, tickers=tickers)
    results = {p: solve(pf, p=p, n_restarts=10) for p in [1, 2, 3]}

    Landscape(pf).scan().plot(results[1], save=True, name="mag7")
    Thermodynamics(pf).plot(results, save=True, name="mag7")
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from scripts.portfolio import PortfolioProblem
from scripts.qaoa      import make_hamiltonians, qaoa_energy
from scripts.plotting  import PALETTE, apply_style, title, rel_path, out_dir


# ════════════════════════════════════════════════════════════════════════
# Landscape — 2D scan over (gamma, beta) at p=1
# ════════════════════════════════════════════════════════════════════════
class Landscape:
    def __init__(self, problem: PortfolioProblem, n_pts: int = 40):
        self.pf = problem
        self.n_pts = n_pts
        self.gammas = np.linspace(0, 2 * np.pi, n_pts)
        self.betas  = np.linspace(0, np.pi, n_pts)
        _, _, _, _, _, self._HC = make_hamiltonians(problem)
        self.energy: np.ndarray | None = None

    def scan(self):
        n = self.pf.n
        E = np.empty((self.n_pts, self.n_pts))
        for i, g in enumerate(self.gammas):
            for j, b in enumerate(self.betas):
                E[i, j] = qaoa_energy(np.array([g, b]), self._HC, n, p=1)
        self.energy = E
        return self

    def plot(self, result_p1: dict | None = None,
             save: bool = False, name: str = "landscape"):
        if self.energy is None:
            raise RuntimeError("Call .scan() before .plot()")

        apply_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f"QAOA energy landscape — {name}",
                     fontweight="bold", fontsize=13, color=PALETTE["charcoal"],
                     x=0.02, ha="left", y=0.995)

        GG, BB = np.meshgrid(self.gammas, self.betas, indexing="ij")
        cf = ax.contourf(GG / np.pi, BB / np.pi, self.energy,
                         levels=25, cmap="viridis")
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(r"$E(\gamma, \beta) = \langle\psi|H_C|\psi\rangle$",
                       fontsize=10)

        if result_p1 is not None:
            g_opt = float(result_p1["gammas"][0]) % (2 * np.pi)
            b_opt = float(result_p1["betas"][0])
            ax.scatter(g_opt / np.pi, b_opt / np.pi,
                       c=PALETTE["red"], s=220, marker="*",
                       edgecolors=PALETTE["charcoal"], linewidth=1.2,
                       zorder=5,
                       label=f"Optimum E={result_p1['energy']:.4f}")
            ax.legend(fontsize=10)

        ax.set_xlabel(r"$\gamma / \pi$", fontsize=12)
        ax.set_ylabel(r"$\beta / \pi$", fontsize=12)
        title(ax, "Energy landscape (p=1)",
              r"$E(\gamma, \beta) = \langle\psi|H_C|\psi\rangle$ over the two variational angles")
        ax.grid(False)

        plt.tight_layout()
        if save:
            path = out_dir("plots", "analysis") / f"{name}_landscape.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"saved -> {rel_path(path)}")
        plt.show()


# ════════════════════════════════════════════════════════════════════════
# Thermodynamics — spin-glass picture
# ════════════════════════════════════════════════════════════════════════
class Thermodynamics:
    def __init__(self, problem: PortfolioProblem):
        self.pf = problem
        _, _, _, _, _, HC = make_hamiltonians(problem)
        self._E_all = HC.real
        self._E0 = float(self._E_all.min())

    def thermal_energy(self, beta_th) -> np.ndarray:
        """Vectorised <H>_beta over an array of inverse temperatures."""
        beta_th = np.atleast_1d(beta_th)
        out = np.empty_like(beta_th, dtype=float)
        for i, b in enumerate(beta_th):
            shifted = -b * (self._E_all - self._E0)
            w = np.exp(shifted)
            Z = w.sum()
            out[i] = self._E0 + np.dot(self._E_all - self._E0, w) / Z
        return out

    def plot(self, qaoa_results: dict[int, dict] | None = None,
             save: bool = False, name: str = "thermodynamics"):
        apply_style()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Statistical physics connection — {name}",
                     fontweight="bold", fontsize=13, color=PALETTE["charcoal"],
                     x=0.02, ha="left", y=0.995)

        # [0] Thermal energy vs beta_th
        ax = axes[0]
        beta_vals = np.logspace(-2, 3, 200)
        therm = self.thermal_energy(beta_vals)
        ax.semilogx(beta_vals, therm, color=PALETTE["blue"], lw=2.2,
                    label=r"$\langle H\rangle_{\beta}$")
        ax.axhline(self._E0, color=PALETTE["red"], ls="--", lw=1.6,
                   label=f"$E_0 = {self._E0:.4f}$")
        ax.axhline(therm[0], color=PALETTE["grey"], ls=":", lw=1.4,
                   label=f"High-T avg = {therm[0]:.4f}")

        if qaoa_results is not None:
            qaoa_colors = [PALETTE["teal"], PALETTE["ochre"], PALETTE["purple"],
                           PALETTE["coral"], PALETTE["navy"]]
            for k, (p, res) in enumerate(sorted(qaoa_results.items())):
                ax.axhline(res["energy"],
                           color=qaoa_colors[k % len(qaoa_colors)],
                           ls=(0, (4, 1.5)), lw=1.6,
                           label=f"QAOA p={p}: E={res['energy']:.4f}")

        ax.set_xlabel(r"Inverse temperature $\beta_{\rm th}$", fontsize=11)
        ax.set_ylabel(r"$\langle H\rangle_\beta$", fontsize=11)
        title(ax, "Thermal energy vs temperature",
              r"$\langle H\rangle_\beta \to E_0$ as $\beta_{\rm th} \to \infty$")
        ax.legend(fontsize=8, loc="upper right")

        # [1] Histogram of Ising energies
        ax = axes[1]
        n_bins = min(max(int(np.sqrt(len(self._E_all))), 8), 30)
        ax.hist(self._E_all, bins=n_bins, color=PALETTE["blue_muted"],
                edgecolor=PALETTE["charcoal"], linewidth=0.5, alpha=0.85)
        ax.axvline(self._E0, color=PALETTE["red"], ls="--", lw=1.8,
                   label=f"$E_0 = {self._E0:.4f}$")
        ax.set_xlabel(r"Ising energy $H(z)$", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        title(ax, "Ising energy distribution",
              rf"Histogram of $H(z)$ over all $2^n = {len(self._E_all)}$ bitstrings")
        ax.legend(fontsize=9)

        plt.tight_layout()
        if save:
            path = out_dir("plots", "analysis") / f"{name}_thermodynamics.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"saved -> {rel_path(path)}")
        plt.show()
