"""
analysis.py — QAOA diagnostics: parameter landscape and statistical physics.

Two classes:

    Landscape       -- 2D scan of QAOA energy E(γ, β) at p=1, with optional
                       overlay of optimization restarts. Shows the rugged
                       variational surface and where the optimizer lands.

    Thermodynamics  -- Connection to the classical spin-glass picture:
                       thermal energy ⟨H⟩_β as a function of inverse
                       temperature, and the histogram of Ising energies
                       across all 2^n configurations. QAOA energies overlay
                       as horizontal lines, showing the zero-T limit.

Usage
-----
    from scripts.portfolio import Portfolio
    from scripts.qaoa      import QAOA
    from scripts.analysis  import Landscape, Thermodynamics

    pf = Portfolio(mu, Sigma, lam=2.0, A=0.5, K=2, tickers=tickers)
    qaoa = QAOA(pf, seed=42)

    # Run optimisation at multiple depths first
    results = {p: qaoa.optimise(p=p, n_restarts=15) for p in [1, 2, 3]}

    # Diagnostics
    Landscape(qaoa).scan().plot(results[1], save=True, name="mag7")
    Thermodynamics(qaoa).plot(results, save=True, name="mag7")
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from scripts.snp import PALETTE, _apply_style, _rel, title

_ROOT = Path(__file__).resolve().parent.parent
_PLOTS_DIR = _ROOT / "plots" / "analysis"


# ════════════════════════════════════════════════════════════════════════
# Landscape — 2D scan over (γ, β) at p=1
# ════════════════════════════════════════════════════════════════════════
class Landscape:
    """
    Scan QAOA energy E(γ, β) on a 2D grid at p=1.

    Parameters
    ----------
    qaoa : QAOA
        A QAOA instance built from a Portfolio.
    n_pts : int
        Grid resolution per axis (default 40).
    """

    def __init__(self, qaoa, n_pts=40):
        self.qaoa = qaoa
        self.n_pts = n_pts
        self.gammas = np.linspace(0, 2 * np.pi, n_pts)
        self.betas = np.linspace(0, np.pi / 2, n_pts)
        self.energy = None  # filled by scan()

    def scan(self):
        """Compute E(γ, β) on the 2D grid. Returns self for chaining."""
        E = np.empty((self.n_pts, self.n_pts))
        for i, g in enumerate(self.gammas):
            for j, b in enumerate(self.betas):
                E[i, j] = self.qaoa.energy(np.array([g, b]), p=1)
        self.energy = E
        return self

    def plot(self, result_p1=None, save=False, name="landscape"):
        """
        Plot the 2D landscape.

        Parameters
        ----------
        result_p1 : dict, optional
            A QAOA p=1 optimisation result; the (γ*, β*) found will be
            marked on the heatmap.
        save : bool
        name : str
            Filename prefix (e.g. basket name). Saves to plots/analysis/.
        """
        if self.energy is None:
            raise RuntimeError("Call .scan() before .plot()")

        _apply_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        fig.suptitle(f"QAOA energy landscape — {name}",
                     fontweight="bold", fontsize=13, color=PALETTE["charcoal"],
                     x=0.02, ha="left", y=0.995)

        # Filled contour plot of the energy surface
        GG, BB = np.meshgrid(self.gammas, self.betas, indexing="ij")
        cf = ax.contourf(GG / np.pi, BB / np.pi, self.energy,
                         levels=25, cmap="viridis")
        cbar = plt.colorbar(cf, ax=ax)
        cbar.set_label(r"$E(\gamma, \beta) = \langle\psi|H_C|\psi\rangle$",
                       fontsize=10)

        # Mark the optimum (if provided)
        if result_p1 is not None:
            g_opt = result_p1["gammas"][0]
            b_opt = result_p1["betas"][0]
            # Wrap γ to [0, 2π) for display
            g_opt = g_opt % (2 * np.pi)
            ax.scatter(g_opt / np.pi, b_opt / np.pi,
                       c=PALETTE["red"], s=220, marker="*",
                       edgecolors=PALETTE["charcoal"], linewidth=1.2,
                       zorder=5, label=f"Optimum E={result_p1['energy']:.4f}")
            ax.legend(fontsize=10)

        ax.set_xlabel(r"$\gamma / \pi$", fontsize=12)
        ax.set_ylabel(r"$\beta / \pi$", fontsize=12)
        title(ax, "Energy landscape (p=1)",
              r"$E(\gamma, \beta) = \langle\psi|H_C|\psi\rangle$ over the two variational angles")
        ax.grid(False)

        plt.tight_layout()
        if save:
            _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            path = _PLOTS_DIR / f"{name}_landscape.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"✓ saved → {_rel(path)}")
        plt.show()


# ════════════════════════════════════════════════════════════════════════
# Thermodynamics — spin-glass connection
# ════════════════════════════════════════════════════════════════════════
class Thermodynamics:
    """
    Statistical-physics view of the portfolio Ising Hamiltonian.

    The classical partition function at inverse temperature β_th is
        Z(β_th) = Σ_z exp(-β_th H(z))
    and the thermal energy is
        ⟨H⟩_β = Σ_z H(z) exp(-β_th H(z)) / Z.

    As β_th → ∞, ⟨H⟩_β → E_0 (the ground state). QAOA at large p
    approximates this zero-temperature limit within the variational class.

    Parameters
    ----------
    qaoa : QAOA
        A QAOA instance (we use its H_C diagonal for energies).
    """

    def __init__(self, qaoa):
        self.qaoa = qaoa
        # H_C is diagonal; its diagonal IS the array of all 2^n Ising energies
        self._E_all = qaoa._HC_diag.real
        self._E0 = float(self._E_all.min())

    def thermal_energy(self, beta_th):
        """⟨H⟩_β at inverse temperature β_th (scalar or array)."""
        beta_th = np.atleast_1d(beta_th)
        out = np.empty_like(beta_th, dtype=float)
        for i, b in enumerate(beta_th):
            # Stable softmax-style: subtract min to avoid overflow at large β
            shifted = -b * (self._E_all - self._E0)
            w = np.exp(shifted)
            Z = w.sum()
            out[i] = self._E0 + np.dot(self._E_all - self._E0, w) / Z
        return out if out.size > 1 else float(out[0])

    def plot(self, qaoa_results=None, save=False, name="thermodynamics"):
        """
        Two-panel figure:
            [0] Thermal energy ⟨H⟩_β vs β_th (log scale), with QAOA
                energies as horizontal lines.
            [1] Histogram of Ising energies over all 2^n bitstrings,
                with the ground state marked.

        Parameters
        ----------
        qaoa_results : dict[int, dict], optional
            Mapping {p: result_dict} from QAOA.optimise. Energies are
            overlaid as dashed horizontal lines.
        save : bool
        name : str
            Filename prefix; saved to plots/analysis/.
        """
        _apply_style()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f"Statistical physics connection — {name}",
                     fontweight="bold", fontsize=13, color=PALETTE["charcoal"],
                     x=0.02, ha="left", y=0.995)

        # ── [0] Thermal energy vs β_th ───────────────────────────────
        ax = axes[0]
        beta_vals = np.logspace(-2, 3, 200)
        therm = self.thermal_energy(beta_vals)

        ax.semilogx(beta_vals, therm, color=PALETTE["blue"], lw=2.2,
                    label=r"$\langle H\rangle_{\beta}$")
        ax.axhline(self._E0, color=PALETTE["red"], ls="--", lw=1.6,
                   label=f"Ground state $E_0 = {self._E0:.4f}$")
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
        ax.set_ylabel(r"Thermal energy $\langle H\rangle_\beta$", fontsize=11)
        title(ax, "Thermal energy vs temperature",
              r"$\langle H\rangle_\beta \to E_0$ as $\beta_{\rm th} \to \infty$ — QAOA approximates this zero-T limit")
        ax.legend(fontsize=8, loc="upper right")

        # ── [1] Histogram of Ising energies ──────────────────────────
        ax = axes[1]
        # Bin count: heuristic — sqrt(N) but capped
        n_bins = min(max(int(np.sqrt(len(self._E_all))), 8), 30)
        ax.hist(self._E_all, bins=n_bins, color=PALETTE["blue_muted"],
                edgecolor=PALETTE["charcoal"], linewidth=0.5, alpha=0.85)
        ax.axvline(self._E0, color=PALETTE["red"], ls="--", lw=1.8,
                   label=f"Ground state $E_0 = {self._E0:.4f}$")
        ax.set_xlabel(r"Ising energy $H(z)$", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        title(ax, "Ising energy distribution",
              rf"Histogram of $H(z)$ over all $2^n = {len(self._E_all)}$ bitstrings")
        ax.legend(fontsize=9)

        plt.tight_layout()
        if save:
            _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            path = _PLOTS_DIR / f"{name}_thermodynamics.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"✓ saved → {_rel(path)}")
        plt.show()


# ════════════════════════════════════════════════════════════════════════
# Combined diagnostics figure
# ════════════════════════════════════════════════════════════════════════
def plot_diagnostics(qaoa, qaoa_results, save=False, name="diagnostics",
                     n_pts=40):
    """
    Single combined figure with all three diagnostic panels:
        [top]      QAOA landscape with optimum
        [bot left] Thermal energy vs β_th
        [bot right] Ising energy histogram

    Parameters
    ----------
    qaoa : QAOA
        Fitted QAOA instance.
    qaoa_results : dict[int, dict]
        {p: result_dict} from QAOA.optimise. Result for p=1 is used to
        mark the optimum on the landscape; all results overlay on the
        thermal panel.
    save : bool
    name : str
    n_pts : int
        Landscape grid resolution.
    """
    _apply_style()
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28,
                  height_ratios=[1.2, 1])
    fig.suptitle(f"QAOA diagnostics — {name}",
                 fontweight="bold", fontsize=14, color=PALETTE["charcoal"],
                 x=0.02, ha="left", y=0.995)

    # ── [top, full width] Landscape ──────────────────────────────────
    landscape = Landscape(qaoa, n_pts=n_pts).scan()
    ax = fig.add_subplot(gs[0, :])
    GG, BB = np.meshgrid(landscape.gammas, landscape.betas, indexing="ij")
    cf = ax.contourf(GG / np.pi, BB / np.pi, landscape.energy,
                     levels=25, cmap="viridis")
    plt.colorbar(cf, ax=ax, label=r"$E(\gamma, \beta)$")

    if 1 in qaoa_results:
        res = qaoa_results[1]
        g_opt = res["gammas"][0] % (2 * np.pi)
        b_opt = res["betas"][0]
        ax.scatter(g_opt / np.pi, b_opt / np.pi,
                   c=PALETTE["red"], s=220, marker="*",
                   edgecolors=PALETTE["charcoal"], linewidth=1.2, zorder=5,
                   label=f"p=1 optimum E={res['energy']:.4f}")
        ax.legend(fontsize=10)

    ax.set_xlabel(r"$\gamma / \pi$", fontsize=12)
    ax.set_ylabel(r"$\beta / \pi$", fontsize=12)
    title(ax, "Energy landscape (p=1)",
          r"$E(\gamma, \beta) = \langle\psi|H_C|\psi\rangle$ over the two variational angles")
    ax.grid(False)

    # ── [bot left] Thermal energy ────────────────────────────────────
    therm = Thermodynamics(qaoa)
    ax = fig.add_subplot(gs[1, 0])
    beta_vals = np.logspace(-2, 3, 200)
    therm_E = therm.thermal_energy(beta_vals)
    ax.semilogx(beta_vals, therm_E, color=PALETTE["blue"], lw=2.2,
                label=r"$\langle H\rangle_{\beta}$")
    ax.axhline(therm._E0, color=PALETTE["red"], ls="--", lw=1.6,
               label=f"$E_0 = {therm._E0:.4f}$")
    ax.axhline(therm_E[0], color=PALETTE["grey"], ls=":", lw=1.4,
               label="High-T avg")
    qaoa_colors = [PALETTE["teal"], PALETTE["ochre"], PALETTE["purple"]]
    for k, (p, res) in enumerate(sorted(qaoa_results.items())):
        ax.axhline(res["energy"],
                   color=qaoa_colors[k % len(qaoa_colors)],
                   ls=(0, (4, 1.5)), lw=1.4,
                   label=f"QAOA p={p}")
    ax.set_xlabel(r"Inverse temperature $\beta_{\rm th}$", fontsize=11)
    ax.set_ylabel(r"$\langle H\rangle_\beta$", fontsize=11)
    title(ax, "Thermal energy vs temperature",
          r"$\langle H\rangle_\beta \to E_0$ as $\beta_{\rm th} \to \infty$; QAOA energies overlaid")
    ax.legend(fontsize=8, loc="upper right")

    # ── [bot right] Histogram ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    n_bins = min(max(int(np.sqrt(len(therm._E_all))), 8), 30)
    ax.hist(therm._E_all, bins=n_bins, color=PALETTE["blue_muted"],
            edgecolor=PALETTE["charcoal"], linewidth=0.5, alpha=0.85)
    ax.axvline(therm._E0, color=PALETTE["red"], ls="--", lw=1.8,
               label=f"$E_0 = {therm._E0:.4f}$")
    ax.set_xlabel(r"Ising energy $H(z)$", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    title(ax, "Ising energy distribution",
          rf"Histogram of $H(z)$ over all $2^n = {len(therm._E_all)}$ bitstrings")
    ax.legend(fontsize=9)

    if save:
        _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        path = _PLOTS_DIR / f"{name}_diagnostics.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"✓ saved → {_rel(path)}")
    plt.show()