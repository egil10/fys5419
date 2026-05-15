"""
compare.py — Side-by-side comparison of QAOA and brute-force baselines.

Usage
-----
    from scripts.snp       import SNP
    from scripts.baskets   import config
    from scripts.portfolio import Portfolio
    from scripts.compare   import Compare

    tickers, start, end = config("mag7")
    snp = SNP(tickers, start, end).cached_fetch(name="mag7")
    pf = Portfolio(snp.mu, snp.Sigma, lam=2.0, A=0.5, K=2, tickers=snp.tickers)

    cmp = Compare(pf).run(p_values=[1, 2, 3])
    cmp.report()
    cmp.plot(save=True, name="mag7")
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

from scripts.qaoa import QAOA
from scripts.snp  import PALETTE, _apply_style, _rel, title

_ROOT = Path(__file__).resolve().parent.parent
_PLOTS_DIR = _ROOT / "plots" / "compare"


class Compare:
    """
    Run brute-force and QAOA (at multiple depths p) on a single portfolio
    and collect results for joint reporting and plotting.

    Parameters
    ----------
    portfolio : Portfolio
        The mean-variance problem to solve.
    """

    def __init__(self, portfolio):
        self.pf = portfolio
        self.bf = None         # brute-force result
        self.qaoa_results = {} # {p: result_dict}

    # ── Run ───────────────────────────────────────────────────────────
    def run(self, p_values=(1, 2, 3),
            qaoa_restarts=15, qaoa_seed=42,
            verbose=True):
        """Run brute-force and QAOA at each p in p_values."""
        if verbose:
            print(f"=== Compare on {self.pf} ===")

        self.bf = self.pf.brute_force()
        if verbose:
            print(f"\n[Brute force] {self.bf['tickers']}  C = {self.bf['cost']:.6f}")

        qaoa = QAOA(self.pf, seed=qaoa_seed)
        self._E0 = qaoa.ground_state_energy()
        if verbose:
            print(f"[QAOA]        H_C ground state E0 = {self._E0:.6f}")
        for p in p_values:
            res = qaoa.optimise(p=p, n_restarts=qaoa_restarts)
            self.qaoa_results[p] = {
                "result":    res,
                "qaoa":      qaoa,
                "ratio":     qaoa.approximation_ratio(res["energy"]),
                "top":       qaoa.decode(res["probs"], top_k=1)[0],
            }
            if verbose:
                t = self.qaoa_results[p]
                print(f"              p={p}: E = {res['energy']:.6f}  "
                      f"ratio = {t['ratio']:.4f}  top = {t['top']['bitstring']}")
        return self

    # ── Reporting ─────────────────────────────────────────────────────
    def report(self):
        """Print a method comparison table."""
        pf = self.pf
        bf = self.bf

        def portfolio_str(x):
            return "+".join(t for t, v in zip(pf.tickers, x) if v)

        line = "=" * 70
        print(f"\n{line}")
        print(f"METHOD COMPARISON  (n={pf.n}, K={pf.K}, λ={pf.lam}, A={pf.A})")
        print(line)
        print(f"  {'Method':<22} {'Portfolio':<24} {'Cost':>14}  {'Ratio':>6}")
        print("-" * 70)

        print(f"  {'Brute force':<22} {portfolio_str(bf['x']):<24} "
              f"{bf['cost']:>14.6f}  {'1.0000':>6}")

        for p, t in self.qaoa_results.items():
            top = t["top"]
            print(f"  {'QAOA p='+str(p)+' (top-1)':<22} "
                  f"{portfolio_str(top['x']):<24} "
                  f"{top['C_finance']:>14.6f}  {t['ratio']:>6.4f}")
        print(line)

    # ── Plotting ──────────────────────────────────────────────────────
    def plot(self, save=False, name="comparison"):
        """
        2-panel comparison figure:
            [0] QAOA energy & approx ratio vs p (ground state E0 marked)
            [1] QAOA probs (best p) — budget-K states highlighted, uniform baseline
        """
        _apply_style()
        pf = self.pf
        ps = sorted(self.qaoa_results.keys())
        best_p = max(ps, key=lambda p: self.qaoa_results[p]["ratio"])

        fig = plt.figure(figsize=(14, 5.5))
        gs = GridSpec(1, 2, figure=fig, wspace=0.3)
        fig.suptitle(f"QAOA vs brute force — {name}", fontweight="bold",
                     fontsize=14, color=PALETTE["charcoal"],
                     x=0.02, ha="left", y=0.995)

        # ── [0] QAOA energy + ratio vs p ──────────────────────────────
        ax = fig.add_subplot(gs[0, 0])
        energies = [self.qaoa_results[p]["result"]["energy"] for p in ps]
        ratios = [self.qaoa_results[p]["ratio"] for p in ps]
        ax.plot(ps, energies, "o-", color=PALETTE["blue"], lw=2, ms=10,
                label="QAOA energy")
        ax.axhline(self._E0, color=PALETTE["red"], ls="--", lw=1.5,
                   label=f"Ground state E₀ = {self._E0:.4f}")
        ax.set_xlabel("Circuit depth p"); ax.set_ylabel("Energy")
        ax.set_xticks(ps)
        title(ax, "QAOA energy vs depth",
              "Lower is better; ground state $E_0$ (red dashed) is the brute-force optimum")
        ax.legend(fontsize=9)
        for p, e, r in zip(ps, energies, ratios):
            ax.annotate(f"r={r:.3f}", (p, e),
                        xytext=(8, 8), textcoords="offset points", fontsize=9)

        # ── [1] QAOA measurement probabilities (best p) ───────────────
        ax = fig.add_subplot(gs[0, 1])
        probs = self.qaoa_results[best_p]["result"]["probs"]
        n_states = len(probs)
        budget_ok = np.array([
            bin(k).count("1") == pf.K for k in range(n_states)
        ])
        bit_labels = [format(k, f"0{pf.n}b") for k in range(n_states)]
        cols = [PALETTE["red"] if ok else PALETTE["blue_muted"]
                for ok in budget_ok]
        ax.bar(range(n_states), probs, color=cols,
               edgecolor=PALETTE["charcoal"], linewidth=0.4)
        ax.axhline(1.0 / n_states, color=PALETTE["grey"], ls="--", lw=1,
                   label=f"Uniform 1/{n_states}")
        ax.set_xticks(range(n_states))
        ax.set_xticklabels(bit_labels, rotation=90, fontsize=7)
        ax.set_ylabel("Probability")
        title(ax, "QAOA measurement probabilities",
              f"Best depth p={best_p}; red bars satisfy the budget K={pf.K}")
        ax.legend(fontsize=9)
        ax.text(0.98, 0.95, f"red = budget {pf.K}", transform=ax.transAxes,
                ha="right", va="top", color=PALETTE["red"],
                fontsize=9, style="italic")

        if save:
            _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
            path = _PLOTS_DIR / f"{name}_comparison.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"✓ saved → {_rel(path)}")
        plt.show()

    def __repr__(self):
        ps = list(self.qaoa_results.keys())
        return f"Compare({self.pf}, qaoa_p={ps})"
