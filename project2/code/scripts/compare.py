"""
compare.py — Side-by-side runner for the classical baselines and QAOA.

Pure orchestration: build a problem, call `scripts.classical.run_all`,
call `scripts.qaoa.solve` at one or more depths, print a table, and plot
the headline figures. No cost function or QAOA internals live here.

Usage
-----
    from scripts.data    import load_returns
    from scripts.baskets import config
    from scripts.portfolio import PortfolioProblem
    from scripts.compare   import Compare

    tickers, start, end = config("mag7")
    r  = load_returns(tickers, start, end, cache_name="mag7")
    pf = PortfolioProblem(r.mu, r.Sigma, lam=2.0, A=0.5, K=2,
                          tickers=list(r.tickers))

    cmp = Compare(pf).run(p_values=[1, 2, 3])
    cmp.report()
    cmp.plot(save=True, name="mag7")
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from scripts.portfolio import PortfolioProblem
from scripts.classical import brute_force
from scripts.qaoa      import solve, decode_top_k
from scripts.plotting  import PALETTE, apply_style, title, rel_path, PLOTS_DIR


class Compare:
    """Run brute force + QAOA at multiple depths on one problem instance."""

    def __init__(self, problem: PortfolioProblem):
        self.pf = problem
        self.bf = None                  # brute_force SolverResult
        self.qaoa_results: dict[int, dict] = {}
        self._E0: float = float("nan")

    # ── Run ────────────────────────────────────────────────────────────
    def run(self, p_values=(1, 2, 3),
            n_restarts: int = 10, seed: int = 42,
            verbose: bool = True):
        if verbose:
            print(f"=== Compare on {self.pf} ===")

        self.bf = brute_force(self.pf)
        if verbose:
            print(f"\n[Brute force] {self.bf.tickers(self.pf)}  "
                  f"C = {self.bf.cost:.6f}")

        for p in p_values:
            res = solve(self.pf, p=p, n_restarts=n_restarts, seed=seed)
            self.qaoa_results[p] = res
            self._E0 = res["ground_state_energy"]
            top1 = decode_top_k(res["probs"], self.pf, k=1)[0]
            res["top1"] = top1
            if verbose:
                print(f"[QAOA p={p}] E={res['energy']:.6f}  "
                      f"ratio={res['ratio']:.4f}  "
                      f"top={top1['bitstring']}  C(top)={top1['cost']:.6f}")
        return self

    # ── Reporting ──────────────────────────────────────────────────────
    def report(self):
        assert self.bf is not None, "call .run() before .report()"
        pf = self.pf

        def portfolio_str(x):
            return "+".join(t for t, v in zip(pf.tickers, x) if v)

        line = "=" * 70
        print(f"\n{line}")
        print(f"METHOD COMPARISON  (n={pf.n}, K={pf.K}, lam={pf.lam}, A={pf.A})")
        print(line)
        print(f"  {'Method':<22} {'Portfolio':<24} {'Cost':>14}  {'Ratio':>6}")
        print("-" * 70)

        print(f"  {'Brute force':<22} {portfolio_str(self.bf.x):<24} "
              f"{self.bf.cost:>14.6f}  {'1.0000':>6}")

        for p in sorted(self.qaoa_results.keys()):
            res = self.qaoa_results[p]
            top = res["top1"]
            print(f"  {'QAOA p='+str(p)+' (top-1)':<22} "
                  f"{portfolio_str(top['x']):<24} "
                  f"{top['cost']:>14.6f}  {res['ratio']:>6.4f}")
        print(line)

    # ── Plotting ───────────────────────────────────────────────────────
    def plot(self, save: bool = False, name: str = "comparison"):
        apply_style()
        pf = self.pf
        ps = sorted(self.qaoa_results.keys())
        best_p = max(ps, key=lambda p: self.qaoa_results[p]["ratio"])

        fig = plt.figure(figsize=(14, 5.5))
        gs = GridSpec(1, 2, figure=fig, wspace=0.3)
        fig.suptitle(f"QAOA vs brute force — {name}", fontweight="bold",
                     fontsize=14, color=PALETTE["charcoal"],
                     x=0.02, ha="left", y=0.995)

        # [0] Energy & ratio vs depth
        ax = fig.add_subplot(gs[0, 0])
        energies = [self.qaoa_results[p]["energy"] for p in ps]
        ratios   = [self.qaoa_results[p]["ratio"]  for p in ps]
        ax.plot(ps, energies, "o-", color=PALETTE["blue"], lw=2, ms=10,
                label="QAOA energy")
        ax.axhline(self._E0, color=PALETTE["red"], ls="--", lw=1.5,
                   label=f"Ground state E0 = {self._E0:.4f}")
        ax.set_xlabel("Circuit depth p"); ax.set_ylabel("Energy")
        ax.set_xticks(ps)
        title(ax, "QAOA energy vs depth",
              "Lower is better; ground state $E_0$ (red dashed) is the brute-force optimum")
        ax.legend(fontsize=9)
        for p, e, r in zip(ps, energies, ratios):
            ax.annotate(f"r={r:.3f}", (p, e),
                        xytext=(8, 8), textcoords="offset points", fontsize=9)

        # [1] Measurement probabilities at the best depth
        ax = fig.add_subplot(gs[0, 1])
        probs = self.qaoa_results[best_p]["probs"]
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

        if save:
            out_dir = PLOTS_DIR / "compare"
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{name}_comparison.pdf"
            fig.savefig(path, bbox_inches="tight")
            print(f"saved -> {rel_path(path)}")
        plt.show()

    def __repr__(self):
        ps = list(self.qaoa_results.keys())
        return f"Compare({self.pf}, qaoa_p={ps})"
