# Project 2 — QAOA for Portfolio Optimisation

Comparing the Quantum Approximate Optimisation Algorithm against classical
baselines on the cardinality-constrained mean-variance portfolio.

## Problem

For `n` assets with mean returns μ and covariance Σ, pick `K` of them to minimise

    C(x) = -μᵀx + λ xᵀΣx + A (Σ xᵢ - K)²,   x ∈ {0,1}ⁿ.

This is a QUBO. Every solver targets the **same `C(x)`** — that is the
apples-to-apples comparison. The one true definition lives in
[`scripts/portfolio.py`](code/scripts/portfolio.py); no solver re-implements it.

## Layout

```
project2/code/
├── scripts/        Pure-Python modules; no notebook side-effects
│   ├── colab.py        Colab + local bootstrap helper (one-line setup)
│   ├── baskets.py      Thematic ticker baskets
│   ├── snp.py          Data loader + 6-panel EDA (legacy class, still used by snp.ipynb)
│   ├── data.py         load_returns(...) -> Returns (functional API)
│   ├── portfolio.py    PortfolioProblem dataclass + eval_cost — THE cost function
│   ├── ising.py        QUBO <-> Ising mapping; H_C diag + H_M sparse builders
│   ├── classical.py    SolverResult + brute_force, greedy, SA, Markowitz-round
│   ├── qaoa.py         Functional QAOA: statevector, energy, decode, solve(...)
│   ├── optimize.py     multi_start_minimize wrapper (COBYLA / SPSA)
│   ├── metrics.py      approx ratio (scaled), gap, P(opt), P(feas), Sharpe
│   ├── plotting.py     Re-exports of style helpers + fig_path(...)
│   ├── compare.py      Compare class — brute force + QAOA at multiple p
│   └── analysis.py     Landscape + Thermodynamics diagnostics
├── notebooks/
│   ├── snp.ipynb         Cache S&P data into data/ (run once per basket)
│   ├── visuals.ipynb     Motivational figures (frontier, cost landscape, …)
│   ├── 01_eda.ipynb      Sanity-check the data, generate EDA figures
│   ├── 02_classical.ipynb Brute force, greedy, Markowitz-round, SA (10 seeds)
│   ├── 03_qaoa.ipynb     Single QAOA run with training restarts + top-5
│   ├── 04_depth.ipynb    Sweep 1 — p ∈ {1..5}, ratio + P(opt) + P(feas)
│   ├── 05_scaling.ipynb  Sweep 2 — n ∈ {4,6,8,10,12}, classical vs QAOA
│   ├── 06_risk.ipynb     Sweep 3 — λ over two decades
│   └── 07_compare.ipynb  Headline figures (reads results/*.json)
├── data/           Cached parquet prices + small CSV previews
├── plots/          PDF outputs from the notebooks
├── results/        Cached sweep outputs (.json) so re-plotting stays fast
└── tests/          pytest — QUBO↔Ising round-trip, classical sanity, QAOA sanity
```

## One-line bootstrap (Colab + local)

The first cell of every numbered notebook runs:

```python
import sys, os
try:
    import google.colab
    !test -d /content/fys5419 || git clone -q https://github.com/egil10/fys5419.git /content/fys5419
    %cd /content/fys5419/project2/code/notebooks
except ImportError:
    pass
sys.path.append('..')
from scripts.colab import setup; setup()
```

On Colab: clones the repo (if missing), `cd`s to the notebook dir, installs the
small dependency set, and adds `scripts/` to `sys.path`. Locally: the
`google.colab` import fails silently, `setup()` walks up from the cwd to find
the `scripts/` dir, and adds it to the path. Idempotent.

## Canonical dataset: ONE universe, sub-sampled

There is exactly one dataset: the **16-asset universe** = Mag7 + quantum +
quantum_big + anti, 2023-01-01 to 2025-12-31 daily log returns. It lives at
[`results/universe_16.npz`](code/results/) (computed on first call to
`load_universe()` and cached forever).

- **Notebooks 02, 03, 04, 06** fix `n=16, K=4` and call `load_universe()`.
  Same μ and Σ everywhere — no per-notebook recomputation.
- **Notebook 05** loads the same universe and **sub-samples** for each
  `n ∈ {4, 6, 8, 10, 12, 14, 16}`. For each `n < 16` it draws 20 random
  subsets, runs every solver, and reports median + IQR. The `n=16`
  endpoint is the single canonical subset (all 16 assets), and by
  construction reproduces the runs in notebooks 02/03/04/06.
- `K` follows the ratio rule `K = round(0.25 * n)` ∈ {1, 2, 2, 3, 3, 4, 4}.
  At `n=16` this gives `K=4`, matching `DEFAULTS["K_AT_16"]`.

Project-wide constants live in `scripts.portfolio.DEFAULTS` so every
notebook imports `lam`, `A`, `K_FRAC`, and `K_AT_16` from the same place.

## Workflow

1. **`snp.ipynb`** — fetch and cache prices (run once per basket; also
   populates the universe parquet via `load_universe()`).
2. **`01_eda.ipynb`** — confirm the data is sane, per-basket EDA figures.
3. **`02_classical.ipynb`** — baselines table at the canonical `n=16, K=4`.
4. **`03_qaoa.ipynb`** — one QAOA run end-to-end at `n=16, K=4, p=3`.
5. **`04_depth.ipynb`** — Sweep 1 (p ∈ {1..5}) at `n=16, K=4`.
   Writes `results/depth_sweep.json`.
6. **`05_scaling.ipynb`** — Sweep 2 (n ∈ {4..16} via random subsets, M=20).
   Writes `results/size_scaling.json`.
7. **`06_risk.ipynb`** — Sweep 3 (λ across two decades) at `n=16, K=4`.
   Writes `results/risk_sweep.json`.
8. **`07_compare.ipynb`** — read all three result files, produce the
   headline figures with median + IQR shading.

## Module ownership (the one-true-definition rules)

| File              | What lives here, and nowhere else                          |
|-------------------|------------------------------------------------------------|
| `data.py`         | Prices → (μ, Σ). No optimisation logic.                    |
| `portfolio.py`    | The cost function `C(x)` (`eval_cost`). Every solver imports from here. |
| `ising.py`        | QUBO ↔ Ising algebra; H_C and H_M builders.                |
| `classical.py`    | Every classical solver; uniform `SolverResult` dataclass.  |
| `qaoa.py`         | QAOA statevector + energy + `solve()` convenience.         |
| `optimize.py`     | Outer-loop wrappers (COBYLA, SPSA, multi-start).           |
| `metrics.py`      | Comparison metrics for tables/plots.                       |
| `plotting.py`     | Shared style — every notebook calls `apply_style()`.       |
| `colab.py`        | Bootstrap that works on Colab + locally.                   |

## Methodology — what you actually measure

- **Approximation ratio** is the **scaled** form
  `r = (E_worst − E) / (E_worst − E_opt) ∈ [0, 1]` (Farhi-style; interpretable
  even when E_opt is small in magnitude). `qaoa.solve()` returns this in `ratio`.
- **P(optimum)** and **P(feasible)** for QAOA (`scripts.metrics`).
- **Multi-start**: QAOA `solve()` defaults to 10 random `(γ, β)` inits and
  keeps the best — single-seed QAOA is not meaningful. SA notebook reports
  median + IQR over 10 seeds.
- Sweeps **cache** to `results/*.json` so the plotting notebook stays fast.

## Tests

```bash
pytest project2/code/tests -q
```

Covers the three "sanity checks before sweeping":

1. QUBO ↔ Ising round-trip (sign-flip catcher).
2. `H_C` diagonal entry `k` equals the Ising energy of bitstring `k` (indexing).
3. QAOA scaled ratio at `n=4, p=10` ≥ 0.95 (mixer correctness).
