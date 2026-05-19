# Project 2 — QAOA for Portfolio Optimisation

Comparing the Quantum Approximate Optimisation Algorithm to classical
baselines on the cardinality-constrained mean-variance portfolio.

## Problem

For `n` assets with mean returns μ and covariance Σ, pick `K` of them
to minimise

    C(x) = -μᵀx + λ xᵀΣx + A (Σ xᵢ - K)²,   x ∈ {0,1}ⁿ.

This is a QUBO. Every solver targets the same `C(x)` — that's the apples-
to-apples comparison.

## Layout

```
project2/code/
├── scripts/        Pure-Python modules; no plotting state, no notebooks import side-effects
│   ├── baskets.py      Thematic ticker baskets
│   ├── snp.py          Data loader + 6-panel EDA (legacy class)
│   ├── data.py         load_returns(...) -> Returns (functional API)
│   ├── portfolio.py    Portfolio container; cost C(x); brute_force (legacy)
│   ├── ising.py        QUBO <-> Ising; H_C diag and H_M sparse builders
│   ├── classical.py    brute_force, greedy, SA, Markowitz-round + SolverResult
│   ├── qaoa.py         QAOA: ansatz, statevector sim, COBYLA restarts (legacy class)
│   ├── optimize.py     multi-start COBYLA / SPSA wrappers
│   ├── metrics.py      approx ratio, P(optimal), P(feasible), Sharpe
│   ├── plotting.py     Re-exports of style helpers
│   ├── compare.py      Legacy: brute force vs QAOA reporting (class-based)
│   └── analysis.py     Legacy: parameter landscape + thermodynamics diagnostics
├── notebooks/      Load problem -> call solver -> plot/save. No class defs.
│   ├── snp.ipynb       Download S&P data into data/
│   ├── eda.ipynb       Sanity check + EDA figures
│   ├── visuals.ipynb   Motivational/report figures (frontier, cost landscape, ...)
│   ├── classical.ipynb Run every classical baseline, table the results
│   ├── qaoa.ipynb      Single-run QAOA demo (training curve, probabilities, top-5)
│   ├── depth.ipynb     Sweep p ∈ {1..5}; ratio + P(opt) + P(feasible)
│   ├── scaling.ipynb   Sweep n ∈ {4,6,8,10}; QAOA vs classical
│   └── compare.ipynb   Headline figures for the report (reads results/*.json)
├── data/           Cached parquet prices + small CSV previews
├── plots/          PDF outputs from the notebooks
├── results/        Sweep outputs (.json/.npz) so plotting notebooks stay fast
└── tests/          pytest — QUBO/Ising round-trip, classical sanity checks
```

## Module ownership (the one-true-definition rules)

| File              | What lives here, and nowhere else                          |
|-------------------|------------------------------------------------------------|
| `data.py`         | Prices → (μ, Σ). No optimisation logic.                    |
| `portfolio.py`    | The cost function `C(x)`. Every solver imports from here.  |
| `ising.py`        | QUBO ↔ Ising algebra; H_C and H_M builders.                |
| `classical.py`    | Every classical solver; uniform `SolverResult`.            |
| `qaoa.py`         | QAOA statevector simulation + energy.                      |
| `optimize.py`     | Outer-loop wrappers (COBYLA, SPSA, multi-start).           |
| `metrics.py`      | Comparison metrics for tables/plots.                       |
| `plotting.py`     | Shared style — every notebook calls `apply_style()`.       |

## Workflow

1. **`snp.ipynb`** — fetch and cache prices (run once per basket).
2. **`eda.ipynb`** — confirm the data is sane, generate report EDA figures.
3. **`classical.ipynb`** — baselines table for one `(n, K)`.
4. **`qaoa.ipynb`** — single QAOA run end-to-end on one `(n, K, p)`.
5. **`depth.ipynb`** — sweep `p`; cache to `results/`.
6. **`scaling.ipynb`** — sweep `n`; cache to `results/`.
7. **`compare.ipynb`** — read `results/*.json`, produce the headline figure(s).

## Tests

```bash
pytest project2/code/tests -q
```

Catches: QUBO ↔ Ising sign flips, H_C diagonal mis-indexing, classical
solvers returning infeasible or non-binary `x`.

## Requirements

Top-level `../requirements.txt` (numpy, pandas, scipy, matplotlib, yfinance, pytest).
