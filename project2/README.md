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
├── scripts/                Pure-Python modules; no notebook side-effects
│   ├── bootstrap.py         Two-line notebook bootstrap (Colab + local)
│   ├── colab.py             Drive-aware out_dir(); sys.path setup
│   ├── baskets.py           Thematic ticker baskets (one source of truth)
│   ├── snp.py               Data loader + 6-panel EDA class
│   ├── data.py              load_returns / load_universe (functional API)
│   ├── portfolio.py         PortfolioProblem + eval_cost (THE cost function)
│   ├── ising.py             QUBO ↔ Ising; H_C diag + H_M sparse
│   ├── classical.py         SolverResult + brute force / greedy / SA / Markowitz
│   ├── qaoa.py              Statevector QAOA + solve() multi-start convenience
│   ├── optimize.py          multi_start_minimize (COBYLA, SPSA), parallel
│   ├── metrics.py           scaled_ratio, gap, P(opt), P(feas), Sharpe
│   ├── plotting.py          Re-exports for the house style + fig_path()
│   └── analysis.py          QAOA diagnostics: (γ,β) landscape, thermodynamics
├── notebooks/
│   ├── 00_snp.ipynb          Pre-fetch all per-basket parquets to data/
│   ├── 00_visuals.ipynb      Motivational figures → plots/visuals/
│   ├── 01_eda.ipynb          16-equity EDA → plots/eda/
│   ├── 02_classical.ipynb    Baselines at n=16, K=4
│   ├── 03_qaoa.ipynb         Single QAOA run + (γ,β) landscape at p=1
│   ├── 04_depth.ipynb        Sweep 1 — p ∈ {1..5}
│   ├── 05_scaling.ipynb      Sweep 2 — n ∈ {4..16} via random subsets
│   ├── 06_risk.ipynb         Sweep 3 — λ across two decades
│   ├── 07_penalty.ipynb      Sweep 4 — A logspace(-2, 1.5, 8), 50 restarts
│   ├── 08_compare.ipynb      Headline plots reading all sweep caches
│   └── 09_xy_mixer.ipynb     Sweep 5 — XY ring + Dicke vs X + uniform (n=8)
├── data/                   Cached parquet prices + CSV previews
├── plots/                  PDF outputs by category (eda/ visuals/ qaoa/ compare/ analysis/ snp/)
├── results/                Cached sweep outputs (.json) so re-plotting is instant
└── tests/                  pytest — QUBO↔Ising round-trip, classical, QAOA, XY
```

Dependencies are pinned in the repo-root `requirements.txt` (one file
for the whole repo).

## One-line bootstrap (Colab + local)

The first cell of every numbered notebook runs:

```python
import os, urllib.request as _u
exec((open('../scripts/bootstrap.py') if os.path.exists('../scripts/bootstrap.py')
      else _u.urlopen('https://raw.githubusercontent.com/egil10/fys5419/main/project2/code/scripts/bootstrap.py')).read())
```

On Colab: clones the repo (if missing), `cd`s to the notebook dir, installs the
dependency set, and adds `scripts/` to `sys.path`. Locally: same path discovery,
no clone or install. Idempotent.

## Canonical dataset: ONE universe, sub-sampled

There is exactly one dataset: the **16-asset universe** = mag7 + quantum +
quantum_big + anti (definitions in [`scripts/baskets.py`](code/scripts/baskets.py)),
2023-01-01 to 2025-12-31 daily log returns. It is cached to
[`results/universe_16.npz`](code/results/) on first call to `load_universe()`.

- **Notebooks 02, 03, 04, 06, 07** fix `n=16, K=4` and call `load_universe()`.
  Same μ and Σ everywhere — no per-notebook recomputation.
- **Notebook 05** loads the same universe and **sub-samples** for each
  `n ∈ {4, 6, 8, 10, 12, 14, 16}`. For each `n < 16` it draws random subsets
  with a tiered count (20 at small n, fewer at large n since QAOA cost
  scales as 2ⁿ). The `n=16` endpoint is the canonical single subset (all 16
  assets) and reproduces the run in notebooks 02/03/04/06/07 by construction.
- `K = round(0.25 * n)` ∈ {1, 2, 2, 3, 3, 4, 4}. At `n=16` this gives `K=4`,
  matching `DEFAULTS["K_AT_16"]`.

Project-wide constants live in [`scripts.portfolio.DEFAULTS`](code/scripts/portfolio.py)
so every notebook imports `lam`, `A`, `K_FRAC`, `K_AT_16` from the same place.

## Workflow

Run top-to-bottom in numeric order. Estimates are Colab CPU.

| nb | role                              | runtime |
|----|-----------------------------------|---------|
| 00_snp     | Pre-fetch per-basket parquets                                     | ~30 s |
| 00_visuals | Motivational figures (frontier, cardinality, wall, cost landscape) | ~30 s |
| 01_eda     | 6-panel EDA on the 16-equity universe                              | ~30 s |
| 02_classical | Brute force / greedy / Markowitz / SA at n=16, K=4               | ~10 s |
| 03_qaoa    | One QAOA solve at p=3 + (γ,β) landscape at p=1                   | ~4–5 min |
| 04_depth   | **Sweep 1** — p ∈ {1..5}; writes `results/depth_sweep.json`        | ~12 min |
| 05_scaling | **Sweep 2** — n ∈ {4..16}; writes `results/size_scaling.json`      | ~45–60 min |
| 06_risk    | **Sweep 3** — λ over two decades; writes `results/risk_sweep.json`  | ~15–18 min |
| 07_penalty | **Sweep 4** — A ∈ {0.5, 2, 8, 32}; writes `results/penalty_sweep.json` | ~10 min |
| 08_compare | Four headline figures from the cached sweeps                       | ~5 s |

Sweep notebooks (04–07) all **cache** their JSON output and **resume** on partial
runs (Colab disconnects cost at most one inner iteration). Re-running with a
cache present just loads-and-skips.

## Module ownership (one-true-definition rules)

| File              | What lives here, and nowhere else                          |
|-------------------|------------------------------------------------------------|
| `baskets.py`      | Ticker baskets. Window dates live in `data.py`.            |
| `data.py`         | Prices → (μ, Σ). UNIVERSE_START / UNIVERSE_END. No optimisation logic. |
| `portfolio.py`    | The cost function `C(x)` (`eval_cost`) + `DEFAULTS`.       |
| `ising.py`        | QUBO ↔ Ising algebra; H_C diagonal builder; H_M sparse.    |
| `classical.py`    | Every classical solver; uniform `SolverResult` dataclass.  |
| `qaoa.py`         | Statevector QAOA + `solve()` convenience (multi-start).    |
| `optimize.py`     | Outer-loop wrappers (COBYLA, SPSA, multi-start, parallel). |
| `metrics.py`      | scaled_ratio, gap, P(opt), P(feas), Sharpe.                |
| `plotting.py`     | Shared style — every notebook calls `apply_style()`.       |
| `analysis.py`     | QAOA diagnostics: (γ,β) landscape, spin-glass thermodynamics. |
| `colab.py`        | Drive-aware `out_dir()` + `setup()`.                       |
| `bootstrap.py`    | Inline-execed bootstrap; clones+chdirs on Colab.            |

## Methodology — what you actually measure

- **Approximation ratio** is the **scaled** form
  `r = (E_worst − E) / (E_worst − E_opt) ∈ [0, 1]` (Farhi convention; stable
  even when E_opt is small in magnitude). `qaoa.solve()` returns this in `ratio`.
  Note that `E_worst` is dominated by the budget penalty `A·K²` — a "0.99" can
  mean "in the top percent of the spectrum" rather than "near the optimum",
  which is why we also report `gap_rel = (E − E_opt)/|E_opt|` and `P(optimum)`.
- **P(optimum)** and **P(feasible)** — measurement-mass diagnostics from `scripts.metrics`.
- **Multi-start mandatory** — `solve()` defaults to 10 random `(γ, β)` inits and
  keeps the best. Single-seed QAOA is not meaningful; the landscape is non-convex.
- Sweeps **cache + resume** to `results/*.json` so re-plotting stays fast.

## Tests

```bash
pip install -r requirements.txt
pytest project2/code/tests -q
```

Covers the three "sanity checks before sweeping":

1. QUBO ↔ Ising round-trip (sign-flip catcher).
2. `H_C` diagonal entry `k` equals the Ising energy of bitstring `k` (indexing).
3. QAOA scaled ratio at `n=4, p=10` ≥ 0.95 (mixer correctness).
