# Plotting conventions

House style for every figure in project 2. Anything new should follow this.

## Palette

Source of truth: [`code/palette/palette.json`](../palette/palette.json) — 18 named colours.

- `scripts/snp.py` loads it as `PALETTE` and applies the style via `apply_style()`.
- Notebooks load it directly:

  ```python
  PALETTE = json.loads(open('../palette/palette.json', encoding='utf-8').read())
  ```

- Never hard-code a hex string. Pick a name from the palette (`PALETTE["red"]`, `PALETTE["blue"]`, …). Editing the JSON re-themes every figure that follows the convention.

## Defaults

- **Size:** `figsize=(12, 6)` baseline; wider when many bars/x-ticks, never taller without reason.
- **Style:** call `apply_style()` once per notebook. Strips top/right spines, **disables the grid globally**, applies the editorial typeface defaults. If you want a grid back for a specific axis call `ax.grid(True)` explicitly.
- **Output format:** PDF, vector. Save with `bbox_inches="tight"`.
- **Output root:** every plot lives under [`code/plots/<category>/`](../plots/) on local and `<Drive>/.../code/plots/<category>/` on Colab. Use `fig_path(category, name)` from `scripts.plotting` — never construct the path by hand. Categories in use:
  - `eda/`       — exploratory dataset overviews (`01_eda` via `SNP.plot`).
  - `visuals/`   — motivational / report figures (`00_visuals`).
  - `qaoa/`      — single-run diagnostics (`03_qaoa`).
  - `compare/`   — headline sweep figures (`08_compare`).
  - `analysis/`  — QAOA diagnostics (`scripts.analysis.Landscape`, `Thermodynamics`).
  - `snp/`       — reserved for any per-basket sanity plots driven by `00_snp` (currently unused).

The recommended pattern in any notebook:

```python
from scripts.plotting import apply_style, PALETTE, title, fig_path
apply_style()
fig, ax = plt.subplots(figsize=(12, 6))
# ... build figure ...
fig.savefig(fig_path('compare', 'depth_sweep'), bbox_inches='tight')
plt.show()
```

`fig_path()` resolves to Drive on Colab and to the local repo otherwise.

## Titles — two lines, left-shifted

Every figure has a bold short headline plus a non-bold descriptive subtitle, both left-aligned. Use the shared helper:

```python
from scripts.plotting import title

title(ax, "Markowitz efficient frontier",
      "Continuous weights on the simplex — mag7 (4 assets, daily 2023–2025)")
```

- The bold line names *what* the plot is in 2–5 words.
- The subtitle adds the *binding parameters* — basket, `K`, `λ`, `p`, date range — and any caveat the reader needs.
- For `fig.suptitle`, pass `x=0.02, ha="left", y=0.995` so the page header matches.

| Don't                          | Do                                                                       |
| ------------------------------ | ------------------------------------------------------------------------ |
| `ax.set_title("Returns")`      | `title(ax, "Log-returns", "Daily log-returns (%) per asset, overlaid")` |
| `Cost vs index`                | `title(ax, "Cost landscape", "All 2^n bitstrings sorted; red = budget K=2")` |
| `2^n`                          | `title(ax, "Combinatorial wall", "Why brute force fails as n grows")` |

## Minimal aesthetic

- One idea per figure. If two ideas fight for the title, split the figure.
- No top or right spines, no boxed legends, no chartjunk shadows or 3-D effects.
- **No grid by default.** `apply_style()` sets `axes.grid: False`. Add `ax.grid(True)` explicitly only when the plot genuinely needs it (rare).
- Annotate sparingly. Direct labels on points beat a packed legend.
- Markers: `s=70–140` for "important" points, `s=6–12` for clouds. Keep ≤ 3 size tiers per figure.
- Prefer filled circles for assets and open circles (`facecolor='white'`) for derived/summary points like MVP/tangency. Avoid stars, diamonds, and crosses unless they encode something the colour can't.
- Use `legend(frameon=False, fontsize=9)`.

## Saving + showing

Inside a notebook, save *before* `plt.show()` so the figure isn't closed when the cell renders:

```python
fig.savefig(fig_path('compare', 'foo'), bbox_inches="tight")
plt.show()
```

For batch scripts, close after saving: `plt.close(fig)`.

## Re-using panels across figures

`SNP._panel_*` methods take an `ax` and draw one panel. Compose them when building a multi-panel figure — don't copy-paste the body. The current panel list:

- `_panel_prices` — log-scaled normalised prices.
- `_panel_returns` — overlaid log-returns.
- `_panel_rolling_vol` — 30-period annualised vol.
- `_panel_hist` — return distribution.
- `_panel_correlation` — leading-eigenvector seriated correlation matrix.
- `_panel_risk_return` — annualised risk-return scatter with jittered labels.

If a new figure type appears in three places, lift it into a `_panel_*` method on `SNP` or a helper in `scripts/`.
