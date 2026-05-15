# Plotting conventions

House style for every figure in project 2. Anything new should follow this.

## Palette

Source of truth: [`code/palette/palette.json`](../code/palette/palette.json) — 18 named colours.

- `scripts/snp.py` loads it as `PALETTE` and applies the style via `_apply_style()`.
- Notebooks load it directly:

  ```python
  PALETTE = json.loads(
      (Path.cwd().parent / "palette" / "palette.json").read_text(encoding="utf-8")
  )
  ```

- Never hard-code a hex string. Pick a name from the palette (`PALETTE["red"]`, `PALETTE["blue"]`, …). Editing the JSON re-themes every figure that follows the convention.

## Defaults

- **Size:** `figsize=(12, 6)`. Wider when a figure has many bars or x-tick labels; never taller without reason.
- **Style:** call `_apply_style()` once per notebook. Strips top/right spines, sets a light dashed grid, applies the editorial typeface defaults.
- **Output format:** PDF, vector. Save with `bbox_inches="tight"`.
- **Output root:** every plot lives under [`project2/code/plots/<category>/`](../code/plots/) — one root only, no parallel directories elsewhere. Categories in use:
  - `eda/`       — exploratory dataset overviews (`snp.SNP.plot`).
  - `visuals/`   — motivational / report figures (`notebooks/visuals.ipynb`).
  - `compare/`   — method comparisons (`scripts/compare.Compare.plot`).
  - `analysis/`  — QAOA diagnostics (`scripts/analysis.Landscape`, `Thermodynamics`).

Create the directory on first use:

```python
PLOTS_DIR = Path.cwd().parent / "plots" / "visuals"   # notebooks are at code/notebooks/
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(PLOTS_DIR / "name.pdf", bbox_inches="tight")
```

Module code computes `_PLOTS_DIR = _ROOT / "plots" / <category>` where `_ROOT = Path(__file__).resolve().parent.parent` resolves to `project2/code/`.

## Titles — two lines, left-shifted

Every figure has a bold short headline plus a non-bold descriptive subtitle, both left-aligned. Use the shared helper:

```python
from scripts.snp import title

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
- Grid is light dashed grey at low alpha — already configured by `_apply_style()`.
- Annotate sparingly. Direct labels on points beat a packed legend.
- Markers: `s=70–140` for "important" points, `s=6–12` for clouds. Keep ≤ 3 size tiers per figure.
- Prefer filled circles for assets and open circles (`facecolor='white'`) for derived/summary points like MVP/tangency. Avoid stars, diamonds, and crosses unless they encode something the colour can't.
- Use `legend(frameon=False, fontsize=9)`.

## Saving + showing

Inside a notebook, save *before* `plt.show()` so the figure isn't closed when the cell renders:

```python
fig.savefig(PLOTS_DIR / f"{name}.pdf", bbox_inches="tight")
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
