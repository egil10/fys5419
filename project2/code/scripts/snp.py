"""
snp.py — Stock data loader for portfolio analysis.

Usage
-----
    from scripts.snp import SNP
    from scripts.baskets import get

    # First run downloads + caches; later runs load from disk
    snp = SNP(get("mag7"), "2020-01-01", "2023-12-31").cached_fetch(name="mag7")
    snp.summary()
    snp.plot(save=True)                  # 2x3 grid (returns panel = stacked rows)
    snp.plot(save=True, individual=True) # one PDF per panel

    mu, Sigma = snp.mu, snp.Sigma
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data"
_PLOTS_DIR = _ROOT / "plots" / "eda"
_PALETTE_PATH = _ROOT / "palette" / "palette.json"

def _rel(path):
    """Display path relative to project root for clean console output."""
    try:
        return path.relative_to(_ROOT)
    except ValueError:
        return path.name

_ANN = {"1d": 252, "1wk": 52, "1mo": 12}

# ── Editorial palette ─────────────────────────────────────────────────
# Source of truth: scripts/palette.json (shared with notebooks/visuals.ipynb).
PALETTE = json.loads(_PALETTE_PATH.read_text(encoding="utf-8"))

_ASSET_CYCLE = [
    PALETTE["red"], PALETTE["blue"], PALETTE["teal"], PALETTE["ochre"],
    PALETTE["purple"], PALETTE["navy"], PALETTE["coral"], PALETTE["blue_muted"],
    PALETTE["crimson"], PALETTE["sky"], PALETTE["salmon"],
]

_CORR_CMAP = LinearSegmentedColormap.from_list(
    "editorial_div",
    [PALETTE["navy"], PALETTE["blue_muted"], "white",
     PALETTE["coral"], PALETTE["crimson"]],
)


def title(ax, bold, subtitle=None):
    """Two-line left-shifted title: bold headline + grey subtitle.

    Use everywhere a plot needs a title — keeps the house style consistent.
    """
    if subtitle:
        ax.set_title(bold, loc="left", fontweight="bold", fontsize=12, pad=22)
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=10,
                color=PALETTE["grey"], fontweight="normal")
    else:
        ax.set_title(bold, loc="left", fontweight="bold", fontsize=12)


def _apply_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "savefig.facecolor": "white",
        "axes.edgecolor":    PALETTE["charcoal"],
        "axes.labelcolor":   PALETTE["charcoal"],
        "axes.titlecolor":   PALETTE["charcoal"],
        "text.color":        PALETTE["charcoal"],
        "xtick.color":       PALETTE["charcoal"],
        "ytick.color":       PALETTE["charcoal"],
        "axes.grid":         True,
        "grid.color":        PALETTE["grid"],
        "grid.linestyle":    "--",
        "grid.alpha":        0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "axes.labelsize":    11,
        "legend.frameon":    False,
    })


class SNP:
    """Download and explore stock price data."""

    def __init__(self, tickers, start=None, end=None, interval="1d"):
        self.tickers = list(tickers)
        self.start, self.end, self.interval = start, end, interval
        self.prices = None

    # ── data ──────────────────────────────────────────────────────────
    def fetch(self):
        """Download adjusted close prices from yfinance."""
        import yfinance as yf
        data = yf.download(self.tickers, start=self.start, end=self.end,
                           interval=self.interval, auto_adjust=True,
                           progress=False)
        close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data[["Close"]]
        if not isinstance(data.columns, pd.MultiIndex):
            close.columns = self.tickers
        self.prices = close[self.tickers].dropna()
        print(f"✓ {len(self.prices)} {self.interval} obs for {self.tickers} "
              f"({self.prices.index[0].date()} → {self.prices.index[-1].date()})")
        return self

    def save(self, name="prices"):
        """Save prices to parquet (and a 20-row CSV preview) in code/project2/data/."""
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = _DATA_DIR / f"{name}.parquet"
        self.prices.to_parquet(path)
        print(f"✓ saved → {_rel(path)}")
        self._write_sample(name)
        return path

    def _write_sample(self, name, rows=20):
        """Write a small CSV preview of self.prices for quick inspection."""
        sample_path = _DATA_DIR / f"{name}_sample.csv"
        self.prices.head(rows).to_csv(sample_path)
        print(f"✓ sample → {_rel(sample_path)}")
        return sample_path

    @classmethod
    def load(cls, name="prices"):
        """Load prices from parquet in code/project2/data/."""
        path = Path(name)
        if not path.exists():
            path = _DATA_DIR / (name if name.endswith(".parquet") else f"{name}.parquet")
        prices = pd.read_parquet(path)
        obj = cls(list(prices.columns))
        obj.prices = prices
        print(f"✓ loaded {len(prices)} obs for {obj.tickers} from {path.name}")
        return obj

    def cached_fetch(self, name=None, force=False):
        """
        Fetch prices, using a local parquet cache if available.

        The cache is invalidated (re-fetched) when the on-disk parquet's
        columns don't match ``self.tickers`` or its date range doesn't
        cover ``[self.start, self.end]``.

        Parameters
        ----------
        name : str, optional
            Cache filename (without extension). Defaults to a fingerprint
            of tickers + date range, e.g. "AAPL-MSFT_2020-01-01_2023-12-31_1d".
        force : bool
            If True, ignore cache and re-download.
        """
        if name is None:
            name = (f"{'-'.join(self.tickers)}"
                    f"_{self.start or 'inf'}_{self.end or 'inf'}"
                    f"_{self.interval}")
        path = _DATA_DIR / f"{name}.parquet"

        if path.exists() and not force:
            cached = pd.read_parquet(path)
            cached_tickers = list(cached.columns)
            stale_reason = None
            if set(cached_tickers) != set(self.tickers):
                stale_reason = (f"tickers {cached_tickers} ≠ requested "
                                f"{self.tickers}")
            elif self.start is not None and (
                cached.index[0] > pd.Timestamp(self.start)
            ):
                stale_reason = (f"cache starts {cached.index[0].date()} "
                                f"after requested {self.start}")
            elif self.end is not None and (
                cached.index[-1] < pd.Timestamp(self.end) - pd.Timedelta(days=7)
            ):
                stale_reason = (f"cache ends {cached.index[-1].date()} "
                                f"before requested {self.end}")

            if stale_reason is None:
                self.prices = cached[self.tickers]
                print(f"✓ cached → {_rel(path)}")
                self._write_sample(name)
                return self

            print(f"⟳ cache stale ({stale_reason}); re-fetching")

        self.fetch()
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.prices.to_parquet(path)
        print(f"✓ cached → {_rel(path)}")
        self._write_sample(name)
        return self

    # ── derived quantities ────────────────────────────────────────────
    @property
    def returns(self):
        return np.log(self.prices / self.prices.shift(1)).dropna()

    @property
    def mu(self):
        return self.returns.mean().values

    @property
    def Sigma(self):
        return self.returns.cov().values

    @property
    def n(self):
        return len(self.tickers)

    @property
    def colors(self):
        return [_ASSET_CYCLE[i % len(_ASSET_CYCLE)] for i in range(self.n)]

    def annualised(self):
        f = _ANN.get(self.interval, 252)
        return self.mu * f, self.Sigma * f

    # ── reporting ─────────────────────────────────────────────────────
    def summary(self):
        mu_a, Sig_a = self.annualised()
        vol_a = np.sqrt(np.diag(Sig_a))
        print(f"\n{self.n} assets, {len(self.prices)} {self.interval} obs")
        print(f"{'Ticker':<8}{'Ann.Ret':>10}{'Ann.Vol':>10}{'Sharpe':>10}")
        for i, t in enumerate(self.tickers):
            print(f"{t:<8}{mu_a[i]*100:>9.2f}%{vol_a[i]*100:>9.2f}%"
                  f"{mu_a[i]/vol_a[i]:>10.3f}")

    # ── plot panels (single-Axes) ─────────────────────────────────────
    def _panel_prices(self, ax):
        normed = self.prices / self.prices.iloc[0]
        for t, c in zip(self.tickers, self.colors):
            ax.plot(normed.index, normed[t], label=t, lw=1.6, color=c)
        ax.set_yscale("log")
        ax.legend(fontsize=9, ncol=min(4, self.n))
        title(ax, "Normalised prices",
              "Log scale, base 1 at the first trading day of the window")
        ax.set_ylabel("Price (base=1, log)"); ax.set_xlabel("Date")
        ax.tick_params(axis='x', rotation=30)

    def _panel_hist(self, ax):
        for t, c in zip(self.tickers, self.colors):
            ax.hist(self.returns[t] * 100, bins=60, alpha=0.5,
                    color=c, label=t, edgecolor=PALETTE["charcoal"], linewidth=0.3)
        ax.axvline(0, color=PALETTE["charcoal"], lw=0.6)
        ax.legend(fontsize=8)
        title(ax, "Return distribution",
              "Histogram of daily log-returns per asset, overlaid")
        ax.set_xlabel("Return (%)"); ax.set_ylabel("Count")

    def _panel_rolling_vol(self, ax, window=30):
        roll_vol = (self.returns.rolling(window).std()
                    * np.sqrt(_ANN.get(self.interval, 252)) * 100)
        for t, c in zip(self.tickers, self.colors):
            ax.plot(roll_vol.index, roll_vol[t], lw=1.2, color=c, label=t)
        ax.legend(fontsize=9, ncol=min(4, self.n))
        title(ax, "Rolling volatility",
              f"{window}-period rolling standard deviation, annualised")
        ax.set_ylabel("Volatility (%)"); ax.set_xlabel("Date")
        ax.tick_params(axis='x', rotation=30)

    def _panel_correlation(self, ax):
        # Seriate by leading eigenvector so highly-correlated tickers
        # cluster near the diagonal (decay outward from top-left).
        corr_df = self.returns.corr()
        eigvals, eigvecs = np.linalg.eigh(corr_df.values)
        leading = eigvecs[:, -1]
        if leading.sum() < 0:
            leading = -leading
        order = np.argsort(-leading)
        ordered = [self.tickers[i] for i in order]
        corr = corr_df.loc[ordered, ordered].values
        n = self.n

        im = ax.imshow(corr, cmap=_CORR_CMAP, vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.55, pad=0.015, fraction=0.035)
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(ordered); ax.set_yticklabels(ordered)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(corr[i, j]) > 0.7
                              else PALETTE["charcoal"],
                        fontsize=9)
        title(ax, "Return correlation",
              "Seriated by leading eigenvector — high correlations cluster near the diagonal")
        ax.grid(False)

    def _panel_risk_return(self, ax):
        mu_a, Sig_a = self.annualised()
        vol_a = np.sqrt(np.diag(Sig_a)) * 100
        ax.scatter(vol_a, mu_a * 100, s=140, c=self.colors,
                   edgecolors=PALETTE["charcoal"], linewidth=0.8,
                   alpha=0.65, zorder=5)
        rng = np.random.default_rng(0)
        for k, t in enumerate(self.tickers):
            dx = rng.uniform(-3, 9)
            dy = rng.uniform(-6, 8)
            ax.annotate(t, (vol_a[k], mu_a[k] * 100),
                        xytext=(7 + dx, 5 + dy), textcoords="offset points",
                        fontsize=10, color=PALETTE["charcoal"])
        ax.set_xlabel("Ann. volatility (%)"); ax.set_ylabel("Ann. return (%)")
        title(ax, "Risk–return",
              "Annualised mean vs volatility per asset")

    def _panel_returns(self, ax):
        rets = self.returns * 100
        for t, c in zip(self.tickers, self.colors):
            ax.plot(rets.index, rets[t], lw=0.7, color=c,
                    alpha=0.55, label=t)
        ax.axhline(0, color=PALETTE["charcoal"], lw=0.5)
        ax.legend(fontsize=8, ncol=min(4, self.n))
        title(ax, "Log-returns",
              "Daily log-returns (%) per asset, overlaid")
        ax.set_ylabel("Return (%)"); ax.set_xlabel("Date")
        ax.tick_params(axis='x', rotation=30)

    # ── public plotting API ───────────────────────────────────────────
    def plot(self, save=False, individual=False, name="overview", figsize=None,
             save_dir=None):
        """
        Plot 6-panel EDA overview (3x2 grid).

        Layout:
            [0,0] Prices            [0,1] Returns (overlay)
            [1,0] Rolling vol       [1,1] Histogram
            [2,0] Correlation       [2,1] Risk–return

        Parameters
        ----------
        save : bool
            If True, save plot(s) as PDF.
        individual : bool
            If True, render each panel as its own figure. Else a 3x2 grid.
        name : str
            Filename prefix for saved plots.
        figsize : tuple, optional
            In individual mode applies to each panel (default (8, 5));
            in grid mode it is the overall figure (default (14, 16)).
        save_dir : Path | str, optional
            Where to write the PDFs. Defaults to `<repo>/plots/eda`.
            Notebooks running on Colab should pass
            `scripts.colab.out_dir("plots", "eda")` so outputs land in Drive.
        """
        _apply_style()

        single_panels = [
            ("prices",      self._panel_prices,      (0, 0)),
            ("returns",     self._panel_returns,     (0, 1)),
            ("rolling_vol", self._panel_rolling_vol, (1, 0)),
            ("histogram",   self._panel_hist,        (1, 1)),
            ("correlation", self._panel_correlation, (2, 0)),
            ("risk_return", self._panel_risk_return, (2, 1)),
        ]

        out_d = Path(save_dir) if save_dir is not None else _PLOTS_DIR
        if save:
            out_d.mkdir(parents=True, exist_ok=True)

        if individual:
            panel_size = figsize or (8, 5)
            for panel_name, draw, _ in single_panels:
                fig, ax = plt.subplots(figsize=panel_size)
                draw(ax)
                plt.tight_layout()
                if save:
                    path = out_d / f"{name}_{panel_name}.pdf"
                    fig.savefig(path, bbox_inches="tight")
                    print(f"✓ saved → {_rel(path)}")
                plt.show()
        else:
            fig = plt.figure(figsize=figsize or (14, 16))
            gs = GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.25)
            fig.suptitle(f"Dataset overview — {name}", fontweight="bold",
                         fontsize=15, color=PALETTE["charcoal"],
                         x=0.02, ha="left", y=0.995)

            for panel_name, draw, (r, c) in single_panels:
                ax = fig.add_subplot(gs[r, c])
                draw(ax)

            if save:
                path = out_d / f"{name}_overview.pdf"
                fig.savefig(path, bbox_inches="tight")
                print(f"✓ saved → {_rel(path)}")
            plt.show()

    def __repr__(self):
        obs = len(self.prices) if self.prices is not None else 0
        return f"SNP({self.tickers}, obs={obs})"