"""
snp.py — Stock data loader for portfolio analysis.

Usage
-----
    from scripts.snp import SNP
    snp = SNP(["AAPL", "MSFT", "AMZN", "GOOG"], "2020-01-01", "2023-12-31")
    snp.fetch()
    snp.summary()
    snp.plot(save=True)                  # one big 4x2 grid
    snp.plot(save=True, individual=True) # one PDF per panel

    mu, Sigma = snp.mu, snp.Sigma
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data"
_PLOTS_DIR = _ROOT / "plots" / "eda"
_ANN = {"1d": 252, "1wk": 52, "1mo": 12}

# ── Editorial palette ─────────────────────────────────────────────────
PALETTE = {
    # reds
    "red":          "#E3120B",   # Economist red
    "crimson":      "#B00020",   # deep crimson
    "coral":        "#F04E45",   # soft coral
    "salmon":       "#FF7A70",   # light salmon
    # blues
    "blue":         "#005BBB",   # editorial blue
    "navy":         "#003F7D",   # deep navy
    "blue_muted":   "#4A90C2",   # muted data blue
    "sky":          "#9CC7E5",   # pale sky
    # neutrals
    "cream":        "#F7F3E8",   # warm cream background
    "ivory":        "#FFFDF7",   # near-white ivory
    "parchment":    "#E8E2D0",   # light parchment
    "warm_grey":    "#C9C3B5",   # warm grey
    "charcoal":     "#2F2F2F",   # charcoal text
    "grey":         "#6E6E6E",   # medium grey
    "grid":         "#D8D8D8",   # light gridline grey
    # accents
    "teal":         "#2A9D8F",   # muted teal
    "ochre":        "#E9A23B",   # restrained gold
    "purple":       "#6B5B95",   # muted purple
}

# Cycle for asset colours (mix of reds, blues, accents — high contrast)
_ASSET_CYCLE = [
    PALETTE["red"], PALETTE["blue"], PALETTE["teal"], PALETTE["ochre"],
    PALETTE["purple"], PALETTE["navy"], PALETTE["coral"], PALETTE["blue_muted"],
    PALETTE["crimson"], PALETTE["sky"], PALETTE["salmon"],
]

# Diverging colormap for correlations (blue → cream → red)
_CORR_CMAP = LinearSegmentedColormap.from_list(
    "editorial_div",
    [PALETTE["navy"], PALETTE["blue_muted"], PALETTE["cream"],
     PALETTE["coral"], PALETTE["crimson"]],
)


def _apply_style():
    """Apply the editorial style globally for this figure."""
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["ivory"],
        "axes.facecolor":    PALETTE["ivory"],
        "savefig.facecolor": PALETTE["ivory"],
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
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = _DATA_DIR / f"{name}.csv"
        self.prices.to_csv(path)
        print(f"✓ saved → {path}")
        return path

    @classmethod
    def load(cls, name="prices"):
        path = _DATA_DIR / f"{name}.csv" if not Path(name).exists() else Path(name)
        prices = pd.read_csv(path, index_col=0, parse_dates=True)
        obj = cls(list(prices.columns))
        obj.prices = prices
        return obj

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
        """Cycle of asset colours from the palette."""
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

    # ── plot panels (each takes an Axes, draws on it) ─────────────────
    def _panel_prices(self, ax):
        normed = self.prices / self.prices.iloc[0]
        for t, c in zip(self.tickers, self.colors):
            ax.plot(normed.index, normed[t], label=t, lw=1.6, color=c)
        ax.legend(fontsize=9, ncol=min(4, self.n))
        ax.set_title("Normalised prices")
        ax.set_ylabel("Price (base=1)"); ax.set_xlabel("Date")

    def _panel_returns(self, ax):
        rets = self.returns * 100
        for t, c in zip(self.tickers, self.colors):
            ax.plot(rets.index, rets[t], lw=0.7, alpha=0.7, color=c, label=t)
        ax.axhline(0, color=PALETTE["charcoal"], lw=0.6)
        ax.legend(fontsize=9, ncol=min(4, self.n))
        ax.set_title("Log-returns (%)")
        ax.set_ylabel("Return (%)"); ax.set_xlabel("Date")

    def _panel_cumulative(self, ax):
        cum = self.returns.cumsum() * 100
        for t, c in zip(self.tickers, self.colors):
            ax.plot(cum.index, cum[t], lw=1.6, color=c, label=t)
        ax.axhline(0, color=PALETTE["charcoal"], lw=0.6)
        ax.legend(fontsize=9, ncol=min(4, self.n))
        ax.set_title("Cumulative log-return (%)")
        ax.set_ylabel("Cumulative return (%)"); ax.set_xlabel("Date")

    def _panel_drawdown(self, ax):
        cum_price = (1 + self.returns).cumprod()
        running_max = cum_price.cummax()
        drawdown = (cum_price / running_max - 1) * 100
        for t, c in zip(self.tickers, self.colors):
            ax.plot(drawdown.index, drawdown[t], lw=1.2, color=c, label=t)
        ax.axhline(0, color=PALETTE["charcoal"], lw=0.6)
        ax.legend(fontsize=9, ncol=min(4, self.n))
        ax.set_title("Drawdown (%)")
        ax.set_ylabel("Drawdown (%)"); ax.set_xlabel("Date")

    def _panel_hist(self, ax):
        for t, c in zip(self.tickers, self.colors):
            ax.hist(self.returns[t] * 100, bins=60, alpha=0.5,
                    color=c, label=t, edgecolor=PALETTE["charcoal"], linewidth=0.3)
        ax.axvline(0, color=PALETTE["charcoal"], lw=0.6)
        ax.legend(fontsize=8)
        ax.set_title("Return distribution")
        ax.set_xlabel("Return (%)"); ax.set_ylabel("Count")

    def _panel_rolling_vol(self, ax, window=30):
        roll_vol = (self.returns.rolling(window).std()
                    * np.sqrt(_ANN.get(self.interval, 252)) * 100)
        for t, c in zip(self.tickers, self.colors):
            ax.plot(roll_vol.index, roll_vol[t], lw=1.2, color=c, label=t)
        ax.legend(fontsize=9, ncol=min(4, self.n))
        ax.set_title(f"Rolling volatility ({window}-period, annualised %)")
        ax.set_ylabel("Volatility (%)"); ax.set_xlabel("Date")

    def _panel_correlation(self, ax):
        corr = self.returns.corr().values
        im = ax.imshow(corr, cmap=_CORR_CMAP, vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(self.n)); ax.set_yticks(range(self.n))
        ax.set_xticklabels(self.tickers); ax.set_yticklabels(self.tickers)
        for i in range(self.n):
            for j in range(self.n):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        color=PALETTE["ivory"] if abs(corr[i, j]) > 0.7
                              else PALETTE["charcoal"],
                        fontsize=9)
        ax.set_title("Return correlation")
        ax.grid(False)

    def _panel_risk_return(self, ax):
        mu_a, Sig_a = self.annualised()
        vol_a = np.sqrt(np.diag(Sig_a)) * 100
        ax.scatter(vol_a, mu_a * 100, s=140, c=self.colors,
                   edgecolors=PALETTE["charcoal"], linewidth=0.8, zorder=5)
        for k, t in enumerate(self.tickers):
            ax.annotate(t, (vol_a[k], mu_a[k] * 100),
                        xytext=(7, 5), textcoords="offset points",
                        fontsize=10, color=PALETTE["charcoal"])
        ax.set_xlabel("Ann. volatility (%)"); ax.set_ylabel("Ann. return (%)")
        ax.set_title("Risk–return")

    # ── public plotting API ───────────────────────────────────────────
    def plot(self, save=False, individual=False):
        """
        Plot 8-panel EDA overview with editorial styling.

        Parameters
        ----------
        save : bool
            If True, save plot(s) as PDF to code/project2/plots/eda/.
        individual : bool
            If True, render each panel as its own figure. Else a 4x2 grid.
        """
        _apply_style()

        panels = [
            ("prices",      self._panel_prices),
            ("returns",     self._panel_returns),
            ("cumulative",  self._panel_cumulative),
            ("drawdown",    self._panel_drawdown),
            ("histogram",   self._panel_hist),
            ("rolling_vol", self._panel_rolling_vol),
            ("correlation", self._panel_correlation),
            ("risk_return", self._panel_risk_return),
        ]

        if save:
            _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

        if individual:
            for name, draw in panels:
                fig, ax = plt.subplots(figsize=(8, 5))
                draw(ax)
                plt.tight_layout()
                if save:
                    path = _PLOTS_DIR / f"{name}.pdf"
                    fig.savefig(path, bbox_inches="tight")
                    print(f"✓ saved → {path}")
                plt.show()
        else:
            fig, axes = plt.subplots(4, 2, figsize=(14, 18))
            fig.suptitle("Dataset Overview", fontweight="bold",
                         fontsize=15, color=PALETTE["charcoal"], y=0.995)
            for (name, draw), ax in zip(panels, axes.flat):
                draw(ax)
            plt.tight_layout()
            if save:
                path = _PLOTS_DIR / "overview.pdf"
                fig.savefig(path, bbox_inches="tight")
                print(f"✓ saved → {path}")
            plt.show()

    def __repr__(self):
        obs = len(self.prices) if self.prices is not None else 0
        return f"SNP({self.tickers}, obs={obs})"