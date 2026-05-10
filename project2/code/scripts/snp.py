"""
snp.py — Stock data loader for portfolio analysis.

Usage
-----
    from scripts.snp import SNP
    snp = SNP(["AAPL", "MSFT", "AMZN", "GOOG"], "2020-01-01", "2023-12-31")
    snp.fetch()
    snp.summary()
    snp.plot()

    mu, Sigma = snp.mu, snp.Sigma   # for portfolio class
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_ANN = {"1d": 252, "1wk": 52, "1mo": 12}


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

    # ── plots ─────────────────────────────────────────────────────────
    def plot(self, save=None):
        """Four-panel overview: prices, returns, correlation, risk-return."""
        fig, ax = plt.subplots(2, 2, figsize=(13, 8))
        fig.suptitle("Dataset Overview", fontweight="bold")

        # prices (normalised)
        (self.prices / self.prices.iloc[0]).plot(ax=ax[0, 0], lw=1.4)
        ax[0, 0].set_title("Normalised prices"); ax[0, 0].set_ylabel("Price (base=1)")
        ax[0, 0].grid(alpha=0.3)

        # log returns
        (self.returns * 100).plot(ax=ax[0, 1], lw=0.8, alpha=0.7)
        ax[0, 1].axhline(0, color="k", lw=0.6)
        ax[0, 1].set_title("Log-returns (%)"); ax[0, 1].set_ylabel("Return (%)")
        ax[0, 1].grid(alpha=0.3)

        # correlation heatmap
        corr = self.returns.corr().values
        im = ax[1, 0].imshow(corr, cmap="RdYlBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax[1, 0], shrink=0.8)
        ax[1, 0].set_xticks(range(self.n)); ax[1, 0].set_yticks(range(self.n))
        ax[1, 0].set_xticklabels(self.tickers); ax[1, 0].set_yticklabels(self.tickers)
        for i in range(self.n):
            for j in range(self.n):
                ax[1, 0].text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                              color="white" if abs(corr[i, j]) > 0.7 else "black",
                              fontsize=9)
        ax[1, 0].set_title("Correlation")

        # risk-return scatter
        mu_a, Sig_a = self.annualised()
        vol_a = np.sqrt(np.diag(Sig_a)) * 100
        ax[1, 1].scatter(vol_a, mu_a * 100, s=100, c=range(self.n), cmap="tab10",
                         edgecolors="k", zorder=5)
        for k, t in enumerate(self.tickers):
            ax[1, 1].annotate(t, (vol_a[k], mu_a[k] * 100),
                              xytext=(6, 4), textcoords="offset points")
        ax[1, 1].set_xlabel("Ann. volatility (%)"); ax[1, 1].set_ylabel("Ann. return (%)")
        ax[1, 1].set_title("Risk–return"); ax[1, 1].grid(alpha=0.3)

        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=130, bbox_inches="tight")
        plt.show()

    def __repr__(self):
        obs = len(self.prices) if self.prices is not None else 0
        return f"SNP({self.tickers}, obs={obs})"