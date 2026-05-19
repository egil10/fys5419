"""
data.py — Single responsibility: raw prices → (mu, Sigma).

One public function, one frozen dataclass. No optimisation logic.

Usage
-----
    from scripts.data import load_returns

    r = load_returns(["AAPL", "MSFT", "GOOGL", "AMZN"],
                     "2023-01-01", "2025-12-31")
    r.mu, r.Sigma, r.tickers, r.prices
"""
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

from scripts.snp import SNP


@dataclass(frozen=True)
class Returns:
    """Frozen container for everything downstream solvers need from the data layer."""
    mu:      np.ndarray   # (n,) daily log-return means
    Sigma:   np.ndarray   # (n, n) daily log-return covariance
    tickers: tuple        # asset names, in column order
    prices:  Any          # pd.DataFrame of adjusted-close prices

    @property
    def n(self) -> int:
        return len(self.tickers)

    def annualised(self, freq: int = 252):
        return self.mu * freq, self.Sigma * freq


def load_returns(
    tickers: Sequence[str],
    start: str,
    end: str,
    cache_name: str | None = None,
) -> Returns:
    """Fetch (or load from cache) prices and return a Returns object.

    Thin wrapper over scripts.snp.SNP — keeps the data layer dumb.
    """
    snp = SNP(list(tickers), start, end).cached_fetch(name=cache_name)
    return Returns(
        mu=snp.mu,
        Sigma=snp.Sigma,
        tickers=tuple(snp.tickers),
        prices=snp.prices,
    )
