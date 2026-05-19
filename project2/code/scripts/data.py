"""
data.py — Single responsibility: raw prices -> (mu, Sigma).

Two public entry points:

    load_returns(tickers, start, end)
        Fetch (or load from cache) a price panel and return a `Returns`
        object with mu, Sigma, tickers, and the raw prices DataFrame.

    load_universe()
        THE canonical 16-asset dataset for this project (Mag7 + quantum +
        quantum_big + anti, 2023-01-01 to 2025-12-31). Cached to
        `results/universe_16.npz` so every notebook gets the *same* mu
        and Sigma — no per-notebook recomputation, no drift.

Notebooks 02, 03, 04, 06 should call `load_universe()` and use all 16
assets. Notebook 05 should `load_universe()` then *subset* by index — that
keeps the n=16 endpoint of the scaling sweep numerically identical to the
runs in the other notebooks.

No optimisation logic lives here.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from scripts.snp     import SNP
from scripts.baskets import get as _get_basket


_ROOT = Path(__file__).resolve().parent.parent
_UNIVERSE_CACHE = _ROOT / "results" / "universe_16.npz"

#: The four baskets that make up the canonical 16-asset universe, in order.
UNIVERSE_BASKETS = ("mag7", "quantum", "quantum_big", "anti")
UNIVERSE_START   = "2023-01-01"
UNIVERSE_END     = "2025-12-31"


@dataclass(frozen=True)
class Returns:
    """Frozen container for everything downstream solvers need from the data layer."""
    mu:      np.ndarray   # (n,) daily log-return means
    Sigma:   np.ndarray   # (n, n) daily log-return covariance
    tickers: tuple        # asset names, in column order
    prices:  Any          # pd.DataFrame of adjusted-close prices (None if loaded from npz)

    @property
    def n(self) -> int:
        return len(self.tickers)

    def annualised(self, freq: int = 252):
        return self.mu * freq, self.Sigma * freq

    def subset(self, idx) -> "Returns":
        """Return a Returns over the subset of assets indexed by `idx` (iterable of ints)."""
        idx = np.asarray(list(idx), dtype=int)
        return Returns(
            mu=self.mu[idx],
            Sigma=self.Sigma[np.ix_(idx, idx)],
            tickers=tuple(self.tickers[i] for i in idx),
            prices=None if self.prices is None else self.prices.iloc[:, idx],
        )


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


def load_universe(force: bool = False) -> Returns:
    """Load the canonical 16-asset universe used across notebooks 02-07.

    The combined ticker list is Mag7 + quantum + quantum_big + anti, in
    that order, with returns computed from 2023-01-01 to 2025-12-31.

    On first call: fetches prices, computes mu and Sigma, caches them to
    `results/universe_16.npz`. Subsequent calls load from the npz so every
    notebook sees the *same* numerical mu and Sigma.

    Pass `force=True` to recompute (e.g. after editing `baskets.py`).
    """
    if _UNIVERSE_CACHE.exists() and not force:
        d = np.load(_UNIVERSE_CACHE, allow_pickle=True)
        return Returns(
            mu=d["mu"],
            Sigma=d["Sigma"],
            tickers=tuple(str(t) for t in d["tickers"]),
            prices=None,
        )

    tickers: list[str] = []
    for basket in UNIVERSE_BASKETS:
        tickers.extend(_get_basket(basket))

    r = load_returns(tickers, UNIVERSE_START, UNIVERSE_END,
                     cache_name="universe_16")
    _UNIVERSE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        _UNIVERSE_CACHE,
        mu=r.mu,
        Sigma=r.Sigma,
        tickers=np.array(r.tickers),
    )
    return r
