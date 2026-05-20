"""
portfolio.py — THE one true definition of the portfolio cost.

    ┌──────────────────────────────────────────────────────────────────────┐
    │   C(x) = -mu^T x + lam x^T Sigma x + A (sum x - K)^2,  x in {0,1}^n  │
    └──────────────────────────────────────────────────────────────────────┘

Every solver — classical (`scripts.classical`) and quantum
(`scripts.qaoa`) — imports `eval_cost` (or calls `.cost(x)` on a
`PortfolioProblem`). No solver re-implements C(x). If you ever need to
change the cost function, you change it HERE, exactly once.

`Portfolio` is an alias of `PortfolioProblem` so older code that
constructs `Portfolio(mu, Sigma, lam, A, K, tickers=...)` keeps working.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


#: Canonical QUBO parameters. Every notebook should import these so the
#: problem definition is identical everywhere; if you want to vary one
#: (e.g. sweep 3 over `lam`), override locally — don't redefine the others.
DEFAULTS = {
    "lam":    2.0,
    "A":      0.5,
    "K_FRAC": 0.25,   # K = round(K_FRAC * n)
    "K_AT_16": 4,     # convenience for the n=16 canonical instance
}


@dataclass(frozen=True)
class PortfolioProblem:
    """Cardinality-constrained mean-variance problem container.

    Parameters
    ----------
    mu      : (n,)      expected returns
    Sigma   : (n, n)    covariance matrix
    lam     : float     risk-aversion coefficient
    A       : float     budget-penalty coefficient
    K       : int       target portfolio size (number of selected assets)
    tickers : sequence of str, optional
        Asset labels. Defaults to "Asset 0", "Asset 1", ...
    """
    mu:      np.ndarray
    Sigma:   np.ndarray
    lam:     float
    A:       float
    K:       int
    tickers: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        # Frozen dataclass — use object.__setattr__ to normalise inputs
        object.__setattr__(self, "mu",    np.asarray(self.mu,    dtype=float))
        object.__setattr__(self, "Sigma", np.asarray(self.Sigma, dtype=float))
        object.__setattr__(self, "lam",   float(self.lam))
        object.__setattr__(self, "A",     float(self.A))
        object.__setattr__(self, "K",     int(self.K))
        if not self.tickers:
            object.__setattr__(
                self, "tickers",
                tuple(f"Asset {i}" for i in range(self.mu.size)),
            )
        else:
            object.__setattr__(self, "tickers", tuple(self.tickers))

    @property
    def n(self) -> int:
        return int(self.mu.size)

    def cost(self, x) -> float:
        """Convenience wrapper around eval_cost(x, self)."""
        return eval_cost(x, self)

    def __repr__(self) -> str:
        return (f"PortfolioProblem(n={self.n}, K={self.K}, "
                f"lam={self.lam}, A={self.A})")


def eval_cost(x, problem: PortfolioProblem) -> float:
    """Evaluate C(x) = -mu^T x + lam x^T Sigma x + A (sum x - K)^2."""
    x = np.asarray(x, dtype=float)
    return float(
        -problem.mu @ x
        + problem.lam * x @ problem.Sigma @ x
        + problem.A * (x.sum() - problem.K) ** 2
    )


