"""
portfolio.py — Mean-variance portfolio problem container.

A small data class that holds (mu, Sigma, lam, A, K) and provides
the cost function and brute-force baseline for testing/validation.
"""
import numpy as np
from itertools import product


class Portfolio:
    """
    Mean-variance portfolio with budget constraint.

    Cost:
        C(x) = -μᵀx + λ xᵀΣx + A (Σ x_i - K)²

    Parameters
    ----------
    mu      : (n,) expected returns
    Sigma   : (n,n) covariance matrix
    lam     : risk-aversion parameter
    A       : budget-penalty coefficient
    K       : target portfolio size (number of selected assets)
    tickers : list of asset names (optional, used for pretty printing)
    """

    def __init__(self, mu, Sigma, lam, A, K, tickers=None):
        self.mu = np.asarray(mu)
        self.Sigma = np.asarray(Sigma)
        self.lam = float(lam)
        self.A = float(A)
        self.K = int(K)
        self.n = len(self.mu)
        self.tickers = tickers or [f"Asset {i}" for i in range(self.n)]

    def cost(self, x):
        """Mean-variance cost C(x) for a binary vector x."""
        x = np.asarray(x, dtype=float)
        return float(-self.mu @ x
                     + self.lam * x @ self.Sigma @ x
                     + self.A * (x.sum() - self.K) ** 2)

    def brute_force(self):
        """
        Enumerate all 2^n portfolios and return the best feasible (budget=K) one.

        Returns
        -------
        dict with: x, cost, bitstring, all_valid (sorted list of all budget-K solutions)
        """
        best_cost, best_x = np.inf, None
        valid = []
        for bits in product([0, 1], repeat=self.n):
            x = np.array(bits)
            c = self.cost(x)
            if x.sum() == self.K:
                valid.append({
                    "x":         x,
                    "cost":      c,
                    "bitstring": "".join(map(str, x)),
                    "tickers":   [self.tickers[i] for i in range(self.n) if x[i]],
                })
                if c < best_cost:
                    best_cost, best_x = c, x
        valid.sort(key=lambda r: r["cost"])
        return {
            "x":         best_x,
            "cost":      best_cost,
            "bitstring": "".join(map(str, best_x)),
            "tickers":   [self.tickers[i] for i in range(self.n) if best_x[i]],
            "all_valid": valid,
        }

    def __repr__(self):
        return f"Portfolio(n={self.n}, K={self.K}, λ={self.lam}, A={self.A})"