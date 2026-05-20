"""
baskets.py — Thematic stock baskets for portfolio experiments.

Each basket is 4 tickers, designed to expose QAOA/QNN to different
correlation and return regimes:

    mag7         Mega-cap tech (high correlation, low dispersion)
    quantum      Quantum-computing pure-plays (high vol, speculative)
    quantum_big  Big-tech with quantum exposure (institutional angle)
    anti         Macro-opposed: oil, airline, gold miner, bank

The date window for every basket is the project-wide universe window
defined in `scripts.data.UNIVERSE_START` / `UNIVERSE_END` — there is one
source of truth for the data window, not four.
"""

BASKETS = {
    "mag7": {
        "tickers":     ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "description": "Mega-cap tech — high correlation, low dispersion",
    },
    "quantum": {
        "tickers":     ["IONQ", "RGTI", "QBTS", "QUBT"],
        "description": "Quantum-computing pure-plays — speculative, high vol",
    },
    "quantum_big": {
        "tickers":     ["IBM", "HON", "ACN", "NVDA"],
        "description": "Established firms with quantum exposure",
    },
    "anti": {
        "tickers":     ["XOM", "DAL", "NEM", "JPM"],
        "description": "Macro-opposed — oil producer, airline, gold miner, bank",
    },
}


def get(name):
    """Return the ticker list for a named basket."""
    return BASKETS[name]["tickers"]


def names():
    """Return all basket names."""
    return list(BASKETS.keys())


def describe(name):
    """Print basket name, description, and tickers."""
    b = BASKETS[name]
    print(f"{name:12s} — {b['description']}")
    print(f"             tickers: {b['tickers']}")
