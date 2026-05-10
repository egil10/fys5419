"""
baskets.py — Thematic stock baskets for portfolio experiments.

Each basket is 4 tickers, designed to expose QAOA/QNN to different
correlation and return regimes:

    mag7        Mega-cap tech (high correlation, low dispersion)
    quantum     Quantum-computing pure-plays (high vol, speculative)
    quantum_big Big-tech with quantum exposure (institutional angle)
    diversified Cross-sector mix (low correlation, varied risk)
"""

BASKETS = {
    "mag7": {
        "tickers": ["AAPL", "MSFT", "GOOGL", "META"],
        "description": "Mega-cap tech — high correlation, low dispersion",
    },
    "quantum": {
        "tickers": ["IONQ", "RGTI", "QBTS", "QUBT"],
        "description": "Quantum-computing pure-plays — speculative, high volatility",
    },
    "quantum_big": {
        "tickers": ["IBM", "HON", "NVDA", "AMZN"],
        "description": "Established firms with quantum exposure",
    },
    "diversified": {
        "tickers": ["JPM", "XOM", "JNJ", "WMT"],
        "description": "Cross-sector mix — finance, energy, healthcare, retail",
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