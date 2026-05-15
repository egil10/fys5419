"""
baskets.py — Thematic stock baskets for portfolio experiments.

Each basket is 4 tickers, designed to expose QAOA/QNN to different
correlation and return regimes:

    mag7        Mega-cap tech (high correlation, low dispersion)
    quantum     Quantum-computing pure-plays (high vol, speculative)
    quantum_big Big-tech with quantum exposure (institutional angle)
    diversified Cross-sector mix (low correlation, varied risk)
"""

START = "2023-01-01"
END = "2025-12-31"

BASKETS = {
    "mag7": {
        "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
        "start": START,
        "end": END,
        "description": "Mega-cap tech — high correlation, low dispersion",
    },
    "quantum": {
        "tickers": ["IONQ", "RGTI", "QBTS", "QUBT"],
        "start": START,
        "end": END,
        "description": "Quantum-computing pure-plays — speculative, high volatility",
    },
    "quantum_big": {
        "tickers": ["IBM", "HON", "ACN", "NVDA"],
        "start": START,
        "end": END,
        "description": "Established firms with quantum exposure",
    },
    "anti": {
        "tickers": ["XOM", "DAL", "NEM", "JPM"],
        "start": START,
        "end": END,
        "description": "Macro-opposed basket — oil producer, airline, gold miner, and bank with offsetting sensitivities",
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

def config(name):
    """Return (tickers, start, end) for a named basket."""
    b = BASKETS[name]
    return b["tickers"], b["start"], b["end"]