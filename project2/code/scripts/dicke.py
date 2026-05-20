"""
dicke.py — Dicke state preparation.

The Dicke state on n qubits with Hamming weight K is the equal
superposition over all C(n, K) basis states of weight exactly K:

    |D^n_K> = (1 / sqrt(C(n, K))) * sum_{|x|=K} |x>.                    (1)

It is the symmetric ground state of the XY-ring mixer in the constrained
subspace, so a QAOA that combines |D^n_K> initial preparation with the
XY ring mixer (see `scripts.xy_mixer`) lives entirely on the
weight-K manifold — exact budget feasibility throughout the dynamics, not
just at the final measurement.

For small n (which is the regime we benchmark in notebook 08 — n <= 12)
we build |D^n_K> by direct enumeration of all weight-K bitstrings.
A scalable Bärtschi-Eidenbenz preparation circuit would be needed for
n >> 20; not relevant here.
"""
from __future__ import annotations
from math import comb
import numpy as np


def dicke_state(n: int, K: int) -> np.ndarray:
    """Return |D^n_K> as a length-2^n statevector (qubit 0 = MSB).

    Parameters
    ----------
    n : int    number of qubits
    K : int    target Hamming weight (0 <= K <= n)

    Returns
    -------
    psi : np.ndarray of shape (2^n,), complex
        Equal-amplitude superposition over weight-K basis states.

    Raises
    ------
    ValueError if K is outside [0, n].
    """
    if not (0 <= K <= n):
        raise ValueError(f"K={K} out of range [0, {n}]")
    dim = 1 << n
    psi = np.zeros(dim, dtype=complex)
    amp = 1.0 / np.sqrt(comb(n, K))
    for k in range(dim):
        if bin(k).count('1') == K:
            psi[k] = amp
    return psi
