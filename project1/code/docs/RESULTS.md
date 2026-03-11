# Results — FYS5419 Project 1: A VQE Study of the Lipkin Model

> This document provides an index and executive summary of the results for each part of the project. Detailed write-ups are found in the individual part files linked below.

## Overview

This project follows a progressive structure, building from single-qubit quantum mechanics to many-body VQE:

| Part | Topic | Key Result | Write-up |
|------|-------|------------|----------|
| **a** | [Quantum Basics](part-a.md) | Bell states maximize entropy ($S=1$) | Gates, measurements, entropy |
| **b** | [1-Qubit Eigenvalues](part-b.md) | Avoided crossing at $\lambda = 2/3$ | Analytical + numerical |
| **c** | [1-Qubit VQE](part-c.md) | VQE error $< 10^{-10}$ | Own code + Qiskit |
| **d** | [2-Qubit Eigenvalues](part-d.md) | Entropy jump at avoided crossing | All 4 eigenstates |
| **e** | [2-Qubit VQE](part-e.md) | VQE error $< 10^{-10}$ | Manual + Qiskit |
| **f** | [Lipkin Classical](part-f.md) | Pauli decomp. + HF phase transition | $J=1$ and $J=2$ |
| **g** | [Lipkin VQE](part-g.md) | $J=1$: $10^{-11}$, $J=2$: $10^{-2}$ | Many-body VQE |

## Progression of Ideas

The project is organized as a pedagogical sequence:

1. **Parts a–c** establish the foundations: one-qubit operations, Pauli decomposition, and the VQE workflow for the simplest possible system.

2. **Parts d–e** introduce entanglement and multi-qubit physics, showing how the VQE scales to two qubits with a hardware-efficient ansatz.

3. **Parts f–g** tackle the physically motivated Lipkin model, demonstrating how nuclear physics Hamiltonians are translated to qubit representations and solved with VQE.

## Key Physics Insights

- **Entanglement is Hamiltonian-driven.** The von Neumann entropy in Parts a and d shows that entanglement emerges from the interaction terms in the Hamiltonian, not from external preparation.

- **Avoided crossings are universal.** The level-crossing behavior appears in the 1-qubit system (Part b), the 2-qubit system (Part d), and the Lipkin model (Part f) — they all share the same mechanism of off-diagonal interaction lifting degeneracies.

- **VQE ≈ exact for small systems but degrades with scale.** The VQE achieves machine precision for 1-qubit and 2-qubit systems but shows $O(10^{-2})$ errors for the 4-qubit Lipkin model, highlighting the challenges of the variational approach.

- **The Lipkin phase transition.** The critical coupling $V_c = \varepsilon/(N-1)$ marks the transition from a mean-field ground state to a correlated many-body state. HF captures this transition qualitatively but becomes quantitatively exact only in the large-$N$ limit.

## Plot Inventory

All plots are stored as PDFs in [`../plots/`](../plots/):

| Plot File | Part | Content |
|-----------|------|---------|
| `part-a_measurement_dist.pdf` | a | Bell state measurement histogram |
| `part-b_eigenvalues.pdf` | b | 1-qubit eigenvalue spectrum |
| `part-b_eigenvector_mixing.pdf` | b | Ground state composition |
| `part-c_vqe_full_spectrum.pdf` | c | VQE vs exact (GS + ES) |
| `part-c_vqe_spectrum_error.pdf` | c | VQE error (log scale) |
| `part-d_eigenvalues.pdf` | d | 2-qubit eigenvalue spectrum |
| `part-d_entropy.pdf` | d | Entanglement entropy (all states) |
| `part-e_vqe_comparison.pdf` | e | 2-qubit VQE benchmark |
| `part-e_vqe_precision.pdf` | e | 2-qubit VQE error |
| `part-f_j1.pdf` | f | Lipkin $J=1$ spectrum + HF |
| `part-f_j2.pdf` | f | Lipkin $J=2$ spectrum + HF |
| `part-f_scaling.pdf` | f | Energy per particle scaling |
| `part-g_vqe_manybody.pdf` | g | Lipkin VQE vs exact |
| `part-g_vqe_precision.pdf` | g | Lipkin VQE error |
