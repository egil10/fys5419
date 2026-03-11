# Results — FYS5419 Project 1: A VQE Study of the Lipkin Model

This document indexes the results write-ups, structured to match the report's Results section.

---

## Structure

The results are organized in four sections, mirroring the report:

| # | Section | Parts | File |
|---|---------|-------|------|
| 1 | [Single-Qubit Operations and Bell State Entanglement](part-a.md) | a | `part-a.md` |
| 2 | [Two-Level System: Exact Diagonalization vs. VQE](part-b.md) | b, c | `part-b.md` |
| 3 | [Two-Qubit System and Interaction-Driven Entanglement](part-d.md) | d, e | `part-d.md` |
| 4 | [The Lipkin Model](part-f.md) | f, g | `part-f.md` |

Each file is written as a **results section** — it applies the methods described in the report, presents numerical outcomes and figures, and provides physical interpretation. Theory and algorithm descriptions are in the Methods section of the report, not repeated here.

## Plot Inventory

All figures in [`../plots/`](../plots/):

| File | Section | Content |
|------|---------|---------|
| `part-a_measurement_dist.pdf` | 1 | Bell state measurement histogram |
| `part-b_eigenvalues.pdf` | 2 | 1-qubit eigenvalue spectrum vs $\lambda$ |
| `part-b_eigenvector_mixing.pdf` | 2 | Ground state composition |
| `part-c_vqe_full_spectrum.pdf` | 2 | VQE vs exact (GS + ES) |
| `part-c_vqe_spectrum_error.pdf` | 2 | VQE error (log scale) |
| `part-d_eigenvalues.pdf` | 3 | 2-qubit eigenvalue spectrum |
| `part-d_entropy.pdf` | 3 | Entanglement entropy (all states) |
| `part-e_vqe_comparison.pdf` | 3 | 2-qubit VQE benchmark |
| `part-e_vqe_precision.pdf` | 3 | 2-qubit VQE error |
| `part-f_j1.pdf` | 4 | Lipkin $J=1$ spectrum + HF |
| `part-f_j2.pdf` | 4 | Lipkin $J=2$ spectrum + HF |
| `part-f_scaling.pdf` | 4 | Energy per particle scaling |
| `part-g_vqe_manybody.pdf` | 4 | Lipkin VQE vs exact |
| `part-g_vqe_precision.pdf` | 4 | Lipkin VQE error |
