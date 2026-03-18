# FYS5419 — Quantum Computing & Quantum Machine Learning

University of Oslo · Spring 2026

> *"Nature isn't classical, dammit, and if you want to make a simulation of nature, you'd better make it quantum mechanical."*
> — Richard Feynman

---

## Report

The full written report is available here:
**[fys5419-overleaf](https://github.com/egil10/claudoverius)**

It covers the VQE study of the Lipkin–Meshkov–Glick model in detail — from single-qubit gates through four-qubit systems — including methods, results, numerical stability analysis, and appendices on the VQE algorithm and Python environment.

---

## Projects

### [Project 1 — VQE Study of the Lipkin Model](project1/)

Implementing the Variational Quantum Eigensolver from scratch and applying it to the Lipkin–Meshkov–Glick model, a cornerstone of nuclear structure theory. Benchmarked against exact diagonalization and Qiskit, with analysis of entanglement entropy, avoided crossings, Hartree-Fock, and quantum phase transitions.

### [Project 2 — QAOA](project2/)

Quantum Approximate Optimization Algorithm.

---

## Structure

```
fys5419/
├── project1/
│   └── code/
│       ├── scripts/       # part-a.py through part-g.py
│       ├── notebooks/     # Jupyter notebook mirrors
│       ├── plots/         # Generated figures (PDF)
│       ├── results/       # Numerical output
│       └── docs/          # Per-part write-ups
├── project2/
│   └── code/
│       ├── scripts/
│       ├── notebooks/
│       └── plots/
├── lectures/              # Lecture notes and materials
└── requirements.txt
```

## Stack

Python · NumPy · SciPy · Matplotlib · Qiskit · PennyLane

---

## References

- [Course page — UiO FYS5419](https://www.uio.no/studier/emner/matnat/fys/FYS5419/)
- [Lipkin model — Phys. Rev. C 106, 024319 (2022)](https://journals.aps.org/prc/pdf/10.1103/PhysRevC.106.024319)
