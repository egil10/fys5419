# FYS5419 — Quantum Computing & Machine Learning

*Can a quantum computer find the ground state of a nucleus — or the minimum-risk portfolio — faster than a classical one?*

University of Oslo · Spring 2026 · Egil Furnes

<br>

## The idea

Quantum systems are hard to simulate classically. As the number of particles grows, the Hilbert space explodes exponentially — and brute-force diagonalization becomes impossible. Quantum computers don't simulate quantum mechanics; they *are* quantum mechanics. The variational principle — find the state that minimises the energy — becomes a feedback loop between a parameterised quantum circuit and a classical optimiser. That's the Variational Quantum Eigensolver.

The same logic applies to combinatorial optimisation. A portfolio selection problem — pick *k* assets to minimise volatility — maps naturally to a QUBO. Encode the constraints into a cost Hamiltonian, evolve the quantum state through alternating layers of cost and mixer unitaries, and let the algorithm find the ground state. That's QAOA.

Both algorithms are near-term. Both are variational. Both are genuinely quantum. And both are studied here from scratch.

<br>

## Why it's interesting

**The Lipkin model is a minimal universe.** The Lipkin–Meshkov–Glick model packs the essential physics of a many-body quantum system — phase transitions, entanglement, symmetry breaking — into a Hamiltonian you can hold in your head. It's the perfect laboratory for benchmarking VQE before scaling up.

**Avoided crossings are where quantum mechanics gets weird.** Near a phase transition, energy levels repel each other instead of crossing. Entanglement entropy peaks here. Classical methods struggle precisely where quantum effects are strongest. VQE finds the ground state here without needing to diagonalise the full Hamiltonian.

**Portfolios are physics problems in disguise.** Covariance matrices and correlation functions are structurally identical. The minimum-volatility portfolio is the ground state of a quadratic energy functional. QAOA treats it accordingly — and the connection to quantum phase estimation is not accidental.

**Crisis windows test the algorithm honestly.** Covariance matrices estimated during the 2008 crash, 2020 pandemic shock, and 2022 rate spike are not well-behaved. If QAOA can find low-volatility portfolios under those conditions, it means something.

**Everything is built from scratch.** No black-box imports. The VQE loop, the ansatz, the gradient descent, the Hamiltonian construction — all explicit, all inspectable. The goal is understanding, not just results.

<br>

## Projects

**[Project 1 — VQE & the Lipkin Model](project1/)**
Variational Quantum Eigensolver applied to the Lipkin–Meshkov–Glick model. One to four qubits. Benchmarked against exact diagonalization and Qiskit. Entanglement entropy, avoided crossings, Hartree-Fock, quantum phase transitions.

**[Project 2 — QAOA for Portfolio Optimization](project2/)**
Quantum Approximate Optimization Algorithm applied to cardinality-constrained minimum-volatility portfolio selection. Problem cast as a QUBO. Tested against historical crisis windows: 2008, 2020, 2022.

<br>

## Report

The full written report lives here: **[fys5419-overleaf-2](https://github.com/egil10/fys5419-overleaf-2)**

<br>

## Stack

Python · NumPy · SciPy · Matplotlib · Qiskit · PennyLane

<br>

---

*Department of Physics · University of Oslo*
