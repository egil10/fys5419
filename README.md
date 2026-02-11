# FYS5419 -- Quantum Computing & Quantum Machine Learning

> *"Nature isn't classical, dammit, and if you want to make a simulation of nature, you'd better make it quantum mechanical."*
> -- Richard Feynman

Welcome. This is where qubits go to work.

This repo contains all code, notebooks, and reports for **FYS5419** at the University of Oslo, Spring 2026. The course sits right at the intersection of quantum physics, computer science, and machine learning -- and honestly, it's one of the most exciting corners of modern science.

---

## The Projects

| # | Title | What's Going On | Status |
|---|-------|-----------------|--------|
| 1 | **A VQE Study of the Lipkin Model** | Building variational quantum eigensolvers from scratch, hunting for ground states of nuclear Hamiltonians, and watching entanglement do its thing. | In Progress |
| 2 | **QAOA** | Quantum Approximate Optimization Algorithm -- details TBD. | Upcoming |

---

## Repository Structure

```
fys5419/
|
|-- project1/
|   |-- project1.ipynb                                        # Full project specification
|   |-- FYS5419 PROJECT 1 - A VQE Study of the Lipkin Interaction.pdf
|   |-- code/
|       |-- notebooks/
|       |   |-- part-a.ipynb      # Qubit basics, gates, Bell states, entropy
|       |   |-- part-b.ipynb      # One-qubit Hamiltonian (exact solver)
|       |   |-- part-c.ipynb      # VQE for one-qubit system
|       |   |-- part-d.ipynb      # Two-qubit entanglement & von Neumann entropy
|       |   |-- part-e.ipynb      # VQE for two-qubit system
|       |   |-- part-f.ipynb      # Lipkin model -- classical diagonalization
|       |   |-- part-g.ipynb      # VQE for the Lipkin model
|       |-- scripts/              # Reusable Python modules
|       |-- data/
|       |   |-- raw/              # Raw input data
|       |   |-- processed/        # Processed outputs
|       |-- plots/                # Figures and visualizations
|
|-- project2/
|   |-- project2.txt
|   |-- FYS5419 PROJECT 2 - QOAO.pdf
|   |-- code/
|       |-- notebooks/
|       |-- scripts/
|       |-- data/
|       |-- plots/
|
|-- README.md
```

---

## What This Course Covers

The big picture: classical computers hit a wall when simulating quantum systems. Quantum computers don't. This course teaches you to actually build and run the algorithms that make that possible.

**Core topics:**
- Quantum gates, circuits, and measurement -- the building blocks
- Variational Quantum Eigensolver (VQE) -- the workhorse of near-term quantum computing
- Entanglement, density matrices, and von Neumann entropy -- the physics that makes it all non-trivial
- Quantum noise and error -- because real hardware isn't perfect
- Quantum machine learning -- applying QML to classical and quantum data

**By the end, the goal is to:**
- Design quantum circuits for many-particle systems
- Run algorithms on actual quantum hardware
- Know when quantum beats classical (and when it doesn't)
- Have a working intuition for entanglement and decoherence

---

## Prerequisites

This isn't a beginner course. It assumes solid ground in:

- **Quantum Mechanics** (FYS3110 / FYS4480)
- **Machine Learning** (FYS-STK4155)
- **Linear Algebra** -- you'll be living in Hilbert spaces
- **Python** -- NumPy, SciPy, and Matplotlib are your daily drivers

---

## Resources

- [Course page at UiO](https://www.uio.no/studier/emner/matnat/fys/FYS5419/)
- [Lipkin model reference -- Phys. Rev. C 106, 024319](https://journals.aps.org/prc/pdf/10.1103/PhysRevC.106.024319)
- IBM Quantum / cloud platforms for real hardware access

---

<p align="center">
  <em>From superposition to solutions -- one qubit at a time.</em>
</p>