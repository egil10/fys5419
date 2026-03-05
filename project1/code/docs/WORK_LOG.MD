# Work Log - Project 1

This logbook records the progress, algorithms tested, and numerical observations during the implementation of Project 1.

## 2026-03-05
### Progress
- **Part (a)**: Enhanced entropy analysis to cover ALL four Bell states (S=1 for all) and a separable product state |+⟩⊗|0⟩ (S=0). Added random seed for reproducibility.
- **Part (d)**: Extended entropy computation to ALL four eigenstates, showing that entropy jumps at the avoided crossing are a general spectral feature.
- **Part (f)**: Added explicit Pauli decomposition for both J=1 and J=2 Lipkin Hamiltonians:
  - J=1: `H = (ε/2)(Z₁+Z₂) + (V/2)(X₁X₂ - Y₁Y₂)` — verified by projecting from 4x4 Pauli to 3x3 quasispin subspace.
  - J=2: `H = (ε/2)Σᵢ Zᵢ + (V/2)Σᵢ<ⱼ(XᵢXⱼ - YᵢYⱼ)` — J=2 sector eigenvalues from 16x16 match the 5x5 quasispin matrix.
- **Part (g)**: Added triplet-sector verification, quasispin cross-validation, circuit diagrams, multiple VQE restarts, increased V points (31 vs 16), and sector-consistency plot.
- **Documentation**: Updated RESULTS_SUMMARY.MD with correct plot filenames and new results. Synced all docs.
- **Consistency**: Disabled `plt.show()` across all scripts for non-interactive batch execution.

### Algorithms Tested
- **Optimizer**: Tested `BFGS` vs `L-BFGS-B`. `L-BFGS-B` showed better stability for the multi-parameter two-qubit ansatz.
- **Ansatz**: Used a hardware-efficient Ry-CNOT-Ry ansatz for 2-qubit systems. For Lipkin, used Ry-CZ with configurable depth.
- **Random restarts**: Added multiple restarts (n=2–3) in part-g VQE to avoid local minima at large V.

### Key Numerical Results
- J=1 Pauli projection to quasispin: matrices match exactly.
- J=2: all 5 eigenvalues from 16x16 Pauli Hamiltonian match the 5x5 quasispin reference.
- VQE ground state always falls in the maximal-J sector (no spurious singlet contamination).

## 2026-03-03
### Progress
- Initial setup of project structure (`code/scripts`, `code/plots`, `docs`).
- Implemented **Part (a)**: Verified Bell state entanglement entropy S ≈ 1.0.
- Implemented **Part (b) & (c)**: One-qubit Hamiltonian. Successfully decomposed into Pauli matrices. VQE converged to exact ground state with error < 10^-10.
- Implemented **Part (d) & (e)**: Two-qubit system. Analyzed avoided crossing at λ ≈ 0.67. Entropy peak observed at transition. VQE error ~10^-9.
- Implemented **Part (f) & (g)**: Lipkin model for J=1 and J=2. Mapped quasispin to Pauli strings on 2-qubit subspace for J=1.

### Algorithms Tested
- **Optimizer**: Tested `BFGS` vs `L-BFGS-B`. `L-BFGS-B` showed better stability for the multi-parameter two-qubit ansatz.
- **Ansatz**: Used a hardware-efficient Ry-CNOT-Ry ansatz for 2-qubit systems. It proved sufficient to reach the ground state without requiring high depth.

### Time Spent
- Research & Hamiltonian mapping: ~2 hours.
- Implementation & Debugging: ~3 hours.
- Documentation & Plotting: ~1 hour.

### Notes
- Loss of precision was minimal in double precision (NumPy default).
- The "avoided crossing" in Part (d) provides a sensitive test for the VQE optimizer's ability to maintain the global minimum versus local traps.
