# Work Log - Project 1

This logbook records the progress, algorithms tested, and numerical observations during the implementation of Project 1.

## 2026-03-03
### Progress
- Initial setup of project structure (`code/scripts`, `code/plots`, `docs`).
- Implemented **Part (a)**: Verified Bell state entanglement entropy $S \approx 1.0$.
- Implemented **Part (b) & (c)**: One-qubit Hamiltonian. Successfully decomposed into Pauli matrices. VQE converged to exact ground state with error $< 10^{-10}$.
- Implemented **Part (d) & (e)**: Two-qubit system. Analyzed avoided crossing at $\lambda \approx 0.67$. Entropy peak observed at transition. VQE error $\sim 10^{-9}$.
- Implemented **Part (f) & (g)**: Lipkin model for $J=1$ and $J=2$. Mapped quasispin to Pauli strings on 2-qubit subspace for $J=1$.

### Algorithms Tested
- **Optimizer**: Tested `BFGS` vs `L-BFGS-B`. `L-BFGS-B` showed better stability for the multi-parameter two-qubit ansatz.
- **Ansatz**: Used a hardware-efficient $R_y$-CNOT-$R_y$ ansatz for 2-qubit systems. It proved sufficient to reach the ground state without requiring high depth.

### Time Spent
- Research & Hamiltonian mapping: ~2 hours.
- Implementation & Debugging: ~3 hours.
- Documentation & Plotting: ~1 hour.

### Notes
- Loss of precision was minimal in double precision (NumPy default).
- The "avoided crossing" in Part (d) provides a sensitive test for the VQE optimizer's ability to maintain the global minimum versus local traps.
