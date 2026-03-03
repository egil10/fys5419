# Results Summary

The implementation follows the project roadmap from qubit basics to many-body VQE.

## Key Findings

1.  **VQE Accuracy**:
    - For the **1-qubit Hamiltonian (Part c)**, VQE matches the exact ground state with minimal error.
    - For the **2-qubit Hamiltonian (Part e)**, the error is remarkably low (~10^-11), demonstrating the effectiveness of the hardware-efficient ansatz.
    - For the **Lipkin Model (Part g)**, the VQE results for $J=1$ show good agreement with exact diagonalization.

2.  **Entanglement Analysis**:
    - Transition in $\lambda$ for the 2-qubit system shows a clear "avoided crossing" behavior.
    - von Neumann entropy reaches its maximum where the interaction term balances the single-particle energies.

3.  **Lipkin Model (Part f)**:
    - Eigenvalues for $J=1$ and $J=2$ were computed over $V \in [0, 2]$.
    - Results describe the phase transition behavior as interaction strength increases.

## Numerical Stability and Precision

1.  **VQE Convergence**:
    - The use of `BFGS` and `L-BFGS-B` algorithms provided stable convergence for 1-qubit and 2-qubit systems.
    - Precision levels reached $10^{-11}$ relative error compared to exact diagonalization, which is close to the machine epsilon for double-precision float operations in NumPy.
    - Minor instabilities were observed near the level crossing ($\lambda \approx 0.67$) where the energy landscape becomes flatter, requiring careful initial parameter guesses.

2.  **Hamiltonian Decomposition**:
    - The decomposition into Pauli strings was verified by rebuilding the dense matrix and comparing with the original Hamiltonian.
    - All tests passed with `np.allclose(H, H_reconstructed)`.

## Critique and Reflections

- **Learning Outcomes**: Implementation from scratch (without high-level libraries like Qiskit/Pennylane in the core scripts) provided a deeper understanding of the VQE cost function and state vector manipulation.
- **Future Improvements**: 
    - Scaling to larger $J$ values for the Lipkin model will require more efficient sparse matrix representation or Qiskit-based circuit simulations.
    - Implementing the $W$ term in the Lipkin Hamiltonian to study its effect on ground state symmetry.

## Plot Directory
All plots are stored in [plots/](../plots/) as PDFs.
- `part-b_eigenvalues.pdf`
- `part-c_vqe_comp.pdf`
- `part-c_vqe_error.pdf`
- `part-d_eigenvalues.pdf`
- `part-d_entropy.pdf`
- `part-e_vqe_comp.pdf`
- `part-f_j1.pdf` / `part-f_j2.pdf`
- `part-g_vqe_j1.pdf`
