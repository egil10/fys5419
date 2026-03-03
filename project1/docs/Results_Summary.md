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

## Plot Directory
All plots are stored in [code/plots/](../code/plots/) as PDFs.
- `part-b_eigenvalues.pdf`
- `part-c_vqe_comp.pdf`
- `part-c_vqe_error.pdf`
- `part-d_eigenvalues.pdf`
- `part-d_entropy.pdf`
- `part-e_vqe_comp.pdf`
- `part-f_j1.pdf` / `part-f_j2.pdf`
- `part-g_vqe_j1.pdf`
