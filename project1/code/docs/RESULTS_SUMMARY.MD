# Results Summary

The implementation follows the project roadmap from qubit basics to many-body VQE.

## Key Findings

1.  **VQE Accuracy**:
    - For the **1-qubit Hamiltonian (Part c)**, VQE matches the exact ground state with minimal error (~10^-14).
    - For the **2-qubit Hamiltonian (Part e)**, the error is remarkably low (~10^-11), demonstrating the effectiveness of the hardware-efficient ansatz.
    - For the **Lipkin Model (Part g)**, the VQE results for both J=1 and J=2 show excellent agreement with exact diagonalization, verified against both Pauli and quasispin references.

2.  **Entanglement Analysis**:
    - All four Bell states exhibit maximal von Neumann entropy S=1, while product states have S=0, regardless of superposition complexity (Part a).
    - Transition in λ for the 2-qubit system shows a clear "avoided crossing" behavior (Part d).
    - von Neumann entropy computed for **all four eigenstates** shows entropy jumps at the avoided crossing point, with similar behaviors across the spectrum (Part d).

3.  **Lipkin Model (Parts f & g)**:
    - Eigenvalues for J=1 and J=2 were computed over V ∈ [0, 1.5].
    - **Pauli decomposition** explicitly derived:
      - J=1: `H = (ε/2)(Z₁ + Z₂) + (V/2)(X₁X₂ − Y₁Y₂)`
      - J=2: `H = (ε/2) Σᵢ Zᵢ + (V/2) Σᵢ<ⱼ (XᵢXⱼ − YᵢYⱼ)`
    - Triplet-sector eigenvalues from Pauli Hamiltonian verified to match quasispin reference.
    - Hartree-Fock comparison shows phase transition at V_c = ε/(N−1).
    - VQE correctly finds the physical ground state without explicit symmetry projection.

## Numerical Stability and Precision

1.  **VQE Convergence**:
    - The use of `BFGS` algorithm with multiple random restarts provided stable convergence.
    - Precision levels reached ~10^-14 relative error compared to exact diagonalization for 1-qubit and 2-qubit systems.
    - For many-body Lipkin VQE, precision depends on V strength but remains well below 10^-4.
    - Minor instabilities near the level crossing (λ ≈ 0.67) were handled by restart strategies.

2.  **Hamiltonian Decomposition**:
    - The decomposition into Pauli strings was verified by rebuilding the dense matrix and comparing with the original Hamiltonian.
    - Triplet-sector projection verified: P·H_Pauli·P^T ≈ H_quasispin for both J=1 and J=2.

## Critique and Reflections

- **Learning Outcomes**: Implementation from scratch (without high-level libraries like Qiskit/Pennylane in the core VQE logic) provided a deeper understanding of the VQE cost function and state vector manipulation.
- **Sector Consistency**: The qubit mapping includes unphysical sectors (e.g., J=0 singlet for N=2), but the Lipkin model's ground state always lies in the maximal-J sector, so VQE finds the correct answer without projection.
- **Future Improvements**: 
    - Scaling to larger J values for the Lipkin model will require more efficient sparse matrix representation or Qiskit-based circuit simulations.
    - Implementing the W term in the Lipkin Hamiltonian to study its effect on ground state symmetry.

## Plot Directory
All plots are stored in [plots/](../plots/) as PDFs.
- `part-a_measurement_dist.pdf`
- `part-b_eigenvalues.pdf`
- `part-b_eigenvector_mixing.pdf`
- `part-c_vqe_full_spectrum.pdf`
- `part-c_vqe_spectrum_error.pdf`
- `part-d_eigenvalues.pdf`
- `part-d_entropy.pdf`
- `part-e_vqe_comparison.pdf`
- `part-e_vqe_precision.pdf`
- `part-f_j1.pdf` / `part-f_j2.pdf`
- `part-f_scaling.pdf`
- `part-g_vqe_manybody.pdf`
- `part-g_vqe_precision.pdf`
- `part-g_sector_check.pdf`
