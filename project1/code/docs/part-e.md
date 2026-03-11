# Part e) — VQE for the Two-Qubit Hamiltonian (15 pt)

## Objective

Apply the VQE method to find the ground state energy of the two-qubit Hamiltonian from Part d. Design the appropriate quantum circuit, implement a custom VQE solver, and compare against both exact diagonalization and Qiskit.

---

## Hamiltonian Pauli Decomposition

The two-qubit Hamiltonian $H = H_0 + \lambda H_I$ must be expressed as a sum of Pauli strings for VQE. The non-interacting part $H_0 = \text{diag}(\epsilon_{00}, \epsilon_{10}, \epsilon_{01}, \epsilon_{11})$ is decomposed as:

$$H_0 = h_0 (I \otimes I) + h_1 (Z \otimes I) + h_2 (I \otimes Z) + h_3 (Z \otimes Z),$$

where the coefficients are obtained by inverting the Pauli tensor product eigenvalue structure:

$$h_0 = \tfrac{1}{4}(\epsilon_{00} + \epsilon_{01} + \epsilon_{10} + \epsilon_{11}) = 4.0,$$
$$h_1 = \tfrac{1}{4}(\epsilon_{00} + \epsilon_{01} - \epsilon_{10} - \epsilon_{11}) = 0.25,$$
$$h_2 = \tfrac{1}{4}(\epsilon_{00} - \epsilon_{01} + \epsilon_{10} - \epsilon_{11}) = -2.75,$$
$$h_3 = \tfrac{1}{4}(\epsilon_{00} - \epsilon_{01} - \epsilon_{10} + \epsilon_{11}) = -0.5.$$

The interaction part directly decomposes as $H_I = H_x (X \otimes X) + H_z (Z \otimes Z)$. The complete Hamiltonian in Pauli form:

$$H(\lambda) = h_0 (II) + h_1 (ZI) + h_2 (IZ) + (h_3 + \lambda H_z)(ZZ) + \lambda H_x (XX).$$

## Ansatz Circuit

We use a **hardware-efficient ansatz** consisting of:

```
|0⟩ — Ry(θ₁) — ●  — Ry(θ₃) —
                |
|0⟩ — Ry(θ₂) — X  — Ry(θ₄) —
```

**Structure:** $R_y$ rotation layer → CNOT entangler → $R_y$ rotation layer.

This gives 4 variational parameters. The rationale is:
1. **$R_y$ gates** span the real subspace of each qubit. Since the Hamiltonian is real-valued, we do not need $R_z$ or $R_x$ rotations.
2. **CNOT gate** creates entanglement between the two qubits, which is necessary since the ground state can be entangled (as shown in Part d).
3. **Two layers** of rotations (before and after CNOT) provide sufficient expressibility for this 4-dimensional Hilbert space.

## Measurement Protocol

For the VQE energy computation, five expectation values are needed: $\langle II \rangle = 1$, $\langle ZI \rangle$, $\langle IZ \rangle$, $\langle ZZ \rangle$, and $\langle XX \rangle$. In the manual implementation, these are computed as:

$$\langle P_1 \otimes P_2 \rangle = \langle \psi | (P_1 \otimes P_2) | \psi \rangle,$$

using explicit matrix-vector products. On a real quantum computer, $\langle ZI \rangle$, $\langle IZ \rangle$, and $\langle ZZ \rangle$ require only computational-basis measurements (reading out individual qubits), while $\langle XX \rangle$ requires a basis rotation (applying $H$ to each qubit before measurement).

## Results

### Energy Comparison

The VQE is swept over $\lambda \in [0, 1]$ (21 points), comparing three methods:

![Two-qubit VQE comparison](../plots/part-e_vqe_comparison.pdf)

All three approaches — exact diagonalization, manual VQE, and Qiskit VQE — produce indistinguishable results.

### Precision

![VQE precision](../plots/part-e_vqe_precision.pdf)

| Method | Max Absolute Error |
|--------|-------------------|
| Manual VQE | $\sim 10^{-10}$ |
| Qiskit VQE | $\sim 10^{-10}$ |

The very small error ($\sim 10^{-10}$) indicates that the BFGS optimizer converges to the global minimum reliably for all values of $\lambda$.

## Discussion

1. **Ansatz Effectiveness.** The $R_y$–CNOT–$R_y$ ansatz is sufficient for this two-qubit system. With 4 parameters operating on a 4-dimensional Hilbert space (constrained to real states), the ansatz can represent any real two-qubit state. This means the VQE can achieve *exact* results.

2. **Optimizer Convergence.** The BFGS algorithm converges smoothly for all values of $\lambda$, starting from the zero-parameter initial guess $\theta = (0, 0, 0, 0)$ corresponding to $|00\rangle$. No random restarts were needed, suggesting a benign optimization landscape for this system.

3. **Manual vs Qiskit Consistency.** The manual implementation using explicit matrix multiplication matches Qiskit exactly. This validates that the manual code correctly handles the ansatz construction, Hamiltonian evaluation, and measurement procedure.

4. **Comparison with Part d.** The VQE energy curve follows the exact spectrum from Part d, correctly tracking the ground state through the interaction-driven level restructuring.
