# Results: Two-Qubit System and Interaction-Driven Entanglement
*Covers Parts d and e*

---

## Exact Diagonalization

The two-qubit Hamiltonian with parameters $H_x = 2.0$, $H_z = 3.0$, $\epsilon_{00} = 0.0$, $\epsilon_{10} = 2.5$, $\epsilon_{01} = 6.5$, $\epsilon_{11} = 7.0$ is diagonalized over $\lambda \in [0,1]$ (201 points).

![Two-qubit eigenvalue spectrum](../plots/part-d_eigenvalues.pdf)

At $\lambda = 0$, the eigenvalues are the non-interacting energies $\{0.0, 2.5, 6.5, 7.0\}$. As $\lambda$ increases:
- The $\sigma_z \otimes \sigma_z$ interaction shifts the diagonal elements, pushing $|00\rangle$ and $|11\rangle$ up while pulling $|01\rangle$ and $|10\rangle$ down.
- The $\sigma_x \otimes \sigma_x$ coupling creates off-diagonal mixing between $|00\rangle \leftrightarrow |11\rangle$ and $|01\rangle \leftrightarrow |10\rangle$, producing avoided crossings.

The block structure of the Hamiltonian (two independent $2 \times 2$ blocks) is reflected in the pairwise behavior of the energy levels.

## Entanglement Across the Spectrum

The von Neumann entropy is computed for **all four eigenstates**, not just the ground state.

![Entropy for all eigenstates](../plots/part-d_entropy.pdf)

### Key observations:

1. **$\lambda = 0$: Zero entropy.** All eigenstates are computational basis states (product states), so $S = 0$.

2. **Monotonic increase.** As $\lambda$ grows, the off-diagonal interaction mixes the basis states, generating entanglement. The entropy increases smoothly.

3. **Sharp jump at the avoided crossing.** Near the avoided crossing region, the entropy shows a rapid increase — the ground state transitions from being dominated by a single basis state to being a strongly entangled superposition. This is the most physically significant feature: a small change in $\lambda$ produces a dramatic shift in quantum correlations.

4. **Hamiltonian-driven entanglement.** The entanglement is not externally imposed. It emerges directly from the competition between the diagonal energy spacing and the off-diagonal coupling $H_x$. When the coupling becomes comparable to the energy gap, entanglement grows sharply.

---

## Two-Qubit VQE

### Pauli Decomposition

The non-interacting Hamiltonian $H_0 = \text{diag}(\epsilon_{00}, \epsilon_{10}, \epsilon_{01}, \epsilon_{11})$ is decomposed into four Pauli strings:

$$H_0 = h_0\, II + h_1\, ZI + h_2\, IZ + h_3\, ZZ,$$

with $h_0 = 4.0$, $h_1 = 0.25$, $h_2 = -2.75$, $h_3 = -0.5$. The interaction adds $(h_3 + \lambda H_z)\, ZZ + \lambda H_x\, XX$, giving 5 Pauli terms total.

### Ansatz

The hardware-efficient ansatz consists of two $R_y$ rotation layers separated by a CNOT entangler:

$$|\psi(\theta)\rangle = \big(R_y(\theta_3) \otimes R_y(\theta_4)\big) \cdot \text{CNOT} \cdot \big(R_y(\theta_1) \otimes R_y(\theta_2)\big)|00\rangle$$

This gives 4 variational parameters. The $R_y$ rotations span the real subspace of each qubit, and the CNOT provides the entanglement needed to represent correlated states. For a 4-dimensional real Hilbert space, 4 real parameters are sufficient.

### Energy Comparison

The VQE sweep over $\lambda \in [0,1]$ (21 points) compares three approaches:

![Two-qubit VQE benchmark](../plots/part-e_vqe_comparison.pdf)

All three curves (exact, manual VQE, Qiskit VQE) are indistinguishable.

### Precision

| Implementation | Max Absolute Error |
|----------------|-------------------|
| Manual VQE | $\sim 10^{-10}$ |
| Qiskit VQE | $\sim 10^{-10}$ |

![Two-qubit VQE precision](../plots/part-e_vqe_precision.pdf)

### Discussion

1. **Ansatz expressibility.** The $R_y$–CNOT–$R_y$ circuit can represent any real two-qubit state. This means the VQE achieves exact results — the error is purely from the optimizer, not the ansatz.

2. **Optimizer behavior.** BFGS converges reliably from the zero-parameter initialization $\theta = (0,0,0,0)$. No random restarts were needed, suggesting a smooth optimization landscape for this system.

3. **Manual vs Qiskit.** The exact match between the manual implementation (explicit matrix operations) and Qiskit (circuit-based statevector simulation) validates our VQE code. Both correctly handle the ansatz, Pauli measurements, and optimization loop.

4. **Connection to entanglement.** The VQE implicitly captures the entanglement structure of the ground state — as $\lambda$ increases through the crossing region, the optimal CNOT angle changes to produce the appropriate entanglement, tracked by the entropy analysis from the first part.
