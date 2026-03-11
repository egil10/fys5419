# Part c) — VQE for the One-Qubit Hamiltonian (10 pt)

## Objective

Implement the Variational Quantum Eigensolver (VQE) for the one-qubit Hamiltonian from Part b, set up the necessary quantum circuit, and compare the VQE results with exact diagonalization. Write our own VQE code and compare with Qiskit.

---

## VQE Algorithm

The VQE is a hybrid quantum-classical algorithm. Its key steps are:

1. **Hamiltonian decomposition** into Pauli strings: $H(\lambda) = c_0 I + c_z \sigma_z + c_x \sigma_x$
2. **Ansatz preparation**: parameterized quantum state $|\psi(\theta)\rangle = R_y(\theta)|0\rangle$
3. **Measurement**: estimate $\langle \sigma_z \rangle$ and $\langle \sigma_x \rangle$ separately
4. **Classical optimization**: minimize $E(\theta) = c_0 + c_z \langle \sigma_z \rangle + c_x \langle \sigma_x \rangle$ over $\theta$

### Ansatz Choice

For the one-qubit system, the ansatz $|\psi(\theta)\rangle = R_y(\theta)|0\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$ is used. This is the simplest possible parameterized state and it spans the full real single-qubit Hilbert space. Since the Hamiltonian is real and symmetric, the $R_y$ rotation is sufficient to explore the entire relevant state space.

### Measurement Procedure

The expectation values are obtained by *simulated measurements*:

- $\langle \sigma_z \rangle = |\langle 0|\psi\rangle|^2 - |\langle 1|\psi\rangle|^2 = \cos^2(\theta/2) - \sin^2(\theta/2) = \cos\theta$
- $\langle \sigma_x \rangle$: apply a Hadamard gate to rotate into the $X$-eigenbasis, then measure in the computational basis. The result is $\langle \sigma_x \rangle = \sin\theta$.

These correspond exactly to what a quantum computer would do: measure in the $Z$-basis for $\langle \sigma_z \rangle$, and rotate into the $X$-basis (via $H$) before measurement for $\langle \sigma_x \rangle$.

## Implementation

Both a **manual VQE** (using NumPy matrix operations) and a **Qiskit-based VQE** (using `SparsePauliOp` and `Statevector`) were implemented. The optimizer in both cases is `scipy.optimize.minimize` with the `BFGS` method.

### Excited State

For the one-qubit system, the excited state is uniquely determined as the state orthogonal to the ground state. In the $R_y(\theta)$ parameterization, if $\theta_\text{opt}$ gives the ground state, then $\theta_\text{opt} + \pi$ gives the excited state. Both eigenvalues of the full spectrum are therefore accessible.

## Results

### Ground State and Excited State Energies

The VQE sweep over $\lambda \in [0, 1]$ (31 points) produces energies in excellent agreement with exact diagonalization:

- **Ground state max error**: $\sim 5.41 \times 10^{-11}$
- **Excited state max error**: same order of magnitude

![Full spectrum comparison](../plots/part-c_vqe_full_spectrum.pdf)

### Precision

![VQE error](../plots/part-c_vqe_spectrum_error.pdf)

The error remains below $10^{-10}$ across the entire $\lambda$ range, confirming that:
1. The $R_y$ ansatz is expressive enough for this problem (it is in fact *exact*).
2. The BFGS optimizer converges reliably to the global minimum.
3. No local minima issues exist for this one-parameter optimization landscape.

## Discussion

1. **Circuit Sufficiency.** The $R_y(\theta)|0\rangle$ ansatz is *exactly sufficient* for this one-qubit problem. The real Hamiltonian has real eigenvectors, and $R_y$ spans the full real Bloch circle. No additional gates (e.g., $R_z$) are needed.

2. **Scaling to Larger Systems.** This trivial one-parameter case serves as a validation benchmark. In larger systems, the VQE faces challenges — the optimization landscape has local minima, the ansatz may not be expressive enough, and the number of measurements scales with the number of Pauli terms. These challenges are addressed in Parts e and g.

3. **Comparison with Part b.** The VQE perfectly reproduces the eigenvalue curves and the avoided crossing behavior from Part b. This establishes the VQE as a faithful solver for this simple system and builds confidence for its application to more complex Hamiltonians.

4. **Qiskit Validation.** The manual VQE and Qiskit-based VQE produce identical results, confirming that our custom implementation correctly mimics the circuit-based simulation.
