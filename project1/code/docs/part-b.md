# Results: Two-Level System — Exact Diagonalization vs. VQE
*Covers Parts b and c*

---

## Hamiltonian Parameters

Using the parameters from the project description ($E_1 = 0$, $E_2 = 4$, $V_{11} = 3$, $V_{22} = -3$, $V_{12} = V_{21} = 0.2$), the Pauli decomposition coefficients are:

| Coefficient | Value | Expression |
|-------------|-------|------------|
| $\mathcal{E}$ | 2.0 | $(E_1 + E_2)/2$ |
| $\Omega$ | $-2.0$ | $(E_1 - E_2)/2$ |
| $c$ | 0.0 | $(V_{11} + V_{22})/2$ |
| $\omega_z$ | 3.0 | $(V_{11} - V_{22})/2$ |
| $\omega_x$ | 0.2 | $V_{12}$ |

The decomposition was verified by reconstructing the matrix from these coefficients — the match is exact to machine precision.

## Exact Eigenvalues

The eigenvalues $E_\pm(\lambda)$ are computed both analytically (from the characteristic equation) and numerically (`numpy.linalg.eigh`) over 101 values of $\lambda \in [0,1]$. The maximum discrepancy between the two methods is $\sim 10^{-15}$.

![Eigenvalue spectrum](../plots/part-b_eigenvalues.pdf)

The spectrum shows an **avoided crossing** at $\lambda = 2/3$. At this point, the diagonal matrix elements cross ($3\lambda = 4 - 3\lambda$), but the off-diagonal coupling $V_{12} = 0.2$ prevents a true degeneracy, creating a gap of $\Delta E \approx 0.27$.

## Ground State Composition

Tracking the eigenvector composition of the ground state as a function of $\lambda$:

| $\lambda$ | $\text{Prob}(\|0\rangle)$ | $\text{Prob}(\|1\rangle)$ | Character |
|-----------|--------------------------|--------------------------|-----------|
| 0.0 | 100% | 0% | Pure $\|0\rangle$ |
| 2/3 | $\sim$50% | $\sim$50% | Maximally mixed |
| 1.0 | $\sim$1% | $\sim$99% | Dominated by $\|1\rangle$ |

![Ground state mixing](../plots/part-b_eigenvector_mixing.pdf)

The ground state character inverts as $\lambda$ passes through the crossing point. Below $\lambda = 2/3$, it is predominantly $|0\rangle$; above, $|1\rangle$ dominates. This transition is smooth (not a sharp jump) because the coupling $V_{12}$ mixes the two basis states. At $\lambda = 1$, only $\sim$1% of $|0\rangle$ remains, consistent with the theoretical prediction.

---

## VQE Implementation

### Ansatz

The trial state is $|\psi(\theta)\rangle = R_y(\theta)|0\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$. This single-parameter rotation spans the full real single-qubit Hilbert space — since the Hamiltonian is real and symmetric, $R_y$ is exactly sufficient.

### Measurement

The energy is evaluated as $E(\theta) = c_0 + c_z \langle \sigma_z \rangle_\theta + c_x \langle \sigma_x \rangle_\theta$, where:
- $\langle \sigma_z \rangle$ is obtained from computational-basis measurement
- $\langle \sigma_x \rangle$ requires a Hadamard rotation before measurement (basis change to $X$-eigenbasis)

### VQE Results

The VQE is swept over $\lambda \in [0, 1]$ (31 points). Both the ground state and the excited state (accessed via $\theta_\text{opt} + \pi$) are computed.

![VQE full spectrum](../plots/part-c_vqe_full_spectrum.pdf)

The VQE markers lie exactly on the exact diagonalization curves for both eigenvalues.

### Precision

| Quantity | Max Error |
|----------|-----------|
| Ground state | $5.41 \times 10^{-11}$ |
| Excited state | Same order |

![VQE error](../plots/part-c_vqe_spectrum_error.pdf)

The error remains below $10^{-10}$ across the entire $\lambda$ range. This confirms that:
1. The $R_y$ ansatz is *exact* for this 1-qubit problem — there are no expressibility limitations.
2. The BFGS optimizer converges to the global minimum reliably.
3. The one-parameter landscape has no local minima.

### Qiskit Cross-validation

An independent VQE implementation using Qiskit's `SparsePauliOp` and `Statevector` produces identical results, validating the manual implementation.

### Discussion

This one-qubit VQE serves as a proof-of-concept: it verifies the full VQE workflow (Pauli decomposition → ansatz → measurement → optimization) in a setting where the answer is known exactly. The real challenge begins when the system is too large for a single rotation to span the Hilbert space, as we explore in the next sections.
