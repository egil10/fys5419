# Part b) — One-Qubit Hamiltonian: Eigenvalue Problem (10 pt)

## Objective

Solve the eigenvalue problem for a one-qubit Hamiltonian $H(\lambda) = H_0 + \lambda H_I$, decompose it via Pauli matrices, and study the behavior of the eigenstates as functions of the interaction strength $\lambda$.

---

## Hamiltonian Setup

We consider the $2 \times 2$ Hamiltonian with parameters $E_1 = 0$, $E_2 = 4$, $V_{11} = -V_{22} = 3$, $V_{12} = V_{21} = 0.2$:

$$H_0 = \begin{pmatrix} 0 & 0 \\ 0 & 4 \end{pmatrix}, \quad H_I = \begin{pmatrix} 3 & 0.2 \\ 0.2 & -3 \end{pmatrix}.$$

### Pauli Decomposition

Any $2 \times 2$ Hermitian matrix can be decomposed in the Pauli basis $\{I, \sigma_x, \sigma_y, \sigma_z\}$:

$$H_0 = \mathcal{E} I + \Omega \sigma_z, \quad \mathcal{E} = \frac{E_1 + E_2}{2} = 2.0, \quad \Omega = \frac{E_1 - E_2}{2} = -2.0.$$

$$H_I = c I + \omega_z \sigma_z + \omega_x \sigma_x, \quad c = 0.0, \quad \omega_z = 3.0, \quad \omega_x = 0.2.$$

The full Hamiltonian as a function of $\lambda$ is therefore:

$$H(\lambda) = (\mathcal{E} + \lambda c) I + (\Omega + \lambda \omega_z) \sigma_z + (\lambda \omega_x) \sigma_x.$$

The decomposition was verified numerically: the reconstructed matrix matches the original to machine precision.

## Eigenvalue Solution

### Analytical

The characteristic equation for $H(\lambda)$ yields:

$$E_\pm(\lambda) = 2 \pm \sqrt{4 - 12\lambda + 9.04\lambda^2},$$

where we used the explicit matrix elements $H_{11} = 3\lambda$, $H_{22} = 4 - 3\lambda$, $H_{12} = 0.2\lambda$.

### Numerical

The eigenvalues are computed via `numpy.linalg.eigh` over a dense grid of $\lambda \in [0, 1]$ (101 points). The discrepancy between analytical and numerical eigenvalues is at most $\sim 10^{-15}$, confirming both approaches.

![Eigenvalue spectrum](../plots/part-b_eigenvalues.pdf)

## Eigenvector Characterization

The ground state composition is tracked by computing $|\langle 0 | \psi_0 \rangle|^2$ and $|\langle 1 | \psi_0 \rangle|^2$ as functions of $\lambda$:

| $\lambda$ | $\text{Prob}(\|0\rangle)$ | $\text{Prob}(\|1\rangle)$ | Dominant State |
|-----------|--------------------------|--------------------------|----------------|
| $0.0$ | 100.0% | 0.0% | $\|0\rangle$ |
| $2/3$ | ~50% | ~50% | Mixed |
| $1.0$ | ~1% | ~99% | $\|1\rangle$ |

![Ground state mixing](../plots/part-b_eigenvector_mixing.pdf)

## Discussion

1. **Avoided Crossing.** The system exhibits an *avoided crossing* at $\lambda = 2/3$. At this point, the diagonal elements cross ($3\lambda = 4 - 3\lambda \Rightarrow \lambda = 2/3$), but the off-diagonal coupling $V_{12} = 0.2$ prevents an actual degeneracy. The gap at the crossing is determined by the coupling strength: $\Delta E \approx 2|V_{12}|\lambda = 2 \times 0.2 \times 2/3 \approx 0.27$.

2. **State Character Inversion.** Below $\lambda = 2/3$, the ground state is predominantly $|0\rangle$ (the unperturbed ground state). Above $\lambda = 2/3$, the interaction reverses the energy ordering, and the ground state becomes dominated by $|1\rangle$ — the state that was originally the excited state. At $\lambda = 1$, the $|0\rangle$ component is only $\sim 1\%$.

3. **Physical Interpretation.** The coupling strength $V_{12} = 0.2$ is small relative to the diagonal interactions ($V_{11} = 3$), which means the mixing near the crossing is narrow. A larger coupling would produce a smoother, wider transition. This behavior — where level crossings are "avoided" due to off-diagonal perturbations — is a universal feature of quantum mechanics and plays a central role in adiabatic quantum computation.
