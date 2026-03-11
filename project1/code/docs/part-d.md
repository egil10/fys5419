# Part d) — Two-Qubit Hamiltonian and Entanglement (15 pt)

## Objective

Extend the eigenvalue analysis to a two-qubit interacting system, compute the eigenvalues as functions of the interaction strength $\lambda$, and study the role of entanglement via the von Neumann entropy.

---

## Hamiltonian Definition

The two-qubit system consists of subsystems $A$ and $B$, with the computational basis $\{|00\rangle, |10\rangle, |01\rangle, |11\rangle\}$. The Hamiltonian is:

$$H = H_0 + \lambda H_I,$$

where $H_0 = \text{diag}(\epsilon_{00}, \epsilon_{10}, \epsilon_{01}, \epsilon_{11})$ is diagonal in the computational basis, and the interaction Hamiltonian is:

$$H_I = H_x (\sigma_x \otimes \sigma_x) + H_z (\sigma_z \otimes \sigma_z).$$

The full matrix reads:

$$H = \begin{pmatrix} \epsilon_{00} + \lambda H_z & 0 & 0 & \lambda H_x \\ 0 & \epsilon_{10} - \lambda H_z & \lambda H_x & 0 \\ 0 & \lambda H_x & \epsilon_{01} - \lambda H_z & 0 \\ \lambda H_x & 0 & 0 & \epsilon_{11} + \lambda H_z \end{pmatrix}$$

with parameters $H_x = 2.0$, $H_z = 3.0$, and non-interacting energies $\epsilon_{00} = 0.0$, $\epsilon_{10} = 2.5$, $\epsilon_{01} = 6.5$, $\epsilon_{11} = 7.0$.

Note the block structure: the $\sigma_z \otimes \sigma_z$ term modifies the diagonal, while $\sigma_x \otimes \sigma_x$ couples $|00\rangle \leftrightarrow |11\rangle$ and $|01\rangle \leftrightarrow |10\rangle$, creating two independent $2 \times 2$ blocks.

## Eigenvalue Spectrum

The eigenvalues are computed via `numpy.linalg.eigh` over $\lambda \in [0, 1]$ (201 points).

![Two-qubit eigenvalues](../plots/part-d_eigenvalues.pdf)

The spectrum reveals clear interaction-induced effects:
- At $\lambda = 0$, the eigenvalues are simply the non-interacting energies $\{0.0, 2.5, 6.5, 7.0\}$.
- As $\lambda$ increases, the $H_z$ term shifts the diagonal elements (pushing $|00\rangle$ and $|11\rangle$ up, $|01\rangle$ and $|10\rangle$ down), while $H_x$ creates off-diagonal coupling that causes avoided crossings.
- Near the crossing region, the states become strongly mixed — neither computational basis state dominates.

## Entanglement: von Neumann Entropy

For a state $|\psi\rangle = \sum_{ij} \alpha_{ij} |ij\rangle$, the reduced density matrix for subsystem $A$ is obtained by tracing over $B$:

$$\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|).$$

In practice, we reshape the state vector into a $2 \times 2$ matrix where rows correspond to subsystem $A$ and columns to $B$, then compute $\rho_A = \Psi \Psi^\dagger$, where $\Psi_{ab} = \alpha_{ab}$. The von Neumann entropy is:

$$S(\rho_A) = -\sum_i \lambda_i \log_2 \lambda_i,$$

where $\lambda_i$ are the eigenvalues of $\rho_A$.

We compute the entropy for **all four eigenstates**, not just the ground state:

![Entropy across eigenstates](../plots/part-d_entropy.pdf)

### Key Observations

1. **Zero entropy at $\lambda = 0$.** At zero interaction, all eigenstates are pure computational basis states (product states), so $S = 0$.

2. **Entropy increase with interaction.** As $\lambda$ grows, the interaction mixes the computational basis states, creating entanglement. The entropy increases monotonically for the ground state.

3. **Entropy jump at the avoided crossing.** Near the avoided crossing region, there is a sharp increase in entropy. This corresponds to the point where the ground state rapidly changes character — transitioning from being dominated by one computational basis state to being a strongly entangled superposition.

4. **Entanglement is driven by the Hamiltonian.** The key takeaway is that entanglement is not externally imposed: it emerges from the Hamiltonian's off-diagonal elements (the $H_x$ interaction) relative to the unperturbed energy gap. When the interaction becomes comparable to the energy spacing, the eigenstates become strongly entangled.

5. **Mutual information.** Since the total state is pure, $S(\rho_A) = S(\rho_B)$, and the entanglement entropy fully characterizes the bipartite entanglement. For a maximally entangled two-qubit state, the entropy reaches its maximum value of $S = 1$ bit.
