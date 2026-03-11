# Part f) — The Lipkin Model: Classical Analysis (15 pt)

## Objective

Introduce the Lipkin-Meshkov-Glick (LMG) model, derive the Hamiltonian matrices for $J = 1$ ($3 \times 3$) and $J = 2$ ($5 \times 5$), rewrite them in terms of Pauli spin matrices, and diagonalize classically to study the spectrum as a function of the interaction strength $V$.

---

## The Lipkin Model

The Lipkin model describes $N$ fermions distributed over two $N$-fold degenerate energy levels ("shells"), separated by an energy $\varepsilon$. The Hamiltonian in second quantization is:

$$H = H_0 + H_1 + H_2,$$

where:
- $H_0 = \varepsilon J_z$ is the single-particle energy (counting the level occupancy),
- $H_1 = \frac{1}{2}V(J_+^2 + J_-^2)$ is the pair-scattering interaction (promoting/demoting pairs),
- $H_2 = \frac{1}{2}W(-N + J_+ J_- + J_- J_+)$ is a particle-exchange interaction.

The operators $J_z, J_+, J_-$ are quasispin operators obeying $\mathfrak{su}(2)$ algebra, with $J = N/2$. We set $W = 0$ throughout this part.

## J = 1 Case (N = 2 particles)

### Quasispin Matrix

In the $|J=1, m\rangle$ basis with $m \in \{-1, 0, +1\}$:

$$H_{J=1} = \begin{pmatrix} -\varepsilon & 0 & -V \\ 0 & 0 & 0 \\ -V & 0 & \varepsilon \end{pmatrix}$$

This matches the form stated in the project description. The $m = 0$ state is uncoupled — it is an eigenstate with energy 0 for all values of $V$.

### Pauli Decomposition (2-qubit mapping)

To solve on a quantum computer, we map $N = 2$ particles to $N = 2$ qubits. The quasispin operators translate to:

$$J_z = \frac{1}{2}(Z_1 + Z_2), \qquad J_+^2 + J_-^2 = X_1 X_2 - Y_1 Y_2.$$

The resulting Pauli Hamiltonian is:

$$\boxed{H_{J=1} = \frac{\varepsilon}{2}(Z_1 \otimes I + I \otimes Z_2) + \frac{V}{2}(X_1 \otimes X_2 - Y_1 \otimes Y_2)}$$

Or in Pauli string notation: $H = \frac{\varepsilon}{2} ZI + \frac{\varepsilon}{2} IZ + \frac{V}{2} XX - \frac{V}{2} YY$.

### Verification

The 4-qubit Pauli Hamiltonian ($4 \times 4$) contains the physical $J = 1$ triplet subspace (dimension 3) *plus* the unphysical $J = 0$ singlet state (dimension 1). The triplet states in the computational basis are:

| Quasispin state | Computational basis |
|----------------|-------------------|
| $\|J=1, m=-1\rangle$ | $\|11\rangle$ |
| $\|J=1, m=0\rangle$ | $(\|01\rangle + \|10\rangle)/\sqrt{2}$ |
| $\|J=1, m=+1\rangle$ | $\|00\rangle$ |

Projecting the $4 \times 4$ Pauli Hamiltonian onto this triplet subspace recovers the $3 \times 3$ quasispin matrix exactly. ✓

## J = 2 Case (N = 4 particles)

### Quasispin Matrix

In the $|J=2, m\rangle$ basis with $m \in \{-2, -1, 0, +1, +2\}$:

$$H_{J=2} = \begin{pmatrix} -2\varepsilon & 0 & \sqrt{6}V & 0 & 0 \\ 0 & -\varepsilon & 0 & 3V & 0 \\ \sqrt{6}V & 0 & 0 & 0 & \sqrt{6}V \\ 0 & 3V & 0 & \varepsilon & 0 \\ 0 & 0 & \sqrt{6}V & 0 & 2\varepsilon \end{pmatrix}$$

Note the selection rule $\Delta m = \pm 2$ from $J_+^2 + J_-^2$, creating a checkerboard pattern.

### Pauli Decomposition (4-qubit mapping)

For $N = 4$ particles mapped to 4 qubits:

$$J_z = \frac{1}{2}\sum_{i=1}^{4} Z_i, \qquad J_+^2 + J_-^2 = 2\sum_{i < j} (X_i X_j - Y_i Y_j).$$

The Hamiltonian:

$$\boxed{H_{J=2} = \frac{\varepsilon}{2}\sum_{i=1}^{4} Z_i + \frac{V}{2}\sum_{i<j}(X_i X_j - Y_i Y_j)}$$

This involves 4 single-qubit $Z$ terms and $2 \times \binom{4}{2} = 12$ two-qubit interaction terms ($XX$ and $YY$ for each of the 6 pairs). The total Pauli Hamiltonian is $16 \times 16$, containing all angular momentum sectors $J = 0, 1, 2$. The $J = 2$ sector eigenvalues (5 values) were verified to match the quasispin reference. ✓

## Eigenvalue Sweep

The eigenvalues are computed as functions of the interaction strength $V \in [0, 1.5]$ (with $\varepsilon = 1$ and $W = 0$).

### J = 1 (N = 2)

![J=1 spectrum](../plots/part-f_j1.pdf)

### J = 2 (N = 4)

![J=2 spectrum](../plots/part-f_j2.pdf)

## Hartree-Fock Comparison

The Hartree-Fock (HF) approximation provides a mean-field estimate of the ground state energy. For the Lipkin model with $W = 0$:

$$E_\text{HF} = \begin{cases} -\frac{N \varepsilon}{2} & \text{if } \tilde{V} \leq 1 \quad (\text{weak coupling}), \\[4pt] -\frac{N \varepsilon}{4}\left(\tilde{V} + \frac{1}{\tilde{V}}\right) & \text{if } \tilde{V} > 1 \quad (\text{strong coupling}), \end{cases}$$

where $\tilde{V} = (N-1)V/\varepsilon$ is the dimensionless coupling.

The **critical point** occurs at $\tilde{V} = 1$, i.e., $V_c = \varepsilon/(N-1)$:
- For $N = 2$: $V_c = 1.000$
- For $N = 4$: $V_c = 0.333$

### Scaling

![Scaling analysis](../plots/part-f_scaling.pdf)

The energy per particle $E_0/N$ approaches the HF result as $N$ increases, consistent with the HF approximation becoming exact in the thermodynamic limit ($N \to \infty$).

## Discussion

1. **Phase Transition.** The Lipkin model exhibits a quantum phase transition at $V_c$. For $V < V_c$, the ground state is close to $|m = -J\rangle$ (all particles in the lower level). For $V > V_c$, pair correlations drive the ground state toward a superposition of different $m$-values.

2. **Parity Symmetry.** With $W = 0$, the Hamiltonian commutes with the parity operator $\Pi = e^{i\pi(J + J_z)}$, which implies $\Delta m = \pm 2$ selection rules. The even-$m$ and odd-$m$ states decouple into separate blocks.

3. **Hartree-Fock Limitations.** Below $V_c$, HF gives the exact ground state energy (since $|m = -J\rangle$ is the single Slater determinant). Above $V_c$, HF underestimates the correlation energy, with the gap to the exact result being largest just above $V_c$. The gap shrinks for larger $N$ (HF becomes asymptotically exact).

4. **Pauli Embedding Overhead.** The qubit mapping embeds the $J$-sector into a larger Hilbert space ($2^N$ vs $2J+1$). For $N = 4$: the $16 \times 16$ space contains $J = 0$ (dim 1), $J = 1$ (dim 3 × multiplicity 3), and $J = 2$ (dim 5) sectors. The ground state of the full Pauli Hamiltonian lies in the maximal-$J$ sector for the Lipkin model, so no projection is required for VQE — the variational principle naturally finds the correct state.
