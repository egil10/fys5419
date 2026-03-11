# Results: The Lipkin Model
*Covers Parts f and g*

---

## Quasispin Matrix Construction

The Lipkin Hamiltonian is built using the quasispin operators $J_z$, $J_+$, $J_-$ with the pair-scattering interaction ($W = 0$). The resulting matrices are:

### $J = 1$ (N = 2 particles, $3 \times 3$)

$$H_{J=1} = \begin{pmatrix} -\varepsilon & 0 & -V \\ 0 & 0 & 0 \\ -V & 0 & \varepsilon \end{pmatrix}$$

The $m = 0$ state is decoupled — it is an eigenstate with energy 0 for all $V$. The interaction only connects $m = -1 \leftrightarrow m = +1$ (the $\Delta m = \pm 2$ selection rule from $J_+^2 + J_-^2$).

### $J = 2$ (N = 4 particles, $5 \times 5$)

$$H_{J=2} = \begin{pmatrix} -2\varepsilon & 0 & \sqrt{6}V & 0 & 0 \\ 0 & -\varepsilon & 0 & 3V & 0 \\ \sqrt{6}V & 0 & 0 & 0 & \sqrt{6}V \\ 0 & 3V & 0 & \varepsilon & 0 \\ 0 & 0 & \sqrt{6}V & 0 & 2\varepsilon \end{pmatrix}$$

The checkerboard pattern ($\Delta m = \pm 2$) persists, decoupling even-$m$ and odd-$m$ states into separate blocks.

---

## Pauli Decomposition

### $J = 1$: Two-qubit mapping

Mapping the quasispin operators to $N = 2$ qubits gives:

$$\boxed{H_{J=1} = \frac{\varepsilon}{2}(ZI + IZ) + \frac{V}{2}(XX - YY)}$$

**Verification:** The $4 \times 4$ Pauli Hamiltonian contains both the $J = 1$ triplet (dim 3) and $J = 0$ singlet (dim 1). Projecting onto the triplet subspace — spanned by $|11\rangle$ ($m = -1$), $(|01\rangle + |10\rangle)/\sqrt{2}$ ($m = 0$), and $|00\rangle$ ($m = +1$) — exactly recovers the $3 \times 3$ quasispin matrix. ✓

### $J = 2$: Four-qubit mapping

$$\boxed{H_{J=2} = \frac{\varepsilon}{2}\sum_{i=1}^{4} Z_i + \frac{V}{2}\sum_{i<j}(X_i X_j - Y_i Y_j)}$$

This gives 4 single-qubit $Z$ terms and 12 two-qubit terms ($XX$ and $YY$ for each of the 6 pairs), totaling 16 Pauli strings. The $16 \times 16$ Hilbert space contains $J = 0, 1, 2$ sectors. The five $J = 2$ eigenvalues were extracted from the full Pauli spectrum and verified against the quasispin reference. ✓

---

## Eigenvalue Spectrum

The eigenvalues are computed over $V \in [0, 1.5]$ with $\varepsilon = 1$, $W = 0$.

### $J = 1$

![Lipkin J=1 spectrum](../plots/part-f_j1.pdf)

### $J = 2$

![Lipkin J=2 spectrum](../plots/part-f_j2.pdf)

Both spectra show the ground state energy decreasing with increasing $V$, reflecting the energetically favorable pair correlations introduced by the interaction.

---

## Hartree-Fock Comparison

The Hartree-Fock energy for the Lipkin model is:

$$E_\text{HF} = \begin{cases} -N\varepsilon/2 & \text{if } \tilde{V} \leq 1, \\ -\frac{N\varepsilon}{4}(\tilde{V} + 1/\tilde{V}) & \text{if } \tilde{V} > 1, \end{cases}$$

where $\tilde{V} = (N-1)V/\varepsilon$. The critical interaction strengths are:

| System | $V_c = \varepsilon/(N-1)$ |
|--------|--------------------------|
| $N = 2$ ($J = 1$) | **1.000** |
| $N = 4$ ($J = 2$) | **0.333** |

### Scaling analysis

![Energy per particle](../plots/part-f_scaling.pdf)

The energy per particle $E_0/N$ approaches the Hartree-Fock curve as $N$ increases, consistent with HF becoming exact in the thermodynamic limit. Below $V_c$, HF gives the exact answer (the ground state is a single Slater determinant). Above $V_c$, HF systematically overestimates the energy — the correlation energy is missing.

---

## Lipkin VQE

### Circuit Design

A hardware-efficient ansatz with $R_y$ rotations and CZ entanglement is used:

| System | Qubits | Depth | Parameters | Restarts | Optimizer |
|--------|--------|-------|------------|----------|-----------|
| $J = 1$ | 2 | 2 | 4 | 3 | L-BFGS-B |
| $J = 2$ | 4 | 4 | 16 | 4 | L-BFGS-B |

The ansatz uses alternating entanglement layers (even: pairs (0,1),(2,3); odd: pairs (1,2),(3,4) + wrap-around) to ensure full qubit connectivity after two layers. Multiple random restarts mitigate local minima.

### Sector Consistency

Before running VQE, we verify that the ground state of the full Pauli Hamiltonian lies in the correct angular momentum sector:

| System | Pauli GS matches $J = N/2$ sector? |
|--------|-------------------------------------|
| $J = 1$ | ✅ Yes |
| $J = 2$ | ✅ Yes |

The Lipkin interaction maximally lowers the energy of the maximal-$J$ states, so VQE finds the physical ground state without needing symmetry projection.

### Energy Comparison

![VQE vs exact](../plots/part-g_vqe_manybody.pdf)

The VQE ground state energies track the exact quasispin reference across the full $V$ range, including through the phase transition region.

### Precision

| System | Max Error | Mean Error |
|--------|-----------|------------|
| $J = 1$ ($N = 2$) | $\sim 10^{-11}$ | $\sim 10^{-12}$ |
| $J = 2$ ($N = 4$) | $\sim 10^{-2}$ | $\sim 10^{-3}$ |

![VQE precision](../plots/part-g_vqe_precision.pdf)

### Discussion

1. **$J = 1$ — exact accuracy.** With only 4 parameters and 2 qubits, the ansatz is expressive enough to represent the ground state exactly. The error is limited only by floating-point arithmetic.

2. **$J = 2$ — optimization challenges.** The 4-qubit system with 16 parameters encounters local minima in the optimization landscape. Despite multiple restarts, some $V$ values converge to suboptimal solutions, resulting in $O(10^{-2})$ errors. This is a well-known challenge in variational algorithms — the landscape becomes increasingly complex with system size.

3. **Possible improvements.** Several strategies could reduce the $J = 2$ error:
   - Increasing the number of random restarts (10–20)
   - Using deeper circuits (depth 5–6) for greater expressibility
   - Symmetry-adapted ansatze that restrict the search space to the physical $J = 2$ sector
   - Alternative optimizers (COBYLA, Nelder-Mead) that may navigate the landscape differently

4. **Phase transition captured.** Both $J = 1$ and $J = 2$ VQE curves correctly reproduce the critical behavior at $V_c$. The ground state energy transitions from the mean-field value $-N\varepsilon/2$ to the correlated regime, consistent with the exact and Hartree-Fock analysis.

5. **Scaling.** The mapping from $N$ particles to $N$ qubits requires $O(N^2)$ Pauli terms — polynomial and hence efficient. The circuit depth, however, needs to grow with $N$ to maintain expressibility. On real quantum hardware, gate noise and decoherence would further limit the achievable depth.
