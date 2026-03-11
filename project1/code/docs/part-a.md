# Part a) — Quantum Basics: Gates, Bell States, and Entanglement (15 pt)

## Objective

Set up a one-qubit computational basis, apply the Pauli and standard quantum gates, construct Bell states, perform simulated measurements, and compute the von Neumann entropy to quantify entanglement.

---

## 1. One-Qubit Basis and Pauli Matrices

We define the standard computational basis states

$$|0\rangle = \begin{pmatrix} 1 \\ 0 \end{pmatrix}, \quad |1\rangle = \begin{pmatrix} 0 \\ 1 \end{pmatrix},$$

and verify the action of the three Pauli matrices:

| Operator | Action on $\|0\rangle$ | Action on $\|1\rangle$ |
|----------|----------------------|----------------------|
| $\sigma_x$ | $\|1\rangle$ | $\|0\rangle$ |
| $\sigma_y$ | $i\|1\rangle$ | $-i\|0\rangle$ |
| $\sigma_z$ | $\|0\rangle$ | $-\|1\rangle$ |

The results confirm that $\sigma_x$ acts as a bit-flip, $\sigma_y$ as a bit-flip with a phase of $\pm i$, and $\sigma_z$ as a phase-flip (eigenvalue $+1$ for $|0\rangle$, $-1$ for $|1\rangle$).

## 2. Hadamard and Phase Gates

The Hadamard gate $H = \frac{1}{\sqrt{2}}\begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}$ creates equal superpositions:

$$H|0\rangle = |+\rangle = \frac{|0\rangle + |1\rangle}{\sqrt{2}}, \quad H|1\rangle = |-\rangle = \frac{|0\rangle - |1\rangle}{\sqrt{2}}.$$

The Phase gate $S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$ leaves $|0\rangle$ unchanged and adds a relative phase: $S|1\rangle = i|1\rangle$. These gates are fundamental building blocks for quantum circuits — the Hadamard enables basis changes between the $Z$ and $X$ eigenbases, while the Phase gate introduces relative phases necessary for universal quantum computation.

## 3. Bell States

The four maximally entangled Bell states are constructed as:

$$|\Phi^+\rangle = \frac{|00\rangle + |11\rangle}{\sqrt{2}}, \quad |\Phi^-\rangle = \frac{|00\rangle - |11\rangle}{\sqrt{2}},$$

$$|\Psi^+\rangle = \frac{|01\rangle + |10\rangle}{\sqrt{2}}, \quad |\Psi^-\rangle = \frac{|01\rangle - |10\rangle}{\sqrt{2}}.$$

These states form a complete orthonormal basis for the two-qubit maximally entangled subspace.

## 4. Bell State Preparation and Measurement

Starting from $|00\rangle$, we apply a Hadamard gate on the first qubit followed by a CNOT gate:

$$(H \otimes I)|00\rangle = \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle) \xrightarrow{\text{CNOT}} \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle.$$

This is verified numerically.

### Measurement Simulation

We simulate measurements (50 independent runs, 1000 shots each) on the prepared $|\Phi^+\rangle$ state. The averaged results are:

| Outcome | Probability | Std. Dev. |
|---------|------------|-----------|
| $\|00\rangle$ | 0.4988 | ±0.018 |
| $\|01\rangle$ | 0.0000 | ±0.000 |
| $\|10\rangle$ | 0.0000 | ±0.000 |
| $\|11\rangle$ | 0.5012 | ±0.018 |

The results confirm the expected 50/50 distribution between $|00\rangle$ and $|11\rangle$, with zero probability for $|01\rangle$ and $|10\rangle$. This is the hallmark of entanglement in $|\Phi^+\rangle$: the two qubits are perfectly correlated — if the first qubit is measured as $|0\rangle$, the second *must* collapse to $|0\rangle$, and vice versa. The small statistical deviations (±0.018) are consistent with the expected standard error of $\sqrt{p(1-p)/N_{\text{shots}}} \approx 0.016$.

![Measurement distribution](../plots/part-a_measurement_dist.pdf)

## 5. Density Matrix and von Neumann Entropy

To quantify entanglement, we compute the von Neumann entropy of a subsystem. For a bipartite state $|\psi\rangle_{AB}$, the reduced density matrix for subsystem $A$ is obtained by tracing over $B$:

$$\rho_A = \text{Tr}_B(|\psi\rangle\langle\psi|).$$

The von Neumann entropy is then:

$$S(\rho_A) = -\text{Tr}(\rho_A \log_2 \rho_A) = -\sum_i \lambda_i \log_2 \lambda_i,$$

where $\lambda_i$ are the eigenvalues of $\rho_A$.

### Results

| State | Type | $S(\rho_A)$ |
|-------|------|-------------|
| $\|\Phi^+\rangle$ | Maximally entangled | **1.000** |
| $\|\Phi^-\rangle$ | Maximally entangled | **1.000** |
| $\|\Psi^+\rangle$ | Maximally entangled | **1.000** |
| $\|\Psi^-\rangle$ | Maximally entangled | **1.000** |
| $\|00\rangle$ | Separable (product) | **0.000** |
| $\|+\rangle \otimes \|0\rangle$ | Separable (superposition) | **0.000** |

### Discussion

All four Bell states yield the maximum possible entropy $S = 1$ bit for a two-qubit system, confirming they are *maximally entangled*. The reduced density matrix for each Bell state is the maximally mixed state $\rho_A = \frac{1}{2}I$, which carries no information about the individual subsystem.

In contrast, separable states produce $S = 0$, including $|+\rangle \otimes |0\rangle$, which is a superposition but *not* entangled. This demonstrates that superposition and entanglement are distinct quantum phenomena: a product state has zero entropy regardless of whether each subsystem is in a definite state or in a superposition. Entanglement entropy captures strictly inter-subsystem quantum correlations.
