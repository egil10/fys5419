# Results: Single-Qubit Operations and Bell State Entanglement
*Covers Parts a*

---

## Pauli Matrix Actions

Applying the Pauli operators defined in the Methods section to the computational basis states confirms the expected behavior:

| Operator | $\|0\rangle \to$ | $\|1\rangle \to$ | Interpretation |
|----------|-------------------|-------------------|----------------|
| $\sigma_x$ | $\|1\rangle$ | $\|0\rangle$ | Bit-flip |
| $\sigma_y$ | $i\|1\rangle$ | $-i\|0\rangle$ | Bit-flip + phase |
| $\sigma_z$ | $+\|0\rangle$ | $-\|1\rangle$ | Phase-flip |

These serve as the elementary operations from which all quantum circuits and Hamiltonian decompositions in this project are constructed.

## Gate Actions

The Hadamard gate maps computational basis states to equal superpositions: $H|0\rangle = |+\rangle$, $H|1\rangle = |-\rangle$. The Phase gate introduces a relative phase of $i$ on $|1\rangle$ while leaving $|0\rangle$ unchanged. Both gates act as expected.

## Bell State Preparation

Starting from $|00\rangle$, the circuit $(H \otimes I)$ followed by CNOT produces:

$$|00\rangle \xrightarrow{H \otimes I} \frac{1}{\sqrt{2}}(|00\rangle + |10\rangle) \xrightarrow{\text{CNOT}} \frac{1}{\sqrt{2}}(|00\rangle + |11\rangle) = |\Phi^+\rangle,$$

confirmed numerically.

## Measurement Simulation

Repeated measurement of $|\Phi^+\rangle$ (50 runs $\times$ 1000 shots) yields:

| Outcome | Mean Probability | Std. Dev. |
|---------|-----------------|-----------|
| $\|00\rangle$ | 0.4988 | $\pm$0.018 |
| $\|01\rangle$ | 0.0000 | $\pm$0.000 |
| $\|10\rangle$ | 0.0000 | $\pm$0.000 |
| $\|11\rangle$ | 0.5012 | $\pm$0.018 |

The outcomes are exclusively $|00\rangle$ and $|11\rangle$ with equal probability — the defining signature of the $|\Phi^+\rangle$ Bell state. The two qubits are perfectly correlated: measuring one immediately determines the other. The statistical fluctuations ($\pm 0.018$) are consistent with the expected shot noise $\sqrt{p(1-p)/N_\text{shots}} \approx 0.016$.

![Measurement distribution](../plots/part-a_measurement_dist.pdf)

## Entanglement: von Neumann Entropy

The von Neumann entropy $S = -\text{Tr}(\rho_A \log_2 \rho_A)$ is computed after tracing out one subsystem:

| State | Type | $S$ |
|-------|------|-----|
| $\|\Phi^+\rangle$ | Maximally entangled | **1.000** |
| $\|\Phi^-\rangle$ | Maximally entangled | **1.000** |
| $\|\Psi^+\rangle$ | Maximally entangled | **1.000** |
| $\|\Psi^-\rangle$ | Maximally entangled | **1.000** |
| $\|00\rangle$ | Product state | **0.000** |
| $\|+\rangle \otimes \|0\rangle$ | Product (superposition) | **0.000** |

All four Bell states yield the maximum entropy of 1 bit, confirming maximal entanglement. Crucially, the state $|+\rangle \otimes |0\rangle$ has $S = 0$ despite being a superposition — demonstrating that superposition and entanglement are fundamentally distinct. Entanglement entropy measures inter-subsystem quantum correlations specifically, not local coherence.
