# Part g) — VQE for the Lipkin Model (20 pt)

## Objective

Use the VQE method to compute the ground state energy of the Lipkin model for both $J = 1$ (2 qubits) and $J = 2$ (4 qubits). Set up the circuits and simulations, compare with the classical results from Part f, and discuss the scaling challenges.

---

## Pauli Hamiltonian Construction

From Part f, we use the Pauli representation of the Lipkin Hamiltonian. For $N$ qubits:

$$H = \frac{\varepsilon}{2}\sum_{i=1}^{N} Z_i + \frac{V}{2}\sum_{i<j}(X_i X_j - Y_i Y_j).$$

This is implemented using Qiskit's `SparsePauliOp`, which allows efficient construction and evaluation of Pauli Hamiltonians. The number of Pauli terms scales as $N + 2\binom{N}{2} = N + N(N-1) = N^2$ terms.

## Ansatz Design

We use a **hardware-efficient ansatz** with alternating $R_y$ rotation layers and CZ entanglement:

```
Layer structure (for each depth d):
  1. Ry(θ) on each qubit
  2. CZ entanglement:
     - Even layers: pairs (0,1), (2,3), ...
     - Odd layers: pairs (1,2), (3,4), ... + wrap-around (0, N-1)
```

This alternating-brick pattern ensures full qubit connectivity after two layers.

### Configuration per system:

| System | Qubits | Depth | Parameters | Restarts | Optimizer |
|--------|--------|-------|------------|----------|-----------|
| $J = 1$ $(N=2)$ | 2 | 2 | 4 | 3 | L-BFGS-B |
| $J = 2$ $(N=4)$ | 4 | 4 | 16 | 4 | L-BFGS-B |

**Multiple random restarts** are used to mitigate the effect of local minima. The first restart uses $\theta = 0$ (computational basis initialization), and subsequent restarts draw $\theta$ uniformly from $[-\pi, \pi]$.

## Sector Consistency Verification

Before running the VQE sweep, we verify that the Pauli Hamiltonian produces eigenvalues consistent with the quasispin reference:

| System | Eigenvalue Match | Verified |
|--------|-----------------|----------|
| $J = 1$ $(N=2)$, Pauli $4 \times 4$ | Triplet eigenvalues match $3 \times 3$ quasispin | ✅ |
| $J = 2$ $(N=4)$, Pauli $16 \times 16$ | $J=2$ sector eigenvalues match $5 \times 5$ quasispin | ✅ |

This confirms that the ground state of the full Pauli Hamiltonian always lies in the maximal-$J$ sector for the Lipkin model, so the VQE will find the correct physical ground state without explicit symmetry projection.

## Results

### Ground State Energy Comparison

The VQE is swept over $V \in [0, 1.5]$ (21 points) for both $J = 1$ and $J = 2$:

![VQE many-body comparison](../plots/part-g_vqe_manybody.pdf)

The VQE ground state energies (markers) closely track the exact quasispin reference (lines) across the entire interaction range, including through the phase transition region.

### Precision

![VQE precision](../plots/part-g_vqe_precision.pdf)

| System | Max Absolute Error | Mean Error |
|--------|-------------------|------------|
| $J = 1$ $(N=2)$ | $\sim 10^{-11}$ | $\sim 10^{-12}$ |
| $J = 2$ $(N=4)$ | $\sim 10^{-2}$ | $\sim 10^{-3}$ |

- **$J = 1$** achieves machine-precision accuracy — the 4-parameter ansatz with depth 2 is sufficient to represent the ground state exactly.
- **$J = 2$** shows small but non-negligible errors ($\sim 10^{-2}$). These errors arise from the increased complexity of the optimization landscape with 16 parameters and local minima.

## Discussion

### 1. Scalability

The mapping from quasispin to Pauli strings provides a systematic way to encode many-body physics on a quantum computer. For $N$ particles, we need $N$ qubits and the number of Pauli terms scales as $O(N^2)$, which is polynomial. The circuit depth, however, needs to grow to maintain expressibility.

### 2. Optimization Landscape Challenges

The $J = 2$ case reveals a fundamental VQE challenge: the optimization landscape with 16 parameters has many local minima. Despite using L-BFGS-B with multiple random restarts, some values of $V$ converge to suboptimal solutions. This is a well-known issue in variational quantum algorithms, sometimes referred to as the *barren plateau* problem (though in our case the landscape is more accurately characterized by *narrow gorges* around the global minimum).

Possible improvements:
- **More restarts** (e.g., 15–20) to sample the landscape more thoroughly.
- **Deeper ansatz** (depth 5–6) for greater expressibility.
- **Symmetry-adapted ansatze** that enforce parity conservation, reducing the search space.
- **Alternative optimizers** (e.g., COBYLA, Nelder-Mead) that may escape local minima differently.

### 3. Physical Ground State in the Maximal-J Sector

A subtle but important result: the ground state of the full $2^N$-dimensional Pauli Hamiltonian lies in the $J = N/2$ sector for the Lipkin model. This means the VQE naturally finds the physically correct ground state without needing symmetry projection or post-selection. This is because the Lipkin interaction $J_+^2 + J_-^2$ maximally lowers the energy of the maximal-$J$ states.

### 4. Comparison with Part f

The VQE energy curves faithfully reproduce the spectral features from Part f:
- The ground state energy decreases monotonically with $V$.
- The phase transition at $V_c = \varepsilon/(N-1)$ is captured.
- The scaling behavior (energy per particle approaching the Hartree-Fock limit) is preserved.

### 5. Practical Considerations for Quantum Hardware

On real quantum hardware, additional sources of error would include:
- **Shot noise** from finite-statistics measurements.
- **Gate errors** from imperfect hardware operations.
- **Decoherence** limiting circuit depths.
- **Readout errors** biasing measurement outcomes.

Our statevector simulation avoids all of these, providing an upper bound on VQE performance.
