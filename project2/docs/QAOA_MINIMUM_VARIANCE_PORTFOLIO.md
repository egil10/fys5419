# QAOA and Minimum-Variance Portfolios

This note collects a **self-contained mathematical formulation** for comparing **classical minimum-variance portfolio optimization** with a **discrete combinatorial** problem that is natural to encode for the **Quantum Approximate Optimization Algorithm (QAOA)**. It is written to support the accompanying notebook work in `project2/code/notebooks/`.

---

## Table of contents

1. [Setting: returns, covariance, and portfolio variance](#1-setting-returns-covariance-and-portfolio-variance)
2. [Continuous Markowitz minimum-variance portfolio](#2-continuous-markowitz-minimum-variance-portfolio)
3. [From volatilities and correlations to \(\Sigma\)](#3-from-volatilities-and-correlations-to-sigma)
4. [Why QAOA needs a discrete (QUBO/Ising) problem](#4-why-qaoa-needs-a-discrete-quboising-problem)
5. [A standard discrete template: equal-weight subset selection](#5-a-standard-discrete-template-equal-weight-subset-selection)
6. [QUBO sketch and penalty for the cardinality constraint](#6-qubo-sketch-and-penalty-for-the-cardinality-constraint)
7. [Classical baselines that match the discrete problem](#7-classical-baselines-that-match-the-discrete-problem)
8. [QAOA at a high level (what you implement)](#8-qaoa-at-a-high-level-what-you-implement)
9. [What to measure and plot](#9-what-to-measure-and-plot)
10. [Interpretation caveats (important for reports)](#10-interpretation-caveats-important-for-reports)
11. [Pointers to software](#11-pointers-to-software)

---

## 1. Setting: returns, covariance, and portfolio variance

Let there be \(n\) risky assets (stocks) indexed by \(i=1,\ldots,n\).

- **Random simple returns** (or excess returns) are modeled as a random vector  
  $$
    R \in \mathbb{R}^n.
  $$
- The **expected return vector** is  
  $$
    \mu = \mathbb{E}[R] \in \mathbb{R}^n.
  $$
- The **covariance matrix** is  
  $$
    \Sigma = \mathrm{Cov}(R) \in \mathbb{R}^{n\times n},
  $$
  symmetric positive semidefinite (\(\Sigma \succeq 0\)).

A **portfolio** is a vector of weights \(w \in \mathbb{R}^n\) where \(w_i\) is the fraction of wealth in asset \(i\). The portfolio return is \(R_p = w^\top R\), so

$$
  \mathbb{E}[R_p] = w^\top \mu,
  \qquad
  \mathrm{Var}(R_p) = w^\top \Sigma w.
$$

The **portfolio variance** is therefore the quadratic form

$$
  \sigma_p^2(w) = w^\top \Sigma w.
$$

Everything below about “minimum variance” refers to minimizing \(\sigma_p^2(w)\) (or an equivalent discrete energy) under stated constraints.

---

## 2. Continuous Markowitz minimum-variance portfolio

The textbook **long-only, fully invested** minimum-variance portfolio solves

$$
  \begin{aligned}
    \min_{w \in \mathbb{R}^n}\quad & w^\top \Sigma w \\
    \text{s.t.}\quad & \mathbf{1}^\top w = 1, \\
                     & w \ge 0 \quad \text{(componentwise).}
  \end{aligned}
$$

This is a **convex quadratic program (QP)** because \(\Sigma \succeq 0\) and the feasible set is a polyhedron (linear equalities and inequalities).

### Karush–Kuhn–Tucker (KKT) intuition (optional but useful)

At an optimum, there exist multipliers such that stationarity holds together with primal feasibility, dual feasibility, and complementary slackness. In practice you do **not** need to hand-solve KKT for a course notebook: calling a QP-capable solver (e.g. `scipy.optimize` with suitable method, or a dedicated QP library) is standard.

### What this solution represents

The continuous optimizer returns **fractional weights** \(w^\star\) that achieve the **lowest variance** among all long-only fully invested portfolios for the given \(\Sigma\). This is the usual **reference “classical” solution** in portfolio theory.

---

## 3. From volatilities and correlations to \(\Sigma\)

In teaching examples (and often in practice), \(\Sigma\) is assembled from

- asset **volatilities** \(\sigma_i > 0\) (standard deviations of returns), and  
- a **correlation matrix** \(C \in \mathbb{R}^{n\times n}\) with \(C_{ii}=1\), \(C_{ij}\in[-1,1]\), and \(C \succeq 0\).

The standard construction is

$$
  \Sigma_{ij} = \sigma_i \sigma_j \, C_{ij}.
$$

Equivalently, with \(D = \mathrm{diag}(\sigma_1,\ldots,\sigma_n)\),

$$
  \Sigma = D \, C \, D.
$$

**Sanity checks you should enforce when fabricating numbers:**

- \(C\) must be symmetric and positive semidefinite (e.g. build it from a valid parametric family or from empirical returns).
- \(\sigma_i\) must be positive scalars.

---

## 4. Why QAOA needs a discrete (QUBO/Ising) problem

QAOA is typically applied to **combinatorial optimization** problems whose cost function can be written as a **Quadratic Unconstrained Binary Optimization (QUBO)** problem, equivalently an **Ising model** on spins \(\{-1,+1\}\) or bitstrings \(\{0,1\}^n\).

The continuous Markowitz problem in Section 2 has **continuous** decision variables \(w_i \in \mathbb{R}\) with simplex constraints. That is **not** a native QUBO. Therefore, a fair notebook workflow is:

- treat the **continuous QP** as a classical baseline for “standard” portfolio theory; and  
- define a **discrete** portfolio rule that maps naturally to QUBO, solve it with QAOA, and compare **within the discrete problem class**.

Trying to compare “QAOA energy” directly to “Markowitz objective” without a shared decision set mixes different problems.

---

## 5. A standard discrete template: equal-weight subset selection

Fix an integer \(K\) with \(1 \le K \le n\). Introduce binary decision variables

$$
  z_i \in \{0,1\}, \quad i=1,\ldots,n,
$$

where \(z_i=1\) means “asset \(i\) is selected.” The **cardinality constraint** is

$$
  \sum_{i=1}^n z_i = K.
$$

Define the **equal-weight** portfolio on the selected set:

$$
  w_i(z) = \frac{z_i}{K}.
$$

Then \(\sum_i w_i = 1\) automatically holds whenever exactly \(K\) assets are selected.

### Portfolio variance as a quadratic function of \(z\)

Let \(z \in \{0,1\}^n\) be the vector of selections. The portfolio variance is

$$
  \sigma_p^2(z)
  = w(z)^\top \Sigma \, w(z)
  = \frac{1}{K^2}\, z^\top \Sigma z,
$$

because \(w = z/K\).

Expanding the bilinear form is often convenient for QUBO bookkeeping:

$$
  z^\top \Sigma z
  = \sum_{i=1}^n \Sigma_{ii} z_i^2
    + 2 \sum_{1 \le i < j \le n} \Sigma_{ij} z_i z_j.
$$

Since \(z_i^2 = z_i\) for binary variables,

$$
  z^\top \Sigma z
  = \sum_{i=1}^n \Sigma_{ii} z_i
    + 2 \sum_{i<j} \Sigma_{ij} z_i z_j.
$$

Therefore the **discrete minimum-variance** problem among feasible \(z\) is

$$
  \min_{z \in \{0,1\}^n}\ \frac{1}{K^2}\, z^\top \Sigma z
  \quad \text{s.t.}\quad \sum_{i=1}^n z_i = K.
$$

For \(n=3\) and \(K=2\), there are only \(\binom{3}{2}=3\) feasible subsets, which is ideal for **exact brute force** as a gold standard for the discrete problem.

---

## 6. QUBO sketch and penalty for the cardinality constraint

QAOA implementations often expect an **unconstrained** binary polynomial, using a **penalty** to encode constraints.

Define the constraint violation

$$
  P(z) = \left(\sum_{i=1}^n z_i - K\right)^2.
$$

A penalized objective can be written as

$$
  \mathcal{E}_\lambda(z)
  = \frac{1}{K^2}\, z^\top \Sigma z
    + \lambda\, P(z),
$$

with \(\lambda > 0\) chosen large enough so that feasible cardinality-\(K\) solutions beat any infeasible bitstring in energy (in exact optimization). In practice, \(\lambda\) may require tuning when using variational/heuristic optimization.

### Expanding the penalty (useful for implementation)

$$
  P(z)
  = \left(\sum_i z_i\right)^2 - 2K\sum_i z_i + K^2
  = \sum_i z_i + 2\sum_{i<j} z_i z_j - 2K\sum_i z_i + K^2,
$$

using \(z_i^2=z_i\).

Constant terms (\(+K^2\)) shift all energies equally and can be dropped for optimization, but keep them if you want numerical values to match a particular reference implementation.

### From QUBO to Ising (conceptual)

A QUBO is commonly written as minimizing

$$
  x^\top Q x + c^\top x + \text{const},
  \quad x \in \{0,1\}^n,
$$

while an Ising model uses spins \(s_i \in \{-1,+1\}\) with the linear change of variables

$$
  z_i = \frac{1 - s_i}{2}
  \quad\text{or}\quad
  s_i = 1 - 2z_i
  \quad (\text{conventions vary; be consistent in code}).
$$

Quantum libraries (e.g. Qiskit) often accept **Pauli sum** representations (`SparsePauliOp`) obtained from these transformations.

---

## 7. Classical baselines that match the discrete problem

You want **two classical references**:

### A. Brute force over feasible discrete allocations

For small \(n\) (like \(n=3\), \(K=2\)), enumerate all \(z\) with \(\sum z_i = K\), compute \(\sigma_p^2(z) = \frac{1}{K^2} z^\top \Sigma z\), and take the minimum.

This is the **exact optimum** of the discrete problem and is the cleanest comparison target for QAOA on the same discrete objective.

### B. Continuous Markowitz (Section 2)

Keep this as the “standard finance” baseline. It answers a **different** mathematical question unless you also build a discrete encoding that converges to continuous weights in a well-defined limit.

---

## 8. QAOA at a high level (what you implement)

Let \(H_C\) be the Ising/Pauli Hamiltonian whose ground state corresponds (up to conventions) to minimizing \(\mathcal{E}_\lambda(z)\).

QAOA prepares a parameterized quantum state using alternating layers of **problem Hamiltonian evolution** and **mixer Hamiltonian evolution** (often Pauli-\(X\) rotations on each qubit), with depth \(p\) (“`reps`” in libraries).

A typical QAOA objective is to minimize the **expected cost**

$$
  \langle \psi(\gamma,\beta) \mid H_C \mid \psi(\gamma,\beta) \rangle
$$

over QAOA parameters \((\gamma,\beta)\) using a classical optimizer.

**Operational checklist:**

- Fix seeds for reproducibility (simulator + classical optimizer).
- Start with small \(p\) (e.g. \(p=1\) or \(2\)) before increasing depth.
- Remember measurement/statistical noise if using sampling primitives; statevector simulation removes sampling noise but not optimizer non-convexity.

---

## 9. What to measure and plot

### Timings (wall clock)

Measure elapsed time for:

- continuous QP solve (Markowitz),
- brute-force discrete optimum (should be negligible at \(n=3\)),
- QAOA optimization loop (usually dominates for toy problems in simulation).

**Expectation:** classical methods win on runtime for tiny \(n\). A timing plot is still valuable because it documents **computational cost** honestly.

### Solution plots

- **Bar chart of weights:** continuous \(w^\star\) vs discrete equal-weight portfolio implied by QAOA’s best \(z\).
- **Objective values:** \(\sigma_p^2\) for continuous optimum vs discrete brute force vs QAOA’s best found \(z\).

### Optional QAOA diagnostics

- best achieved energy vs optimizer iteration,
- histogram of sampled bitstrings (if using a sampling primitive).

---

## 10. Interpretation caveats (important for reports)

1. **Different decision sets:** continuous simplex weights vs “choose \(K\) equal-weight names” are not equivalent optima for the same \(\Sigma\).
2. **Penalty \(\lambda\):** too small → infeasible strings look optimal; too large → optimization can become ill-conditioned for variational methods.
3. **QAOA is heuristic:** local minima, barren plateaus (in deeper/larger settings), and optimizer choice matter.
4. **Scaling narrative:** QAOA comparisons at \(n=3\) are pedagogical; asymptotic speed claims require careful definitions of hardware, error rates, and problem families.

---

## 11. Pointers to software

- **Qiskit Algorithms QAOA tutorial (graph partitioning example):**  
  [Quantum Approximate Optimization Algorithm — Qiskit Algorithms](https://qiskit-community.github.io/qiskit-algorithms/tutorials/05_qaoa.html)

- **Portfolio quadratic programs** are widely documented in optimization texts; SciPy’s constrained minimizers are a lightweight entry point for the continuous baseline.

- **Qiskit Optimization** can compile some discrete optimization models to QUBO/Ising automatically; it is optional if you prefer deriving the Hamiltonian explicitly for learning purposes.

---

## Appendix: tiny worked symbols for \(n=3\), \(K=2\)

Let

$$
  z \in \{0,1\}^3,\quad z_1+z_2+z_3=2,
  \quad
  w = \frac{z}{2}.
$$

Then

$$
  \sigma_p^2(z)
  = \frac{1}{4}\, z^\top \Sigma z
  = \frac{1}{4}\left(
      \sum_{i=1}^3 \Sigma_{ii} z_i
      + 2\sum_{i<j}\Sigma_{ij} z_i z_j
    \right).
$$

The feasible sets are the three supports \(\{1,2\}\), \(\{1,3\}\), \(\{2,3\}\). Evaluating the expression for each support gives an exact ranking of discrete portfolios without any quantum hardware.

---

*Document purpose: provide a detailed, render-friendly mathematical backbone for a minimum-variance vs QAOA teaching experiment in Project 2.*
