import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from plot_style import setup_economist_style, add_economist_signature

setup_economist_style()

# Define plot path relative to script location
plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Parameters
E1, E2 = 0.0, 4.0
V11, V12, V21, V22 = 3.0, 0.2, 0.2, -3.0

H0 = np.array([[E1, 0], [0, E2]])
HI = np.array([[V11, V12], [V21, V22]])

# Pauli matrices
I = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Decomposition H0 = eps*I + Omega*sigma_z
eps = (E1 + E2) / 2.0
Omega = (E1 - E2) / 2.0
print(f"H0 decomposition: eps={eps}, Omega={Omega}")
H0_decomp = eps*I + Omega*sigma_z
print(f"H0 original matches decomposition: {np.allclose(H0, H0_decomp)}")

# Decomposition HI = c*I + omega_z*sigma_z + omega_x*sigma_x
c = (V11 + V22) / 2.0
omega_z = (V11 - V22) / 2.0
omega_x = V12
print(f"HI decomposition: c={c}, omega_z={omega_z}, omega_x={omega_x}")
HI_decomp = c*I + omega_z*sigma_z + omega_x*sigma_x
print(f"HI original matches decomposition: {np.allclose(HI, HI_decomp)}")

# Solve for lambda in [0, 1]
lambdas = np.linspace(0, 1, 101)
eigvals = []
eigvecs = []

for l in lambdas:
    H = H0 + l * HI
    w, v = np.linalg.eigh(H)
    eigvals.append(w)
    eigvecs.append(v)

eigvals = np.array(eigvals)
eigvecs = np.array(eigvecs)

# Plotting Eigenvalues
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(lambdas, eigvals[:, 0], label="Ground State E-")
ax.plot(lambdas, eigvals[:, 1], label="Excited State E+")
ax.axvline(2/3, color='#E3120B', linestyle='--', alpha=0.7, label="lambda = 2/3")
ax.set_xlabel("Interaction Strength lambda")
ax.set_ylabel("Energy")
ax.legend()
add_economist_signature(ax, "Hamiltonian Eigenvalues", subtitle="Analysis of 2x2 one-qubit system")
plt.savefig(os.path.join(plot_dir, "part-b_eigenvalues.pdf"))

# Plotting Eigenvector Composition (Ground State)
fig, ax = plt.subplots(figsize=(8, 6))
gs_probs = np.abs(eigvecs[:, :, 0])**2 
ax.plot(lambdas, gs_probs[:, 0], label="Prob(|0>)")
ax.plot(lambdas, gs_probs[:, 1], label="Prob(|1>)")
ax.axvline(2/3, color='grey', linestyle='--', alpha=0.5)
ax.set_xlabel("Interaction Strength lambda")
ax.set_ylabel("Probability")
ax.legend()
add_economist_signature(ax, "Ground State Composition", subtitle="State mixing as a function of interaction")
plt.savefig(os.path.join(plot_dir, "part-b_eigenvector_mixing.pdf"))

plt.show()

print(f"\nSaved plots to {plot_dir}")
print(f"At lambda=2/3, E1={eigvals[67, 0]:.4f}, E2={eigvals[67, 1]:.4f}")
print("Level crossing and state mixing analysis complete.")
