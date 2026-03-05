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
eigvals_num = []
eigvals_ana = []
eigvecs = []

def analytical_eigenvalues(l):
    # H = [[3l, 0.2l], [0.2l, 4-3l]]
    # (3l-w)(4-3l-w) - (0.2l)^2 = 0
    # w^2 - (3l + 4-3l)w + 3l(4-3l) - 0.04l^2 = 0
    # w^2 - 4w + (12l - 9l^2 - 0.04l^2) = 0
    # w = [4 +- sqrt(16 - 4(12l - 9.04l^2))] / 2
    # w = 2 +- sqrt(4 - 12l + 9.04l^2)
    term = np.sqrt(4 - 12*l + 9.04*l**2)
    return 2 - term, 2 + term

for l in lambdas:
    # Numerical
    H = H0 + l * HI
    w, v = np.linalg.eigh(H)
    eigvals_num.append(w)
    eigvecs.append(v)
    
    # Analytical
    eigvals_ana.append(analytical_eigenvalues(l))

eigvals_num = np.array(eigvals_num)
eigvals_ana = np.array(eigvals_ana)
eigvecs = np.array(eigvecs)

# Plotting Eigenvalues
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(lambdas, eigvals_num[:, 0], '-', color='#006BA2', lw=2, label="Numerical E-")
ax.plot(lambdas, eigvals_num[:, 1], '-', color='#3E4345', lw=2, label="Numerical E+")
ax.plot(lambdas, eigvals_ana[:, 0], '--', color='#E3120B', alpha=0.6, label="Analytical E-")
ax.plot(lambdas, eigvals_ana[:, 1], '--', color='#E3120B', alpha=0.6, label="Analytical E+")

ax.axvline(2/3, color='#767676', linestyle=':', alpha=0.8, label="Crossing Point (2/3)")
ax.set_xlabel("Interaction Strength lambda")
ax.set_ylabel("Energy")
ax.legend()
add_economist_signature(ax, "Hamiltonian Eigenvalues", subtitle="Comparison of Numerical and Analytical solvers")
plt.savefig(os.path.join(plot_dir, "part-b_eigenvalues.pdf"))

# Plotting Eigenvector Composition (Ground State)
fig, ax = plt.subplots(figsize=(8, 6))
gs_probs = np.abs(eigvecs[:, :, 0])**2 
ax.plot(lambdas, gs_probs[:, 0], color='#006BA2', label="Prob(|0>)")
ax.plot(lambdas, gs_probs[:, 1], color='#E3120B', label="Prob(|1>)")
ax.axvline(2/3, color='#767676', linestyle=':', alpha=0.8)
ax.set_xlabel("Interaction Strength lambda")
ax.set_ylabel("Probability")
ax.legend()
add_economist_signature(ax, "Ground State Character", subtitle="State mixing showing the avoided crossing transition")
plt.savefig(os.path.join(plot_dir, "part-b_eigenvector_mixing.pdf"))

# plt.show()  # Disabled for non-interactive execution

# --- DISCUSSION ---
idx_two_thirds = np.argmin(np.abs(lambdas - 2/3))
idx_one = -1

print("\n" + "="*50)
print(" RESULTS DISCUSSION ")
print("="*50)
print(f"1. Analytical vs Numerical: Max discrepancy = {np.max(np.abs(eigvals_num - eigvals_ana)):.2e}")
print("\n2. State Characterization:")
print(f"   At lambda = 0.0:  Prob(|0>) = {gs_probs[0, 0]*100:.1f}%, Prob(|1>) = {gs_probs[0, 1]*100:.1f}%")
print(f"   At lambda = 2/3:  Prob(|0>) = {gs_probs[idx_two_thirds, 0]*100:.1f}%, Prob(|1>) = {gs_probs[idx_two_thirds, 1]*100:.1f}%")
print(f"   At lambda = 1.0:  Prob(|0>) = {gs_probs[idx_one, 0]*100:.1f}%, Prob(|1>) = {gs_probs[idx_one, 1]*100:.1f}%")

print("\n3. Insights:")
print("- The system exhibits an 'avoided crossing' at lambda = 2/3.")
print("- Below lambda=2/3, the ground state is predominantly |0> (unperturbed ground state).")
print("- Above lambda=2/3, the interaction term dominates, forcing the ground state to become |1>-like.")
print("- At lambda=1, the mixing of |0> is effectively ~1%, matching the theoretical prediction.")
print("="*50)

print(f"\nSaved plots to {plot_dir}")

