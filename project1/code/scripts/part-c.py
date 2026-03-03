import os
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define plot path relative to script location
plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Classical reference from Part b
E1, E2 = 0.0, 4.0
V11, V12, V21, V22 = 3.0, 0.2, 0.2, -3.0

H0 = np.array([[E1, 0], [0, E2]])
HI = np.array([[V11, V12], [V21, V22]])

def get_exact_eigenvalues(l):
    H = H0 + l * HI
    w = np.linalg.eigvalsh(H)
    return w

# Pauli decomposition coefficients (from part b)
# H(l) = eps*I + (l*omega_z + Omega)*sigma_z + l*omega_x*sigma_x
eps = 2.0
Omega = -2.0
omega_z = 3.0
omega_x = 0.2

# Ansatz: |psi(theta)> = Ry(theta)|0> = [cos(theta/2), sin(theta/2)]
# <sigma_z> = cos(theta)
# <sigma_x> = sin(theta)

def expectation_value(theta, l):
    # Pauli expectation values
    z_exp = np.cos(theta)
    x_exp = np.sin(theta)
    
    # Energy
    energy = eps + (l * omega_z + Omega) * z_exp + (l * omega_x) * x_exp
    return energy

def vqe_solver(l):
    # Initial guess for theta
    theta_init = 0.0
    res = minimize(expectation_value, theta_init, args=(l,), method='BFGS')
    return res.fun, res.x[0]

# Sweep lambda from 0 to 1
lambdas = np.linspace(0, 1, 51)
vqe_energies = []
exact_energies = []

for l in lambdas:
    exact = get_exact_eigenvalues(l)
    vqe_energy, opt_theta = vqe_solver(l)
    
    vqe_energies.append(vqe_energy)
    exact_energies.append(exact[0]) # Ground state

vqe_energies = np.array(vqe_energies)
exact_energies = np.array(exact_energies)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(lambdas, exact_energies, 'r-', label="Exact Ground State")
plt.plot(lambdas, vqe_energies, 'bo', markersize=4, label="VQE Ground State")
plt.xlabel("lambda")
plt.ylabel("Energy")
plt.title("VQE vs. Exact Eigenvalues for 2x2 Hamiltonian")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "part-c_vqe_comp.pdf"))

# Error plot
plt.figure(figsize=(10, 6))
plt.semilogy(lambdas, np.abs(vqe_energies - exact_energies) + 1e-16, 'g-')
plt.xlabel("lambda")
plt.ylabel("Absolute Error")
plt.title("VQE Error relative to Exact Ground State")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "part-c_vqe_error.pdf"))

print("VQE simulation complete.")
print(f"Max error across lambda sweep: {np.max(np.abs(vqe_energies - exact_energies)):.2e}")
print(f"Saved plots to {os.path.join(plot_dir, 'part-c_vqe_comp.pdf')} and {os.path.join(plot_dir, 'part-c_vqe_error.pdf')}")
