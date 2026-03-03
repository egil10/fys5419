import os
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from plot_style import setup_economist_style, add_economist_signature

setup_economist_style()

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
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lambdas, exact_energies, '-', color='#006BA2', label="Exact Ground State")
ax.plot(lambdas, vqe_energies, 'o', color='#E3120B', markersize=4, label="VQE Ground State")
ax.set_xlabel("lambda")
ax.set_ylabel("Energy")
ax.legend()
add_economist_signature(ax, "VQE vs. Exact Results", subtitle="Ground state energy of 2x2 one-qubit Hamiltonian")
plt.savefig(os.path.join(plot_dir, "part-c_vqe_comp.pdf"))

# Error plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(lambdas, np.abs(vqe_energies - exact_energies) + 1e-16, color='#37A635')
ax.set_xlabel("lambda")
ax.set_ylabel("Absolute Error")
add_economist_signature(ax, "VQE Numerical Error", subtitle="Precision across interaction range")
plt.savefig(os.path.join(plot_dir, "part-c_vqe_error.pdf"))

# Visualize Energy Landscape for lambda = 1.0
fig, ax = plt.subplots(figsize=(10, 6))
l_plot = 1.0
thetas = np.linspace(-np.pi, np.pi, 200)
energies_land = [expectation_value(t, l_plot) for t in thetas]
vqe_val, vqe_theta = vqe_solver(l_plot)

ax.plot(thetas, energies_land, color='#006BA2', label=f"E(theta) for lambda={l_plot}")
ax.plot(vqe_theta, vqe_val, 'o', color='#E3120B', label="VQE Minimum")
ax.set_xlabel("Variational Parameter theta")
ax.set_ylabel("Expected Energy")
ax.legend()
add_economist_signature(ax, "VQE Energy Landscape", subtitle=f"Optimization surface for lambda={l_plot}")
plt.savefig(os.path.join(plot_dir, "part-c_energy_landscape.pdf"))
plt.show()

print("VQE simulation and landscape visualization complete.")
print(f"Max error across lambda sweep: {np.max(np.abs(vqe_energies - exact_energies)):.2e}")
print(f"Saved plots to {plot_dir}")
