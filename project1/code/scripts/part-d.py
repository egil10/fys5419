import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Define plot path relative to script location
plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Basis and Operators
I = np.eye(2)
sigma_x = np.array([[0, 1], [1, 0]])
sigma_z = np.array([[1, 0], [0, -1]])

# Hamiltonian Parameters
Hx = 2.0
Hz = 3.0
# eps_00, eps_10, eps_01, eps_11 from part-d.ipynb.
# Computational basis order usually |00>, |01>, |10>, |11>
# eps_00 = 0.0, eps_01 = 6.5, eps_10 = 2.5, eps_11 = 7.0
eps00, eps10, eps01, eps11 = 0.0, 2.5, 6.5, 7.0
H0 = np.diag([eps00, eps01, eps10, eps11])

# Interaction Hamiltonian
HI = Hx * np.kron(sigma_x, sigma_x) + Hz * np.kron(sigma_z, sigma_z)

def von_neumann_entropy(state, subsystem_to_trace):
    rho = np.outer(state, np.conj(state))
    rho_reshaped = rho.reshape(2, 2, 2, 2)
    if subsystem_to_trace == 0:
        rho_reduced = np.trace(rho_reshaped, axis1=0, axis2=2)
    else:
        rho_reduced = np.trace(rho_reshaped, axis1=1, axis2=3)
        
    eigvals = np.linalg.eigvalsh(rho_reduced)
    eigvals = eigvals[eigvals > 1e-15]
    entropy = -np.sum(eigvals * np.log2(eigvals))
    return entropy

# Sweep lambda from 0 to 1
lambdas = np.linspace(0, 1, 101)
all_eigvals = []
entropies = []

for l in lambdas:
    H = H0 + l * HI
    w, v = np.linalg.eigh(H)
    all_eigvals.append(w)
    
    # Ground state entropy
    gs = v[:, 0]
    entropies.append(von_neumann_entropy(gs, 0))

all_eigvals = np.array(all_eigvals)

# Plotting Eigenvalues
plt.figure(figsize=(10, 6))
for i in range(4):
    plt.plot(lambdas, all_eigvals[:, i], label=f"E{i}")
plt.xlabel("lambda")
plt.ylabel("Eigenvalues")
plt.title("Two-Qubit Hamiltonian Eigenvalues vs. lambda")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "part-d_eigenvalues.pdf"))
# plt.show()

# Plotting Entropy
plt.figure(figsize=(10, 6))
plt.plot(lambdas, entropies, 'm-')
plt.xlabel("lambda")
plt.ylabel("von Neumann Entropy S(A)")
plt.title("Ground State Entanglement Entropy vs. lambda")
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "part-d_entropy.pdf"))
# plt.show()

print("Two-qubit analysis complete.")
print(f"Saved plots to {os.path.join(plot_dir, 'part-d_eigenvalues.pdf')} and {os.path.join(plot_dir, 'part-d_entropy.pdf')}")
