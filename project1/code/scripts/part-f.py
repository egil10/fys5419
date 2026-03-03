import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

# Define plot path relative to script location
plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def get_quasispin_ops(J):
    dim = int(2*J + 1)
    m_vals = np.arange(J, -J-1, -1) # J, J-1, ..., -J
    
    Jz = np.diag(m_vals)
    Jplus = np.zeros((dim, dim))
    Jminus = np.zeros((dim, dim))
    
    for i, m in enumerate(m_vals):
        # J+ |J, m> = sqrt(J(J+1) - m(m+1)) |J, m+1>
        if m < J:
            val = np.sqrt(J*(J+1) - m*(m+1))
            Jplus[i-1, i] = val
        # J- |J, m> = sqrt(J(J+1) - m(m-1)) |J, m-1>
        if m > -J:
            val = np.sqrt(J*(J+1) - m*(m-1))
            Jminus[i+1, i] = val
            
    return Jz, Jplus, Jminus

def lipkin_hamiltonian(J, eps, V, W=0):
    Jz, Jp, Jm = get_quasispin_ops(J)
    N = 2 * J
    
    H0 = eps * Jz
    H1 = 0.5 * V * (Jp @ Jp + Jm @ Jm)
    H2 = 0.5 * W * (-N * np.eye(int(2*J+1)) + Jp @ Jm + Jm @ Jp)
    
    return H0 + H1 + H2

# Parameters
eps = 1.0
W = 0.0
v_vals = np.linspace(0, 2.0, 101)

# Solve for J=1
eig_j1 = []
for v in v_vals:
    H = lipkin_hamiltonian(1, eps, v, W)
    eig_j1.append(np.linalg.eigvalsh(H))
eig_j1 = np.array(eig_j1)

# Solve for J=2
eig_j2 = []
for v in v_vals:
    H = lipkin_hamiltonian(2, eps, v, W)
    eig_j2.append(np.linalg.eigvalsh(H))
eig_j2 = np.array(eig_j2)

# Plot J=1
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(v_vals, eig_j1[:, i], label=f"E{i}")
plt.xlabel("Interaction Strength V")
plt.ylabel("Eigenvalues")
plt.title("Lipkin Model Eigenvalues for J=1 (N=2)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "part-f_j1.pdf"))

# Plot J=2
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(v_vals, eig_j2[:, i], label=f"E{i}")
plt.xlabel("Interaction Strength V")
plt.ylabel("Eigenvalues")
plt.title("Lipkin Model Eigenvalues for J=2 (N=4)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "part-f_j2.pdf"))

# Plotting Comparison (E0 / N)
plt.figure(figsize=(10, 6))
plt.plot(v_vals, eig_j1[:, 0] / 2.0, 'b-', label="J=1 (N=2)")
plt.plot(v_vals, eig_j2[:, 0] / 4.0, 'r--', label="J=2 (N=4)")
plt.xlabel("Interaction Strength V")
plt.ylabel("Ground State Energy per Particle (E0/N)")
plt.title("Lipkin Model: Scaling of Ground State Energy")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "part-f_scaling.pdf"))

print("Lipkin model classical analysis and scaling comparison complete.")
print(f"Saved plots to {plot_dir}")
