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
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(3):
    ax.plot(v_vals, eig_j1[:, i], label=f"E{i}")
ax.set_xlabel("Interaction Strength V")
ax.set_ylabel("Energy")
ax.legend()
add_economist_signature(ax, "Lipkin Model J=1", subtitle="Eigenvalues for N=2 particles")
plt.savefig(os.path.join(plot_dir, "part-f_j1.pdf"))

# Plot J=2
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(5):
    ax.plot(v_vals, eig_j2[:, i], label=f"E{i}")
ax.set_xlabel("Interaction Strength V")
ax.set_ylabel("Energy")
ax.legend()
add_economist_signature(ax, "Lipkin Model J=2", subtitle="Eigenvalues for N=4 particles")
plt.savefig(os.path.join(plot_dir, "part-f_j2.pdf"))

# Plotting Comparison (E0 / N)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(v_vals, eig_j1[:, 0] / 2.0, color='#006BA2', label="J=1 (N=2)")
ax.plot(v_vals, eig_j2[:, 0] / 4.0, '--', color='#E3120B', label="J=2 (N=4)")
ax.set_xlabel("Interaction Strength V")
ax.set_ylabel("E0 / N")
ax.legend()
add_economist_signature(ax, "Lipkin Scaling Analysis", subtitle="Ground state energy per particle")
plt.savefig(os.path.join(plot_dir, "part-f_scaling.pdf"))
plt.show()

print("Lipkin model classical analysis and scaling comparison complete.")
print(f"Saved plots to {plot_dir}")
