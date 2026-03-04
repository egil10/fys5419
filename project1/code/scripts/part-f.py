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
    # Define m in increasing order: -J, -J+1, ..., J to match standard matrix forms
    m_vals = np.arange(-J, J + 1) 
    
    Jz = np.diag(m_vals)
    Jplus = np.zeros((dim, dim))
    Jminus = np.zeros((dim, dim))
    
    for i in range(dim):
        m = m_vals[i]
        # J+ |J, m> = sqrt(J(J+1) - m(m+1)) |J, m+1>
        if i + 1 < dim:
            val = np.sqrt(J*(J+1) - m*(m+1))
            Jplus[i+1, i] = val
        # J- |J, m> = sqrt(J(J+1) - m(m-1)) |J, m-1>
        if i - 1 >= 0:
            val = np.sqrt(J*(J+1) - m*(m-1))
            Jminus[i-1, i] = val
            
    return Jz, Jplus, Jminus

def lipkin_hamiltonian(J, eps, V, W=0):
    Jz, Jp, Jm = get_quasispin_ops(J)
    N = 2 * J
    
    H0 = eps * Jz
    H1 = 0.5 * V * (Jp @ Jp + Jm @ Jm)
    # Challenge: Adding the W term
    # H2 = 1/2 W (-N/2 + (J+J- + J-J+)/2) is slightly different in some texts, 
    # but we follow the prompt's quasispin form:
    H2 = 0.5 * W * (-N * np.eye(int(2*J+1)) + Jp @ Jm + Jm @ Jp)
    
    return H0 + H1 + H2

def calculate_hf_energy(J, eps, V):
    """
    Calculates the Hartree-Fock ground state energy for the Lipkin model.
    E_HF = -1/2 * N * eps * cos(alpha) - 1/2 * N * (N-1) * V/2 * sin^2(alpha)
    Transition at V_crit = eps / (N-1)
    """
    N = 2 * J
    v_tilde = (N - 1) * V / eps
    
    if np.abs(v_tilde) <= 1.0:
        return -0.5 * N * eps
    else:
        # E_HF = -1/4 * N * eps * ( (N-1)V/eps + eps/(N-1)V )
        return -0.25 * N * eps * (v_tilde + 1.0/v_tilde)

# --- Verification & Simulation ---
eps = 1.0
W = 0.0
v_vals = np.linspace(0, 1.5, 101)

# J=1 (N=2)
eig_j1 = []
hf_j1 = []
for v in v_vals:
    H = lipkin_hamiltonian(1, eps, -v, W) # Using -v to match prompt's sign
    eig_j1.append(np.linalg.eigvalsh(H))
    hf_j1.append(calculate_hf_energy(1, eps, v))
eig_j1 = np.array(eig_j1)

# J=2 (N=4)
eig_j2 = []
hf_j2 = []
for v in v_vals:
    H = lipkin_hamiltonian(2, eps, -v, W)
    eig_j2.append(np.linalg.eigvalsh(H))
    hf_j2.append(calculate_hf_energy(2, eps, v))
eig_j2 = np.array(eig_j2)

# --- Plotting ---

# 1. J=1 Spectrum
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#006BA2', '#3E4345', '#37A635']
for i in range(3):
    ax.plot(v_vals, eig_j1[:, i], label=f"State {i}", color=colors[i])
ax.plot(v_vals, hf_j1, '--', color='#E3120B', alpha=0.8, label="Hartree-Fock GS")
ax.axvline(eps/(2-1), color='grey', ls=':', alpha=0.5, label="Crit. V (N=2)")
ax.set_xlabel("Interaction Strength V")
ax.set_ylabel("Energy")
ax.legend()
add_economist_signature(ax, "Lipkin Model J=1 (N=2)", subtitle="Exact vs Hartree-Fock ground state comparison")
plt.savefig(os.path.join(plot_dir, "part-f_j1_spectrum.pdf"))

# 2. J=2 Spectrum
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(5):
    ax.plot(v_vals, eig_j2[:, i], label=f"State {i}")
ax.plot(v_vals, hf_j2, '--', color='#E3120B', lw=2, label="Hartree-Fock GS")
ax.axvline(eps/(4-1), color='grey', ls=':', alpha=0.5, label="Crit. V (N=4)")
ax.set_xlabel("Interaction Strength V")
ax.set_ylabel("Energy")
ax.legend()
add_economist_signature(ax, "Lipkin Model J=2 (N=4)", subtitle="Evolution of the 5x5 Hamiltonian spectrum")
plt.savefig(os.path.join(plot_dir, "part-f_j2_spectrum.pdf"))

# 3. Scaling Comparison (E0 / N)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(v_vals, eig_j1[:, 0] / 2.0, color='#006BA2', label="J=1 (N=2) Exact")
ax.plot(v_vals, eig_j2[:, 0] / 4.0, color='#37A635', label="J=2 (N=4) Exact")
ax.plot(v_vals, np.array(hf_j2)/4.0, ':', color='#E3120B', lw=2, label="HF (Large N limit)")
ax.set_xlabel("Interaction Strength V")
ax.set_ylabel("Ground State Energy / N")
ax.legend()
add_economist_signature(ax, "Lipkin Scaling Analysis", subtitle="Energy per particle and HF approximation")
plt.savefig(os.path.join(plot_dir, "part-f_scaling.pdf"))

# plt.show() # Disabled for non-interactive execution

# --- DISCUSSION ---
print("\n" + "="*60)
print(" DISCUSSION OF LIPKIN MODEL RESULTS ")
print("="*60)
print("1. MATRIX VERIFICATION:")
print(f"   For J=1, V=0.5, H_J1 is:\n{lipkin_hamiltonian(1, eps, -0.5, 0)}")
print("   Matches the prompt format: [[-eps, 0, -V], [0, 0, 0], [-V, 0, eps]]")

print("\n2. PHASE TRANSITION:")
print("   The Lipkin model exhibits a second-order phase transition.")
print("   The critical points occur at V_c = eps / (N - 1).")
print(f"   For N=2, V_c = {eps/1:.2f}. For N=4, V_c = {eps/3:.2f}.")
print("   In the plots, these points correspond to where the ground state")
print("   energy starts decreasing significantly and HF deviates from -0.5*N*eps.")

print("\n3. HARTREE-FOCK PERFORMANCE:")
print("   The HF approximation provides an upper bound and becomes more accurate")
print("   as the particle number N increases (thermodynamic limit).")
print("   The scaling plot shows J=2 is closer to the HF curve than J=1.")

print("\n4. SYMMETRY:")
print("   The W term (when added) acts as a shift and polarization term.")
print("   For W=0, the Hamiltonian has parity symmetry, causing level crossings.")
print("="*60)

print(f"\nClassical analysis of Lipkin Model complete. Plots saved to {plot_dir}")

