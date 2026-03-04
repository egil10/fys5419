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

# --- Hamiltonian Definition ---
# System: Two qubits A and B
# Basis: |00>, |10>, |01>, |11> (lexicographical relative to the prompt matrix)

# Parameters
Hx = 2.0
Hz = 3.0
# Non-interacting energies: eps_00, eps_10, eps_01, eps_11
eps00, eps10, eps01, eps11 = 0.0, 2.5, 6.5, 7.0

def get_hamiltonian(l):
    """Returns the Hamiltonian matrix H(l) = H0 + l*HI"""
    # Diagonal elements (H0 + l*Hz*SzSz)
    d00 = eps00 + l * Hz
    d10 = eps10 - l * Hz
    d01 = eps01 - l * Hz
    d11 = eps11 + l * Hz
    
    H = np.zeros((4, 4))
    np.fill_diagonal(H, [d00, d10, d01, d11])
    
    # Interaction terms (l*Hx*SxSx)
    H[0, 3] = H[3, 0] = l * Hx
    H[1, 2] = H[2, 1] = l * Hx
    
    return H

def get_von_neumann_entropy(state):
    """
    Computes von Neumann entropy S = -Tr(rho_A * log2(rho_A)).
    State is in basis |00>, |10>, |01>, |11>.
    We treat index 0 (first qubit A) and index 1 (second qubit B).
    """
    # Reshape vector to 2x2 matrix (subsystem A is rows, subsystem B is columns)
    # Basis: |0>A|0>B, |1>A|0>B, |0>A|1>B, |1>A|1>B (indices: 00, 10, 01, 11)
    # Re-ordering to standard |00>, |01>, |10>, |11> for easier tracing
    # psi = alpha_00|00> + alpha_10|10> + alpha_01|01> + alpha_11|11>
    # Matrix form (rows=A, cols=B):
    # [[alpha_00, alpha_01],
    #  [alpha_10, alpha_11]]
    psi_mat = np.array([[state[0], state[2]], 
                        [state[1], state[3]]])
    
    # Reduced density matrix rho_A = Tr_B(|psi><psi|) = psi_mat @ psi_mat.T.conj()
    rho_A = psi_mat @ psi_mat.T.conj()
    
    # Eigenvalues for entropy
    eigvals = np.linalg.eigvalsh(rho_A)
    # Filter for numerical stability
    eigvals = eigvals[eigvals > 1e-15]
    return -np.sum(eigvals * np.log2(eigvals))

# --- Sweep lambda ---
lambdas = np.linspace(0, 1, 201)
energies = []
entropies = []

print("Analyzing two-qubit system over lambda range...")
for l in lambdas:
    H = get_hamiltonian(l)
    w, v = np.linalg.eigh(H)
    
    energies.append(w)
    # Ground state entropy
    entropies.append(get_von_neumann_entropy(v[:, 0]))

energies = np.array(energies)
entropies = np.array(entropies)

# --- Plotting ---
# 1. Eigenvalues
fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#006BA2', '#E3120B', '#3E4345', '#37A635']
for i in range(4):
    ax.plot(lambdas, energies[:, i], color=colors[i], label=f"State {i}")

ax.set_xlabel("Interaction Strength lambda")
ax.set_ylabel("Energy")
ax.legend(title="Eigenstates")
add_economist_signature(ax, "Two-Qubit State Spectrum", subtitle="Energy levels showing interaction effects and avoided crossings")
plt.savefig(os.path.join(plot_dir, "part-d_eigenvalues.pdf"))

# 2. Entropy
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(lambdas, entropies, color='#E3120B', lw=2)
ax.set_xlabel("Interaction Strength lambda")
ax.set_ylabel("von Neumann Entropy (bits)")
add_economist_signature(ax, "Ground State Entanglement", subtitle="Entropy jump marking a phase/character transition")
plt.savefig(os.path.join(plot_dir, "part-d_entropy.pdf"))

# plt.show()

# --- DISCUSSION ---
print("\n" + "="*60)
print(" DISCUSSION OF TWO-QUBIT RESULTS ")
print("="*60)
print("1. SPECTRUM DYNAMICS:")
print("   As lambda increases, the interaction terms (Hx and Hz) lift degeneracies")
print("   and cause energy levels to shift. The ground state character changes")
print("   drastically near the 'jump' point in entropy.")

print("\n2. ENTANGLEMENT & LEVEL CROSSINGS:")
print("   At lambda = 0, entropy is 0.0 because the ground state is a pure computational")
print("   basis state (|00>). As interaction grows, the state becomes a superposition.")
print("   The sharp increase (jump) in entropy marks a point where the ground state")
print("   undergoes a significant change in composition, often termed an 'avoided crossing'.")
print(f"   Peak Entropy observed: {np.max(entropies):.4f} bits.")

print("\n3. IMPLICATION:")
print("   Entanglement is intrinsically driven by the Hamiltonian's off-diagonal elements")
print("   relative to the unperturbed energy gaps. The 'jump' identifies regions where")
print("   small changes in lambda lead to massive shifts in quantum correlation.")
print("="*60)

print(f"\nPlots saved to {plot_dir}")

