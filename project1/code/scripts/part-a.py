import os
import numpy as np
import matplotlib.pyplot as plt

# Define plot path relative to script location
plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

def print_divider(title):
    print("\n" + "="*50)
    print(f" {title} ")
    print("="*50)

# 1. One-Qubit Basis & Pauli Matrices
print_divider("1. One-Qubit Basis & Pauli Matrices")

ket0 = np.array([1, 0], dtype=complex)
ket1 = np.array([0, 1], dtype=complex)

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

print(f"|0>:\n{ket0}")
print(f"|1>:\n{ket1}\n")

print(f"sigma_x |0>:\n{sigma_x @ ket0}")
print(f"sigma_x |1>:\n{sigma_x @ ket1}\n")

print(f"sigma_y |0>:\n{sigma_y @ ket0}")
print(f"sigma_y |1>:\n{sigma_y @ ket1}\n")

print(f"sigma_z |0>:\n{sigma_z @ ket0}")
print(f"sigma_z |1>:\n{sigma_z @ ket1}")

# 2. Hadamard & Phase Gates
print_divider("2. Hadamard & Phase Gates")

H = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)

print(f"H |0>:\n{H @ ket0}")
print(f"H |1>:\n{H @ ket1}\n")

print(f"S |0>:\n{S @ ket0}")
print(f"S |1>:\n{S @ ket1}")

# 3. Bell States
print_divider("3. Bell States")

# Define Kronecker product for basis states
# |00> = |0> \otimes |0>
ket00 = np.kron(ket0, ket0)
ket01 = np.kron(ket0, ket1)
ket10 = np.kron(ket1, ket0)
ket11 = np.kron(ket1, ket1)

phi_plus  = (ket00 + ket11) / np.sqrt(2)
phi_minus = (ket00 - ket11) / np.sqrt(2)
psi_plus  = (ket01 + ket10) / np.sqrt(2)
psi_minus = (ket01 - ket10) / np.sqrt(2)

print(f"|Phi+>:\n{phi_plus}")
print(f"|Phi->:\n{phi_minus}")
print(f"|Psi+>:\n{psi_plus}")
print(f"|Psi->:\n{psi_minus}")

# 4. Hadamard + CNOT on a Bell State & Measurements
print_divider("4. Hadamard + CNOT and Measurements")

# Start from |00>
state = ket00
# Apply Hadamard to first qubit: (H \otimes I)
H_I = np.kron(H, np.eye(2))
state = H_I @ state
# Apply CNOT: (control=0, target=1)
CNOT = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)
state = CNOT @ state

print(f"Final state (Expected |Phi+>):\n{state}\n")

# Measurement simulation
def simulate_measurements(state, num_runs=100, shots_per_run=1000):
    all_counts = []
    probs = np.abs(state)**2
    
    for _ in range(num_runs):
        outcomes = np.random.choice([0, 1, 2, 3], size=shots_per_run, p=probs)
        counts = np.bincount(outcomes, minlength=4) / shots_per_run
        all_counts.append(counts)
    
    return np.array(all_counts)

num_runs = 50
shots_per_run = 1000
measurement_data = simulate_measurements(state, num_runs, shots_per_run)

# Average results
avg_counts = np.mean(measurement_data, axis=0)
std_counts = np.std(measurement_data, axis=0)

print(f"Average probability distribution over {num_runs} runs:")
print(f"|00>: {avg_counts[0]:.4f} (+/- {std_counts[0]:.4f})")
print(f"|01>: {avg_counts[1]:.4f} (+/- {std_counts[1]:.4f})")
print(f"|10>: {avg_counts[2]:.4f} (+/- {std_counts[2]:.4f})")
print(f"|11>: {avg_counts[3]:.4f} (+/- {std_counts[3]:.4f})")

# Plotting the distribution
plt.figure(figsize=(10, 6))
labels = ['|00>', '|01>', '|10>', '|11>']
plt.bar(labels, avg_counts, yerr=std_counts, capsize=10, color='skyblue', edgecolor='navy')
plt.xlabel("Quantum State")
plt.ylabel("Probability")
plt.title(f"Measured State Distribution (Avg of {num_runs} runs, {shots_per_run} shots each)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(plot_dir, "part-a_measurement_dist.pdf"))
# plt.show()

print(f"\nSaved measurement distribution plot to {os.path.join(plot_dir, 'part-a_measurement_dist.pdf')}")

# 5. Density Matrix & von Neumann Entropy
print_divider("5. Density Matrix & von Neumann Entropy")

def von_neumann_entropy(state, subsystem_to_trace):
    # Construct density matrix
    rho = np.outer(state, np.conj(state))
    
    # Reshape for partial trace: (2, 2, 2, 2)
    # rho_abcd = <a,b| rho |c,d>
    rho_reshaped = rho.reshape(2, 2, 2, 2)
    
    if subsystem_to_trace == 0: # Trace out first qubit
        # rho_bd = sum_a rho_ab_ad
        rho_reduced = np.trace(rho_reshaped, axis1=0, axis2=2)
    else: # Trace out second qubit
        # rho_ac = sum_b rho_ab_cb
        rho_reduced = np.trace(rho_reshaped, axis1=1, axis2=3)
            
    # Compute eigenvalues
    eigvals = np.linalg.eigvalsh(rho_reduced)
    # Filter epsilon to avoid log(0)
    eigvals = eigvals[eigvals > 1e-15]
    entropy = -np.sum(eigvals * np.log2(eigvals))
    return entropy

ent_phi = von_neumann_entropy(phi_plus, 1)
print(f"Entropy of |Phi+> (expect 1): {ent_phi}")

# Entropy of a separable state |00>
ent_sep = von_neumann_entropy(ket00, 1)
print(f"Entropy of |00> (expect 0): {ent_sep}")
