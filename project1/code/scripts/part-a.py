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
def measure_qubit(state, qubit_idx, num_shots=1000):
    probs = np.abs(state)**2
    outcomes = np.random.choice([0, 1, 2, 3], size=num_shots, p=probs)
    # outcomes: 0=|00>, 1=|01>, 2=|10>, 3=|11>
    # Qubit 0 is msb, Qubit 1 is lsb
    if qubit_idx == 0:
        results = (outcomes >= 2).astype(int) # 2 and 3 have qubit0 = 1
    else:
        results = (outcomes % 2).astype(int) # 1 and 3 have qubit1 = 1
    return results

res0 = measure_qubit(state, 0)
res1 = measure_qubit(state, 1)

print(f"Qubit 0 mean: {np.mean(res0)}")
print(f"Qubit 1 mean: {np.mean(res1)}")
print(f"Correlation: {np.mean(res0 == res1)}")

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
