import os
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from plot_style import setup_economist_style, add_economist_signature
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import Aer

setup_economist_style()

# Define plot path relative to script location
plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Hamiltonian parameters from Part b
# H = H0 + lambda * HI
# H0 = [[E1, 0], [0, E2]]
# HI = [[V11, V12], [V21, V22]]
E1, E2 = 0.0, 4.0
V11, V12, V21, V22 = 3.0, 0.2, 0.2, -3.0

# Pauli decomposition coefficients (from H = cI + cx*Sx + cy*Sy + cz*Sz)
# H(l) = (eps + l*c)*I + (Omega + l*omega_z)*sigma_z + (l*omega_x)*sigma_x
eps = (E1 + E2) / 2.0        # 2.0
Omega = (E1 - E2) / 2.0      # -2.0
c = (V11 + V22) / 2.0        # 0.0
omega_z = (V11 - V22) / 2.0  # 3.0
omega_x = V12                # 0.2

def get_exact_ground_state(l):
    H0_mat = np.array([[E1, 0], [0, E2]])
    HI_mat = np.array([[V11, V12], [V21, V22]])
    H = H0_mat + l * HI_mat
    w = np.linalg.eigvalsh(H)
    return w[0]

# --- VQE Implementation (My Own Code) ---

def ansatz_circuit_sim(theta):
    """Simulates the state |psi(theta)> = Ry(theta)|0>"""
    return np.array([np.cos(theta/2), np.sin(theta/2)])

def measure_expectation_values(theta):
    """
    Simulates measurement of <sigma_z> and <sigma_x>.
    For <sigma_z>, we measure in the computational basis.
    For <sigma_x>, we apply a Hadamard gate before measuring.
    """
    psi = ansatz_circuit_sim(theta)
    
    # <sigma_z> = |<0|psi>|^2 - |<1|psi>|^2
    z_exp = np.abs(psi[0])**2 - np.abs(psi[1])**2
    
    # <sigma_x>: Apply H -> 1/sqrt(2) * [[1, 1], [1, -1]]
    h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    psi_x = h_gate @ psi
    x_exp = np.abs(psi_x[0])**2 - np.abs(psi_x[1])**2
    
    return z_exp, x_exp

def manual_vqe_energy(theta, l):
    z_exp, x_exp = measure_expectation_values(theta)
    
    # Hamiltonian coefficients
    c_tot = eps + l * c
    cz_tot = Omega + l * omega_z
    cx_tot = l * omega_x
    
    return c_tot + cz_tot * z_exp + cx_tot * x_exp

def manual_vqe_solver(l):
    res = minimize(manual_vqe_energy, 0.0, args=(l,), method='BFGS')
    return res.fun

# --- Qiskit Implementation for Comparison ---

def qiskit_vqe_energy(theta, l):
    # Define Hamiltonian as SparsePauliOp
    c_tot = eps + l * c
    cz_tot = Omega + l * omega_z
    cx_tot = l * omega_x
    
    h_op = SparsePauliOp.from_list([
        ("I", c_tot),
        ("Z", cz_tot),
        ("X", cx_tot)
    ])
    
    # Set up the circuit Ansatz
    qc = QuantumCircuit(1)
    qc.ry(float(theta), 0)
    
    # Calculate expectation value using Statevector (exact simulation)
    psi = Statevector.from_instruction(qc)
    energy = psi.expectation_value(h_op).real
    return energy

def qiskit_vqe_solver(l):
    res = minimize(qiskit_vqe_energy, 0.0, args=(l,), method='BFGS')
    return res.fun

# --- Execution and Comparison ---

lambdas = np.linspace(0, 1, 31)
exact_gs = []
exact_es = []
manual_vqe_gs = []
manual_vqe_es = []

print("Starting VQE simulation sweep (Ground and Excited states)...")
for l in lambdas:
    H0_mat = np.array([[E1, 0], [0, E2]])
    HI_mat = np.array([[V11, V12], [V21, V22]])
    H = H0_mat + l * HI_mat
    w = np.linalg.eigvalsh(H)
    exact_gs.append(w[0])
    exact_es.append(w[1])
    
    # Minimize for ground state
    res = minimize(manual_vqe_energy, 0.0, args=(l,), method='BFGS')
    theta_opt = res.x[0]
    manual_vqe_gs.append(res.fun)
    
    # Excited state for 1-qubit is orthogonal state: theta + pi
    manual_vqe_es.append(manual_vqe_energy(theta_opt + np.pi, l))

exact_gs = np.array(exact_gs)
exact_es = np.array(exact_es)
manual_vqe_gs = np.array(manual_vqe_gs)
manual_vqe_es = np.array(manual_vqe_es)

# --- Plotting ---

fig, ax = plt.subplots(figsize=(10, 6))
# Exact
ax.plot(lambdas, exact_gs, '-', lw=2.5, color='#006BA2', label="Exact GS")
ax.plot(lambdas, exact_es, '-', lw=2.5, color='#3E4345', label="Exact ES")
# VQE
ax.plot(lambdas, manual_vqe_gs, 'o', markersize=6, color='#E3120B', label="VQE GS")
ax.plot(lambdas, manual_vqe_es, 's', markersize=5, color='#E3120B', alpha=0.6, label="VQE ES (orthog)")

ax.set_xlabel("Interaction Strength lambda")
ax.set_ylabel("Energy")
ax.legend()
add_economist_signature(ax, "VQE Full Spectrum Comparison", subtitle="Ground and Excited states for 1-qubit system")
plt.savefig(os.path.join(plot_dir, "part-c_vqe_full_spectrum.pdf"))

# Plot Error
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(lambdas, np.abs(manual_vqe_gs - exact_gs) + 1e-16, 'o-', color='#006BA2', label="GS Error")
ax.semilogy(lambdas, np.abs(manual_vqe_es - exact_es) + 1e-16, 's--', color='#E3120B', label="ES Error")
ax.set_xlabel("lambda")
ax.set_ylabel("Energy Error (log scale)")
ax.legend()
add_economist_signature(ax, "VQE Precision", subtitle="Numerical error for both states")
plt.savefig(os.path.join(plot_dir, "part-c_vqe_spectrum_error.pdf"))

# plt.show()  # Disabled for non-interactive execution

print("\n--- DISCUSSION OF RESULTS ---")
print("1. Consistency: The VQE implementation yields results highly consistent with exact diagonalization for both states.")
print(f"   Max GS error: {np.max(np.abs(manual_vqe_gs - exact_gs)):.2e}")
print(f"   Max ES error: {np.max(np.abs(manual_vqe_es - exact_es)):.2e}")
print("2. Circuit Setup: The ansatz used is |psi(theta)> = Ry(theta)|0>. For a single-qubit problem with real Hamiltonian,")
print("   this is sufficient. Measurement in the Z-basis gives <sigma_z>, and measurement in the X-basis")
print("   (applying H before measurement) gives <sigma_x>.")
print("3. Excited States: For this simple 2x2 system, the excited state is uniquely determined as the state")
print("   orthogonal to the ground state. In the Ry(theta) parameterization, this corresponds to theta + pi.")
print("4. Comparison with Part b): The VQE perfectly reproduces the level mixing and energy curves from Part b.")
print(f"Results saved to {plot_dir}")


