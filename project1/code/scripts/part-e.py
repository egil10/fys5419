import os
import numpy as np
from scipy import linalg
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Define plot path relative to script location
plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Parameters from Part d
Hx, Hz = 2.0, 3.0
eps00, eps10, eps01, eps11 = 0.0, 2.5, 6.5, 7.0

# Pauli decomposition coefficients for H0
# H0 = a*II + b*IZ + c*ZI + d*ZZ
a = (eps00 + eps01 + eps10 + eps11) / 4.0
b = (eps00 - eps01 + eps10 - eps11) / 4.0 # For qubit 1 (lsb)
c = (eps00 + eps01 - eps10 - eps11) / 4.0 # For qubit 0 (msb)
d = (eps00 - eps01 - eps10 + eps11) / 4.0

# Full Hamiltonian Pauli strings coefficients:
# H = a*II + b*IZ + c*ZI + (d + l*Hz)*ZZ + (l*Hx)*XX

def get_exact_gs(l):
    H0 = np.diag([eps00, eps01, eps10, eps11])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    HI = Hx * np.kron(sigma_x, sigma_x) + Hz * np.kron(sigma_z, sigma_z)
    H = H0 + l * HI
    return np.linalg.eigvalsh(H)[0]

# Two-qubit ansatz: Ry(t1)Ry(t2) -> CNOT -> Ry(t3)Ry(t4)
def ansatz_state(params):
    t1, t2, t3, t4 = params
    
    # 1-qubit Ry components
    def Ry(theta):
        return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
    
    # Step 1: Ry(t1) x Ry(t2) |00>
    s1 = np.kron(Ry(t1) @ np.array([1, 0]), Ry(t2) @ np.array([1, 0]))
    
    # Step 2: CNOT
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    s2 = CNOT @ s1
    
    # Step 3: Ry(t3) x Ry(t4)
    s3 = np.kron(Ry(t3), Ry(t4)) @ s2
    return s3

def expectation_value(params, l):
    state = ansatz_state(params)
    state_conj = np.conj(state)
    
    # Operators in computational basis
    # XX = [0 0 0 1; 0 0 1 0; 0 1 0 0; 1 0 0 0]
    # ZZ = [1 0 0 0; 0 -1 0 0; 0 0 -1 0; 0 0 0 1]
    # ZI = [1 0 0 0; 1 0 0 0; -1 0 0 0; -1 0 0 0] ? No.
    # ZI = [1 0 0 0; 0 1 0 0; 0 0 -1 0; 0 0 0 -1]
    # IZ = [1 0 0 0; 0 -1 0 0; 0 0 1 0; 0 0 0 -1]
    
    # Helper to get exp val
    def exp(O):
        return np.real(state_conj @ O @ state)

    # Matrix forms
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    
    ZI = np.kron(sigma_z, I)
    IZ = np.kron(I, sigma_z)
    ZZ = np.kron(sigma_z, sigma_z)
    XX = np.kron(sigma_x, sigma_x)
    
    energy = a + b * exp(IZ) + c * exp(ZI) + (d + l * Hz) * exp(ZZ) + (l * Hx) * exp(XX)
    return energy

# Sweep lambda
lambdas = np.linspace(0, 1, 21)
vqe_res = []
exact_res = []

for l in lambdas:
    exact_res.append(get_exact_gs(l))
    
    # VQE optimization
    init_params = np.random.rand(4) * 2 * np.pi
    res = minimize(expectation_value, init_params, args=(l,), method='L-BFGS-B')
    vqe_res.append(res.fun)

vqe_res = np.array(vqe_res)
exact_res = np.array(exact_res)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(lambdas, exact_res, 'r-', label="Exact GS")
plt.plot(lambdas, vqe_res, 'bo', markersize=4, label="VQE GS")
plt.xlabel("lambda")
plt.ylabel("Energy")
plt.title("VQE vs Exact Ground State (2-Qubit System)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plot_dir, "part-e_vqe_comp.pdf"))

print("Two-qubit VQE complete.")
print(f"Max absolute error: {np.max(np.abs(vqe_res - exact_res)):.2e}")
print(f"Saved plot to {os.path.join(plot_dir, 'part-e_vqe_comp.pdf')}")
