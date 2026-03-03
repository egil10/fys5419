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

# Pauli Matrices
I = np.eye(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# 2-qubit operations
XX = np.kron(X, X)
YY = np.kron(Y, Y)
ZI = np.kron(Z, I)
IZ = np.kron(I, Z)

# Lipkin J=1 (N=2) mapping to 2-qubit subspace
# H = 0.5 * eps * (Z1 + Z2) + 0.25 * V * (X1X2 - Y1Y2)
def get_exact_j1(eps, V):
    H = 0.5 * eps * (ZI + IZ) + 0.25 * V * (XX - YY)
    return np.linalg.eigvalsh(H)[0]

# Ansatz: Ry(t1)Ry(t2) -> CNOT -> Ry(t3)Ry(t4)
def ansatz_2q(params):
    t1, t2, t3, t4 = params
    def Ry(theta):
        return np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
    
    s1 = np.kron(Ry(t1) @ np.array([1, 0]), Ry(t2) @ np.array([1, 0]))
    CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    s2 = CNOT @ s1
    s3 = np.kron(Ry(t3), Ry(t4)) @ s2
    return s3

def exp_val_j1(params, eps, V):
    state = ansatz_2q(params)
    exp_h = np.real(np.conj(state) @ (0.5 * eps * (ZI + IZ) + 0.25 * V * (XX - YY)) @ state)
    return exp_h

# Parameters
eps = 1.0
v_vals = np.linspace(0, 2.0, 21)
vqe_res = []
exact_res = []

for v in v_vals:
    exact_res.append(get_exact_j1(eps, v))
    
    # VQE
    init = np.random.rand(4) * 2 * np.pi
    res = minimize(exp_val_j1, init, args=(eps, v), method='L-BFGS-B')
    vqe_res.append(res.fun)

vqe_res = np.array(vqe_res)
exact_res = np.array(exact_res)

# Plotting J=1
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(v_vals, exact_res, '-', color='#006BA2', label="Exact GS")
ax.plot(v_vals, vqe_res, 'o', color='#E3120B', markersize=4, label="VQE GS")
ax.set_xlabel("V")
ax.set_ylabel("Energy")
ax.legend()
add_economist_signature(ax, "VQE Lipkin Simulation", subtitle="Ground state energy for J=1 (N=2)")
plt.savefig(os.path.join(plot_dir, "part-g_vqe_j1.pdf"))
plt.show()

print("Lipkin VQE analysis complete for J=1.")
print(f"Max absolute error: {np.max(np.abs(vqe_res - exact_res)):.4e}")
print(f"Saved plot to {os.path.join(plot_dir, 'part-g_vqe_j1.pdf')}")
