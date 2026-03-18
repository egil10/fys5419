import os
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import matplotlib as mpl

# Set matplotlib to use math font that looks like LaTeX
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Computer Modern Roman', 'serif']

# Minimalist black and white style
style_bw = {
    "name": "bw",
    "fontsize": 14,
    "subfontsize": 11,
    "backgroundcolor": "#FFFFFF",
}

# Define plot path
plot_dir = os.path.join(os.path.dirname(__file__), "..", "plots", "circuits")
os.makedirs(plot_dir, exist_ok=True)

# ==========================================
# Part C: 1-Qubit Ansatz
# ==========================================
qc_c = QuantumCircuit(1)
# Using $ syntax renders LaTeX-like math text in matplotlib
qc_c.ry(Parameter('$\\theta_0$'), 0)

fig_c = qc_c.draw(output='mpl', style=style_bw)
fig_c.savefig(os.path.join(plot_dir, "circuit_part_c.pdf"), bbox_inches='tight', pad_inches=0.0)
print("Saved circuit_part_c.pdf")

# ==========================================
# Part E: 2-Qubit Ansatz
# ==========================================
qc_e = QuantumCircuit(2)
qc_e.ry(Parameter('$\\theta_0$'), 0)
qc_e.ry(Parameter('$\\theta_1$'), 1)
qc_e.cx(0, 1)
qc_e.ry(Parameter('$\\theta_2$'), 0)
qc_e.ry(Parameter('$\\theta_3$'), 1)

fig_e = qc_e.draw(output='mpl', style=style_bw)
fig_e.savefig(os.path.join(plot_dir, "circuit_part_e.pdf"), bbox_inches='tight', pad_inches=0.0)
print("Saved circuit_part_e.pdf")

# ==========================================
# Part E: 2-Qubit Ansatz (1-indexed matching report eq)
# ==========================================
qc_e_1idx = QuantumCircuit(2)
qc_e_1idx.ry(Parameter('$\\theta_1$'), 0)
qc_e_1idx.ry(Parameter('$\\theta_2$'), 1)
qc_e_1idx.cx(0, 1)
qc_e_1idx.ry(Parameter('$\\theta_3$'), 0)
qc_e_1idx.ry(Parameter('$\\theta_4$'), 1)

fig_e_1idx = qc_e_1idx.draw(output='mpl', style=style_bw)
fig_e_1idx.savefig(os.path.join(plot_dir, "circuit_part_e_1idx.pdf"), bbox_inches='tight', pad_inches=0.0)
print("Saved circuit_part_e_1idx.pdf")

# ==========================================
# Part G: Many-Body Ansatz (Hardware-Efficient)
# ==========================================
def draw_hardware_efficient_ansatz(N, depth=2):
    qc = QuantumCircuit(N)
    for d in range(depth):
        for i in range(N):
            qc.ry(Parameter(f'$\\theta_{{{d*N + i}}}$'), i)
            
        # Entanglement layer
        if d < depth - 1:
            if d % 2 == 0:
                for i in range(0, N - 1, 2):
                    qc.cz(i, i + 1)
            else:
                for i in range(1, N - 1, 2):
                    qc.cz(i, i + 1)
                if N > 2:
                    qc.cz(0, N - 1)
    return qc

# J=1 (N=2, depth=2)
qc_g_j1 = draw_hardware_efficient_ansatz(2, 2)
fig_g_j1 = qc_g_j1.draw(output='mpl', style=style_bw)
fig_g_j1.savefig(os.path.join(plot_dir, "circuit_part_g_j1.pdf"), bbox_inches='tight', pad_inches=0.0)
print("Saved circuit_part_g_j1.pdf")

# J=2 (N=4, depth=4)
qc_g_j2 = draw_hardware_efficient_ansatz(4, 4)
fig_g_j2 = qc_g_j2.draw(output='mpl', style=style_bw)
fig_g_j2.savefig(os.path.join(plot_dir, "circuit_part_g_j2.pdf"), bbox_inches='tight', pad_inches=0.0)
print("Saved circuit_part_g_j2.pdf")

# ==========================================
# Bell State Preparation (Phi+)
# ==========================================
qc_bell = QuantumCircuit(2)
qc_bell.h(0)
qc_bell.cx(0, 1)

fig_bell = qc_bell.draw(output='mpl', style=style_bw)
fig_bell.savefig(os.path.join(plot_dir, "circuit_bell_state.pdf"), bbox_inches='tight', pad_inches=0.0)
print("Saved circuit_bell_state.pdf")

print("\nDone! Diagrams have been generated using a minimal black and white LaTeX style.")
