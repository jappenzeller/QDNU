"""
A-Gate Quantum State Visualization.

Extracts Bloch sphere coordinates, entanglement, and purity metrics
from 2-qubit A-Gate circuits for visualization.
"""

import numpy as np
from typing import Dict, Tuple
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace, concurrence


def get_bloch_coords(rho_single: DensityMatrix) -> Tuple[float, float, float]:
    """
    Extract Bloch vector (x, y, z) from single-qubit density matrix.

    The Bloch vector components are:
        x = 2 * Re(rho_01)
        y = 2 * Im(rho_10)
        z = rho_00 - rho_11

    Returns:
        (x, y, z): Bloch vector coordinates
    """
    data = rho_single.data
    x = 2 * np.real(data[0, 1])
    y = 2 * np.imag(data[1, 0])
    z = np.real(data[0, 0] - data[1, 1])
    return (float(x), float(y), float(z))


def get_purity(rho: DensityMatrix) -> float:
    """Compute purity Tr(rho^2). Pure state = 1, maximally mixed = 0.5."""
    return float(np.real(np.trace(rho.data @ rho.data)))


def extract_visualization_data(circuit: QuantumCircuit) -> Dict:
    """
    Extract all visualization data from a quantum circuit.

    Returns:
        dict: {
            'statevector': complex array [4],
            'probabilities': float array [4],
            'bloch_E': (x, y, z) for excitatory qubit,
            'bloch_I': (x, y, z) for inhibitory qubit,
            'purity_E': float,
            'purity_I': float,
            'concurrence': float (entanglement measure),
            'global_phase': float
        }
    """
    # Get statevector
    sv = Statevector.from_instruction(circuit)
    amplitudes = sv.data

    # Full density matrix
    rho = DensityMatrix(sv)

    # Reduced density matrices (partial trace)
    rho_E = partial_trace(rho, [1])  # Trace out qubit 1 -> E (q0)
    rho_I = partial_trace(rho, [0])  # Trace out qubit 0 -> I (q1)

    # Extract Bloch coordinates
    bloch_E = get_bloch_coords(rho_E)
    bloch_I = get_bloch_coords(rho_I)

    # Compute entanglement
    ent = float(concurrence(sv))

    # Global phase from first non-zero amplitude
    nonzero_idx = np.argmax(np.abs(amplitudes) > 1e-10)
    global_phase = float(np.angle(amplitudes[nonzero_idx]))

    return {
        'statevector': [complex(a) for a in amplitudes],
        'statevector_list': [[float(a.real), float(a.imag)] for a in amplitudes],
        'probabilities': sv.probabilities().tolist(),
        'bloch_E': bloch_E,
        'bloch_I': bloch_I,
        'purity_E': get_purity(rho_E),
        'purity_I': get_purity(rho_I),
        'concurrence': ent,
        'global_phase': global_phase
    }


def bloch_to_spherical(bloch: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Convert Bloch (x, y, z) to spherical (r, theta, phi).

    Returns:
        (r, theta, phi): r = purity indicator, theta = polar angle, phi = azimuthal
    """
    x, y, z = bloch
    r = np.sqrt(x**2 + y**2 + z**2)

    if r < 1e-10:
        return (0.0, 0.0, 0.0)

    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    if phi < 0:
        phi += 2 * np.pi

    return (float(r), float(theta), float(phi))


def spherical_to_bloch(r: float, theta: float, phi: float) -> Tuple[float, float, float]:
    """Convert spherical (r, theta, phi) to Bloch (x, y, z)."""
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return (float(x), float(y), float(z))


def bloch_to_unity_coords(bloch: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Convert standard Bloch coordinates to game engine coordinates (Y=up).

    Standard Bloch: +Z = |0>, Unity/Game: +Y = |0>
    """
    x, y, z = bloch
    return (float(x), float(z), float(y))


def compute_von_neumann_entropy(rho: DensityMatrix) -> float:
    """Compute von Neumann entropy S = -Tr(rho * log(rho))."""
    eigenvalues = np.linalg.eigvalsh(rho.data)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def compute_mutual_information(circuit: QuantumCircuit) -> float:
    """Compute quantum mutual information I(E:I) = S(E) + S(I) - S(EI)."""
    sv = Statevector.from_instruction(circuit)
    rho = DensityMatrix(sv)
    rho_E = partial_trace(rho, [1])
    rho_I = partial_trace(rho, [0])

    S_E = compute_von_neumann_entropy(rho_E)
    S_I = compute_von_neumann_entropy(rho_I)
    S_EI = compute_von_neumann_entropy(rho)

    return S_E + S_I - S_EI


# === Matplotlib Drawing Functions ===

def draw_bloch_sphere(ax, bloch_coords, color='blue', label='', show_axes=True, use_unity_coords=False):
    """
    Draw a Bloch sphere with state vector on a 3D axis.

    Convention: Z-axis (green) is UP, +Z = |0⟩, -Z = |1⟩
    """
    ax.clear()

    # Use standard Bloch coordinates: +Z = |0⟩ (up)
    bx, by, bz = bloch_coords

    # Sphere wireframe
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.2, linewidth=0.5)

    if show_axes:
        # Z-axis GREEN (up = |0⟩)
        ax.quiver(0, 0, 0, 0, 0, 1.2, color='#00FF00', alpha=0.9, arrow_length_ratio=0.08, linewidth=2)
        # X-axis RED
        ax.quiver(0, 0, 0, 1.2, 0, 0, color='red', alpha=0.7, arrow_length_ratio=0.08, linewidth=1.5)
        # Y-axis BLUE
        ax.quiver(0, 0, 0, 0, 1.2, 0, color='blue', alpha=0.7, arrow_length_ratio=0.08, linewidth=1.5)

        # Labels - Z is up/down with |0⟩ and |1⟩
        ax.text(0, 0, 1.5, '|0⟩', color='#00FF00', fontsize=11, fontweight='bold', ha='center')
        ax.text(0, 0, -1.5, '|1⟩', color='#00FF00', fontsize=11, fontweight='bold', ha='center')
        ax.text(1.4, 0, 0, '|+⟩', color='red', fontsize=9)
        ax.text(-1.4, 0, 0, '|−⟩', color='red', fontsize=9)
        ax.text(0, 1.4, 0, '|+i⟩', color='blue', fontsize=9)
        ax.text(0, -1.4, 0, '|−i⟩', color='blue', fontsize=9)

    # State vector
    r = np.sqrt(bx**2 + by**2 + bz**2)

    if r > 0.01:
        ax.quiver(0, 0, 0, bx, by, bz, color=color, arrow_length_ratio=0.12, linewidth=2.5)
        ax.scatter([bx], [by], [bz], color=color, s=80, edgecolors='white', linewidths=1)

    ax.set_xlim([-1.5, 1.5])
    ax.set_ylim([-1.5, 1.5])
    ax.set_zlim([-1.5, 1.5])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(label)
    ax.set_box_aspect([1, 1, 1])

    # Set view angle so Z points up
    ax.view_init(elev=20, azim=45)


def draw_entanglement_indicator(ax, concurrence, purity_E, purity_I):
    """Draw entanglement and purity indicators as horizontal bar chart."""
    ax.clear()

    categories = ['Concurrence', 'Purity E', 'Purity I']
    values = [concurrence, purity_E, purity_I]
    colors = ['magenta', '#FF6B35', '#7B68EE']

    bars = ax.barh(categories, values, color=colors, edgecolor='white', linewidth=1)
    ax.set_xlim([0, 1])
    ax.set_xlabel('Value')
    ax.set_title('Quantum Properties')

    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)


def draw_probabilities(ax, probs):
    """Draw probability distribution bar chart."""
    ax.clear()

    states = ['|00>', '|01>', '|10>', '|11>']
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']

    bars = ax.bar(states, probs, color=colors, edgecolor='white', linewidth=1)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Probability')
    ax.set_title('Measurement Probabilities')

    for bar, prob in zip(bars, probs):
        if prob > 0.01:
            ax.text(bar.get_x() + bar.get_width()/2, prob + 0.02,
                    f'{prob:.3f}', ha='center', fontsize=9)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(__file__).replace('visualization/agate_visualization.py', ''))
    from quantum_agate import create_single_channel_agate

    # Test extraction
    circuit = create_single_channel_agate(a=0.7, b=np.pi/2, c=0.4)
    viz = extract_visualization_data(circuit)

    print("=== Visualization Data ===")
    print(f"Bloch E: {viz['bloch_E']}")
    print(f"Bloch I: {viz['bloch_I']}")
    print(f"Purity E: {viz['purity_E']:.4f}")
    print(f"Purity I: {viz['purity_I']:.4f}")
    print(f"Concurrence: {viz['concurrence']:.4f}")
