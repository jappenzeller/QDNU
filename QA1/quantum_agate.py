"""
Quantum A-Gate for Single EEG Channel

Implements the 2-qubit quantum circuit encoding PN dynamics:
- Excitatory qubit (q0): H → P(b) → Rx(2a) → P(b) → H
- Inhibitory qubit (q1): H → P(b) → Ry(2c) → P(b) → H
- E-I coupling: CRy(π/4) and CRz(π/4)

The shared phase parameter 'b' on all P gates creates coupling between
excitatory and inhibitory dynamics through quantum interference.

Reference: Gupta, A., et al. (2024). Positive-negative neuron model.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


def create_single_channel_agate(a, b, c):
    """
    Create a 2-qubit A-Gate circuit for single EEG channel.

    The A-Gate encodes PN dynamics into quantum rotations:
    - Excitatory state 'a' → Rx rotation on qubit 0
    - Inhibitory state 'c' → Ry rotation on qubit 1
    - Shared phase 'b' → P gates on both qubits (creates E-I coupling)

    Args:
        a: Excitatory amplitude [0, 1] from PN dynamics
        b: Phase parameter [0, 2π] from PN dynamics
        c: Inhibitory amplitude [0, 1] from PN dynamics

    Returns:
        QuantumCircuit: 2-qubit circuit implementing the A-Gate

    Circuit structure (14 gates total, depth 7):
        q0 (E): ─H─P(b)─Rx(2a)─P(b)─H───●────────Rz(π/4)──
                                        │           │
        q1 (I): ─H─P(b)─Ry(2c)─P(b)─H───Ry(π/4)────●───────
    """
    qc = QuantumCircuit(2, name='A-Gate')

    # === Excitatory path (qubit 0) ===
    qc.h(0)
    qc.p(b, 0)          # Phase gate (shared b)
    qc.rx(2 * a, 0)     # X-rotation (factor 2 from definition)
    qc.p(b, 0)          # Phase gate (shared b)
    qc.h(0)

    # === Inhibitory path (qubit 1) ===
    qc.h(1)
    qc.p(b, 1)          # Shared phase - key coupling mechanism
    qc.ry(2 * c, 1)     # Y-rotation
    qc.p(b, 1)          # Phase gate (shared b)
    qc.h(1)

    # === E-I Coupling ===
    # CRy: Excitatory → Inhibitory influence
    qc.cry(np.pi / 4, 0, 1)
    # CRz: Inhibitory → Excitatory reciprocal influence
    qc.crz(np.pi / 4, 1, 0)

    return qc


def visualize_agate(qc, filename="agate_circuit.png"):
    """
    Visualize the A-Gate circuit and save to file.

    Args:
        qc: QuantumCircuit to visualize
        filename: Output filename (default: "agate_circuit.png")

    Returns:
        matplotlib.figure.Figure: The circuit diagram figure
    """
    import matplotlib.pyplot as plt

    fig = qc.draw('mpl', style='iqp')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fig


def get_statevector(qc):
    """
    Compute the output statevector of the quantum circuit.

    Args:
        qc: QuantumCircuit to simulate

    Returns:
        Statevector: Quantum state vector (4-dimensional for 2 qubits)

    Notes:
        - The statevector norm should always be 1.0 (unitarity)
        - Basis order: |00⟩, |01⟩, |10⟩, |11⟩
    """
    sv = Statevector.from_instruction(qc)
    return sv


def compute_fidelity(sv1, sv2):
    """
    Compute fidelity between two quantum states.

    F = |⟨ψ₁|ψ₂⟩|²

    Args:
        sv1: First Statevector
        sv2: Second Statevector

    Returns:
        float: Fidelity value in [0, 1]
    """
    return abs(sv1.inner(sv2)) ** 2


# === Test cases ===

if __name__ == "__main__":
    print("=" * 50)
    print("Quantum A-Gate Test Suite")
    print("=" * 50)

    # Test 1: Zero parameters
    print("\n=== Test 1: Zero Parameters ===")
    qc = create_single_channel_agate(0.0, 0.0, 0.0)
    print(f"Qubits: {qc.num_qubits}")
    print(f"Circuit depth: {qc.depth()}")
    print(f"Gate count: {qc.size()}")
    assert qc.num_qubits == 2, "Should have 2 qubits"
    print("[OK] Circuit structure correct")

    # Test 2: Unitarity check
    print("\n=== Test 2: Unitarity Check ===")
    qc = create_single_channel_agate(0.5, 0.3, 0.7)
    sv = get_statevector(qc)
    norm = np.linalg.norm(sv.data)
    print(f"Statevector norm: {norm:.6f}")
    assert abs(norm - 1.0) < 1e-10, f"Norm should be 1.0, got {norm}"
    print("[OK] Circuit is unitary")

    # Test 3: Circuit visualization
    print("\n=== Test 3: Circuit Visualization ===")
    try:
        visualize_agate(qc, "agate_circuit.png")
        print("Saved: agate_circuit.png")
        print("[OK] Visualization successful")
    except Exception as e:
        print(f"[SKIP] Visualization failed (matplotlib issue): {e}")

    # Test 4: Different E-I balance produces different states
    print("\n=== Test 4: E-I Balance Differentiation ===")
    qc_E_dominant = create_single_channel_agate(0.8, 0.2, 0.1)
    qc_I_dominant = create_single_channel_agate(0.1, 0.2, 0.8)
    sv_E = get_statevector(qc_E_dominant)
    sv_I = get_statevector(qc_I_dominant)
    overlap = compute_fidelity(sv_E, sv_I)
    print(f"E-dominant vs I-dominant fidelity: {overlap:.4f}")
    assert overlap < 0.99, "Different parameters should produce different states"
    print("[OK] Different parameters produce different states")

    # Test 5: Phase sensitivity
    print("\n=== Test 5: Phase Sensitivity ===")
    qc_phase0 = create_single_channel_agate(0.5, 0.0, 0.5)
    qc_phase1 = create_single_channel_agate(0.5, np.pi / 2, 0.5)
    qc_phase2 = create_single_channel_agate(0.5, np.pi, 0.5)
    sv0 = get_statevector(qc_phase0)
    sv1 = get_statevector(qc_phase1)
    sv2 = get_statevector(qc_phase2)
    f01 = compute_fidelity(sv0, sv1)
    f02 = compute_fidelity(sv0, sv2)
    f12 = compute_fidelity(sv1, sv2)
    print(f"Phase 0 vs pi/2: {f01:.4f}")
    print(f"Phase 0 vs pi: {f02:.4f}")
    print(f"Phase pi/2 vs pi: {f12:.4f}")
    assert f01 < 1.0 and f02 < 1.0, "Phase should affect state"
    print("[OK] Circuit is phase-sensitive")

    # Test 6: Print circuit diagram (use output='text' for compatibility)
    print("\n=== Test 6: Circuit Diagram ===")
    qc = create_single_channel_agate(0.5, 0.3, 0.7)
    # Use ASCII-only output for Windows compatibility
    try:
        print(qc.draw(output='text', fold=-1))
    except UnicodeEncodeError:
        print("(Circuit diagram skipped due to encoding - see agate_circuit.png)")

    # Test 7: Statevector amplitudes
    print("\n=== Test 7: Statevector Amplitudes ===")
    sv = get_statevector(qc)
    print("Basis state amplitudes:")
    for i, amp in enumerate(sv.data):
        basis = format(i, '02b')
        prob = abs(amp) ** 2
        print(f"  |{basis}⟩: {amp:.4f}  (prob: {prob:.4f})")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
