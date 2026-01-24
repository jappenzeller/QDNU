# Task 2: Single-Channel A-Gate Quantum Circuit

## Objective
Implement the quantum A-gate for a single EEG channel with separate excitatory (R_X) and inhibitory (R_Y) dynamics.

## Background
The A-gate encodes PN dynamics into quantum rotations:
- Excitatory: θ = a + ib → R_X rotation
- Inhibitory: φ = c + ib → R_Y rotation
- Shared phase `b` creates coupling between E and I dynamics

## Requirements

### Create file: `quantum_agate.py`

1. **Function: `create_single_channel_agate(a, b, c)`**
   - Input: floats a, b, c (from PN dynamics)
   - Create `QuantumCircuit(2)` (2 qubits: excitatory and inhibitory)
   - Qubit 0 = Excitatory, Qubit 1 = Inhibitory
   
   - Excitatory path (qubit 0):
     ```python
     qc.h(0)
     qc.p(b, 0)         # Phase gate
     qc.rx(2*a, 0)      # X-rotation (factor 2 from definition)
     qc.p(b, 0)         # Phase gate (shared b)
     qc.h(0)
     ```
   
   - Inhibitory path (qubit 1):
     ```python
     qc.h(1)
     qc.p(b, 1)         # Shared phase - key coupling
     qc.ry(2*c, 1)      # Y-rotation
     qc.p(b, 1)         # Phase gate
     qc.h(1)
     ```
   
   - E-I coupling:
     ```python
     qc.cry(np.pi/4, 0, 1)  # Controlled-RY
     qc.crz(np.pi/4, 1, 0)  # Reciprocal coupling
     ```
   
   - Return: QuantumCircuit

2. **Function: `visualize_agate(qc, filename="agate_circuit.png")`**
   - Input: QuantumCircuit
   - Draw circuit using matplotlib: `qc.draw('mpl')`
   - Save to file
   - Return: matplotlib figure

3. **Function: `get_statevector(qc)`**
   - Input: QuantumCircuit
   - Use Qiskit `Statevector` to get quantum state
   - Return: Statevector object

4. **Add comprehensive docstrings**

## Test Cases

```python
if __name__ == "__main__":
    from qiskit.quantum_info import Statevector
    import numpy as np
    
    # Test 1: Zero parameters (identity-like)
    qc = create_single_channel_agate(0.0, 0.0, 0.0)
    print(f"Test 1: Circuit with {qc.num_qubits} qubits, {qc.depth()} gates")
    
    # Test 2: Non-zero parameters
    qc = create_single_channel_agate(0.5, 0.3, 0.7)
    sv = get_statevector(qc)
    print(f"Test 2: Statevector norm = {np.linalg.norm(sv.data):.3f}")
    # Should be 1.0 (unitary)
    
    # Test 3: Visualize
    visualize_agate(qc)
    print("Test 3: Circuit visualization saved")
    
    # Test 4: Different E-I balance
    qc_E_dominant = create_single_channel_agate(0.8, 0.2, 0.1)
    qc_I_dominant = create_single_channel_agate(0.1, 0.2, 0.8)
    sv_E = get_statevector(qc_E_dominant)
    sv_I = get_statevector(qc_I_dominant)
    overlap = abs(sv_E.inner(sv_I))**2
    print(f"Test 4: E-dominant vs I-dominant overlap = {overlap:.3f}")
    # Should be < 1.0 (different states)
```

## Acceptance Criteria
- [ ] File `quantum_agate.py` created
- [ ] `create_single_channel_agate()` returns valid QuantumCircuit
- [ ] Circuit has exactly 2 qubits
- [ ] Circuit includes H, P, RX, RY, CRY, CRZ gates
- [ ] Statevector has norm = 1.0 (circuit is unitary)
- [ ] Circuit visualization saves successfully
- [ ] Different parameters produce different quantum states

## Time Estimate
10 minutes

## Dependencies
- qiskit
- numpy
- matplotlib

## Notes
- The shared phase `b` on both qubits is crucial - it couples E and I dynamics
- R_X vs R_Y gives better parameter space coverage than both R_Y
- The CRY and CRZ gates encode the w_EI and w_IE coupling from classical PN model

## Next Task
Task 3: Multi-channel architecture with entanglement
