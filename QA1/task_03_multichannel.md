# Task 3: Multi-Channel Architecture with Entanglement

## Objective
Extend single A-gate to multi-channel system with inter-channel entanglement for detecting phase synchronization across EEG channels.

## Background
For M EEG channels:
- Use 2M qubits (2 per channel: excitatory + inhibitory)
- Add 1 ancilla qubit for global synchronization detection
- Total: 2M + 1 qubits

Entanglement captures cross-channel correlations that require O(M²) classical computation.

## Requirements

### Create file: `multichannel_circuit.py`

1. **Function: `create_multichannel_circuit(params_list)`**
   - Input: `params_list` = list of (a, b, c) tuples, one per channel
   - `num_channels = len(params_list)`
   - `n_qubits = 2 * num_channels + 1`
   - Create `QuantumCircuit(n_qubits)`
   - Qubit 0 = ancilla
   - Qubits 1 to 2M = channel qubits
   
   **Layer 1: Per-channel encoding**
   ```python
   for i in range(num_channels):
       a, b, c = params_list[i]
       q_E = 1 + 2*i      # Excitatory qubit
       q_I = 1 + 2*i + 1  # Inhibitory qubit
       
       # Apply A-gate to this channel (copy logic from Task 2)
       # Excitatory: H, P(b), RX(a), P(b), H
       # Inhibitory: H, P(b), RY(c), P(b), H
       # Coupling: CRY, CRZ
   ```
   
   **Layer 2: Inter-channel entanglement (ring topology)**
   ```python
   # Entangle excitatory qubits
   for i in range(num_channels - 1):
       q_E_i = 1 + 2*i
       q_E_j = 1 + 2*(i+1)
       qc.cnot(q_E_i, q_E_j)
   
   # Entangle inhibitory qubits
   for i in range(num_channels - 1):
       q_I_i = 1 + 2*i + 1
       q_I_j = 1 + 2*(i+1) + 1
       qc.cnot(q_I_i, q_I_j)
   ```
   
   **Layer 3: Global synchronization via ancilla**
   ```python
   qc.h(0)  # Ancilla in superposition
   for i in range(num_channels):
       q_E = 1 + 2*i
       qc.cz(q_E, 0)  # Controlled-Z from E to ancilla
   qc.h(0)  # Interference
   ```
   
   - Return: QuantumCircuit

2. **Function: `get_qubit_indices(num_channels)`**
   - Input: number of channels
   - Return: dictionary mapping labels to qubit indices
   ```python
   {
       'ancilla': 0,
       'E_qubits': [1, 3, 5, ...],
       'I_qubits': [2, 4, 6, ...]
   }
   ```

3. **Function: `add_measurements(qc, num_channels, measure_ancilla=True)`**
   - Input: QuantumCircuit, number of channels, flag for ancilla measurement
   - Add ClassicalRegister with appropriate size
   - If `measure_ancilla=True`: measure qubit 0
   - Optionally measure all E qubits
   - Return: modified QuantumCircuit

## Test Cases

```python
if __name__ == "__main__":
    from qiskit.quantum_info import Statevector
    import numpy as np
    
    # Test 1: Single channel (should match Task 2)
    params_1ch = [(0.5, 0.3, 0.7)]
    qc_1 = create_multichannel_circuit(params_1ch)
    print(f"Test 1: {qc_1.num_qubits} qubits (expected 3)")
    
    # Test 2: Multi-channel
    params_4ch = [
        (0.5, 0.3, 0.7),
        (0.6, 0.35, 0.65),
        (0.55, 0.32, 0.68),
        (0.52, 0.31, 0.69)
    ]
    qc_4 = create_multichannel_circuit(params_4ch)
    print(f"Test 2: {qc_4.num_qubits} qubits (expected 9)")
    print(f"        {qc_4.depth()} gates deep")
    
    # Test 3: Verify unitarity
    sv = Statevector(qc_4)
    print(f"Test 3: Statevector norm = {np.linalg.norm(sv.data):.6f}")
    
    # Test 4: Synchronized vs unsynchronized
    # Synchronized: all channels have similar b values
    sync_params = [(0.5, 0.4, 0.6) for _ in range(4)]
    qc_sync = create_multichannel_circuit(sync_params)
    sv_sync = Statevector(qc_sync)
    
    # Unsynchronized: random b values
    unsync_params = [(0.5, np.random.random(), 0.6) for _ in range(4)]
    qc_unsync = create_multichannel_circuit(unsync_params)
    sv_unsync = Statevector(qc_unsync)
    
    overlap = abs(sv_sync.inner(sv_unsync))**2
    print(f"Test 4: Sync vs Unsync overlap = {overlap:.3f}")
    
    # Test 5: Qubit index mapping
    indices = get_qubit_indices(4)
    print(f"Test 5: Ancilla at {indices['ancilla']}")
    print(f"        E qubits: {indices['E_qubits']}")
    print(f"        I qubits: {indices['I_qubits']}")
    
    # Test 6: With measurements
    qc_meas = add_measurements(qc_4, 4, measure_ancilla=True)
    print(f"Test 6: Circuit has {qc_meas.num_clbits} classical bits")
```

## Acceptance Criteria
- [ ] File `multichannel_circuit.py` created
- [ ] `create_multichannel_circuit()` handles variable number of channels
- [ ] For M channels: 2M+1 qubits created
- [ ] All three layers implemented (encoding, ring, global)
- [ ] Statevector norm = 1.0 (unitary)
- [ ] Synchronized channels produce different state than unsynchronized
- [ ] `get_qubit_indices()` returns correct mapping
- [ ] `add_measurements()` adds classical register correctly

## Time Estimate
12 minutes

## Dependencies
- qiskit
- numpy
- quantum_agate.py (from Task 2)

## Notes
- Ring topology CNOTs create nearest-neighbor entanglement
- Global ancilla measurement detects overall synchronization
- This architecture exploits quantum parallelism: all M² correlations encoded simultaneously

## Next Task
Task 4: Template training from pre-ictal EEG data
