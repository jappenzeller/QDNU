# Quantum Positive-Negative Neuron (QDNU)

A quantum computing architecture for multi-channel EEG analysis based on the Positive-Negative (PN) neuron model.

## Overview

This project implements a quantum circuit architecture that encodes excitatory-inhibitory (E-I) neural dynamics using paired qubits with parameterized rotation gates. The approach leverages entanglement to capture inter-channel correlations with O(M) complexity compared to O(M²) for classical methods.

## Architecture

The core component is the **A-Gate**, a 2-qubit circuit encoding a single PN neuron channel:

```
q₀ (E): ─H─P(b)─Rₓ(2a)─P(b)─H───●────────Rᵤ(π/4)──
                                │           │
q₁ (I): ─H─P(b)─Rᵧ(2c)─P(b)─H───Rᵧ(π/4)────●───────
```

**Parameters:**
- `a`: Excitatory state amplitude
- `b`: Shared phase (E-I coupling)
- `c`: Inhibitory state amplitude

**Key features:**
- 14 gates per channel (4 H, 4 P, 2 R, 2 CR)
- Bidirectional E-I coupling via CRᵧ and CRᵤ
- Shared phase parameter encodes temporal dynamics

## Project Structure

```
QDNU/
├── QA1/
│   ├── QUANTUM_ADVANTAGE.md      # Initial theoretical analysis
│   └── quantum_pn_neuron_paper.md # Publishable paper draft
├── Diagrams/
│   ├── pn_2qubit_circuit.svg     # A-Gate circuit diagram
│   ├── CORRECT_CIRCUIT_DIAGRAM.md # Circuit specification
│   └── *.png                      # Generated circuit visualizations
└── Legacy/                        # Archived exploration code
```

## Theoretical Basis

The quantum PN neuron provides:

| Operation | Classical | Quantum | Advantage |
|-----------|-----------|---------|-----------|
| Correlation encoding | O(M²) | O(M) | M× |
| Template matching | O(M²) | O(M) | M× |
| Parameter storage | O(M²) | O(M) | M× |

For 19-channel clinical EEG, this represents a theoretical 19× reduction in correlation and matching complexity.

## Quick Start

```python
from qiskit import QuantumCircuit
import numpy as np

def create_agate_circuit(a, b, c):
    qc = QuantumCircuit(2)

    # Excitatory qubit (q0)
    qc.h(0)
    qc.p(b, 0)
    qc.rx(2*a, 0)
    qc.p(b, 0)
    qc.h(0)

    # Inhibitory qubit (q1)
    qc.h(1)
    qc.p(b, 1)
    qc.ry(2*c, 1)
    qc.p(b, 1)
    qc.h(1)

    # E-I Coupling
    qc.cry(np.pi/4, 0, 1)
    qc.crz(np.pi/4, 1, 0)

    return qc

# Example usage
qc = create_agate_circuit(a=0.5, b=0.3, c=0.7)
print(qc.draw())
```

## References

- Gupta, A., et al. (2024). Positive-negative neuron model for excitatory-inhibitory neural dynamics.
- Holevo, A. S. (1973). Bounds for the quantity of information transmitted by a quantum communication channel.

## License

Research use only. Contact authors for collaboration.
