# Single-Channel A-Gate: Correct 2-Qubit Circuit

## Circuit Architecture

```
q0 (Excitatory): ─|H|─|P(b)|─|RX(a)|─|P(b)|─|H|───●─────────|
                                                   │         
q1 (Inhibitory): ─|H|─|P(b)|─|RY(c)|─|P(b)|─|H|─|RY|───●────|
                                                        │
                                                      |RZ|
```

## Gate Sequence Breakdown

### Layer 1: Per-Qubit Encoding

**Qubit 0 (Excitatory path):**
1. H - Hadamard
2. P(b) - Phase gate with angle b
3. RX(a) - X-rotation with angle 2a
4. P(b) - Phase gate with angle b (shared!)
5. H - Hadamard

**Qubit 1 (Inhibitory path):**
1. H - Hadamard
2. P(b) - Phase gate with angle b (shared!)
3. RY(c) - Y-rotation with angle 2c
4. P(b) - Phase gate with angle b (shared!)
5. H - Hadamard

### Layer 2: E-I Coupling

**Gate 6: Controlled-RY**
- Control: Qubit 0 (excitatory)
- Target: Qubit 1 (inhibitory)
- Angle: π/4

**Gate 7: Controlled-RZ**
- Control: Qubit 1 (inhibitory)
- Target: Qubit 0 (excitatory)
- Angle: π/4

## Qiskit Implementation

```python
from qiskit import QuantumCircuit
import numpy as np

def create_agate_circuit(a, b, c):
    qc = QuantumCircuit(2)
    
    # === Excitatory qubit (q0) ===
    qc.h(0)
    qc.p(b, 0)
    qc.rx(2*a, 0)  # Factor of 2 from Qiskit convention
    qc.p(b, 0)     # Shared phase
    qc.h(0)
    
    # === Inhibitory qubit (q1) ===
    qc.h(1)
    qc.p(b, 1)     # Shared phase
    qc.ry(2*c, 1)
    qc.p(b, 1)     # Shared phase
    qc.h(1)
    
    # === E-I Coupling ===
    qc.cry(np.pi/4, 0, 1)  # Control=0, Target=1
    qc.crz(np.pi/4, 1, 0)  # Control=1, Target=0
    
    return qc

# Example usage
qc = create_agate_circuit(a=0.5, b=0.3, c=0.7)
qc.draw('mpl')
```

## Visual Layout for Diagram Tool

### Horizontal Layout (time flows left to right):

```
Time:    t0   t1      t2        t3      t4   t5       t6
         |    |       |         |       |    |        |
q0 (E):  H────P(b)────RX(a)─────P(b)────H────●────────┼────
                                             │        │
q1 (I):  H────P(b)────RY(c)─────P(b)────H────RY(π/4)──●────
                                                       │
                                                      RZ(π/4)

Legend:
  H     = Hadamard gate
  P(θ)  = Phase gate (diagonal matrix with e^(iθ))
  RX(θ) = Rotation around X-axis
  RY(θ) = Rotation around Y-axis
  RZ(θ) = Rotation around Z-axis
  ●     = Control qubit
  ⊕     = Target qubit (for controlled gates)
```

## Key Features to Highlight in Diagram

1. **Shared phase parameter b**: Appears 4 times total
   - P(b) on q0 before RX
   - P(b) on q0 after RX
   - P(b) on q1 before RY
   - P(b) on q1 after RY

2. **Different rotation axes**:
   - Excitatory uses RX (rotation around X)
   - Inhibitory uses RY (rotation around Y)
   - This creates orthogonal dynamics

3. **Bidirectional coupling**:
   - E → I coupling via controlled-RY
   - I → E coupling via controlled-RZ

4. **Sandwich structure**:
   - H...H creates basis transformation
   - P(b)...P(b) creates phase accumulation
   - RX/RY in middle encodes amplitude

## Color Coding Suggestion

- **Blue**: Hadamard gates (H)
- **Purple**: Phase gates P(b) - emphasize these are SHARED
- **Red**: RX(a) - excitatory rotation
- **Green**: RY(c) - inhibitory rotation
- **Orange**: CRY and CRZ - coupling gates

## Common Errors to Avoid

❌ **Wrong**: Different b values on each qubit
✓ **Correct**: Same b on all 4 phase gates

❌ **Wrong**: Both qubits using RY
✓ **Correct**: q0 uses RX, q1 uses RY

❌ **Wrong**: Coupling gates before H-P-R-P-H sequence
✓ **Correct**: Coupling gates AFTER the sequence

❌ **Wrong**: Single P(b) on each qubit
✓ **Correct**: Two P(b) gates per qubit (sandwich pattern)

## Gate Count Summary

Total gates: 14
- Hadamard: 4 (2 per qubit)
- Phase: 4 (2 per qubit, all with same angle b)
- Rotation: 2 (1 RX, 1 RY)
- Controlled: 2 (1 CRY, 1 CRZ)

Circuit depth: 7 (sequential time steps)

## Parameters

- **a**: Excitatory state [0, 1] → RX angle = 2a
- **b**: Shared phase [0, ∞) → P angle = b
- **c**: Inhibitory state [0, 1] → RY angle = 2c

All derived from PN dynamics evolution:
- da/dt = -λ_a·a + f(t)·(1-a)
- db/dt = f(t)·(1-b)
- dc/dt = λ_c·c + f(t)·(1-c)
