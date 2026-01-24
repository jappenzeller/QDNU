# Quantum Positive-Negative Neuron Architecture for Multi-Channel EEG Analysis: A Theoretical Framework

## Abstract

We present a quantum computing architecture based on the Positive-Negative (PN) neuron model for multi-channel electroencephalogram (EEG) analysis. The proposed quantum PN neuron encodes excitatory-inhibitory dynamics using paired qubits with parameterized rotation gates, leveraging entanglement to capture inter-channel correlations. We provide a rigorous complexity analysis demonstrating that correlation encoding scales as O(M) quantum gates compared to O(M²) classical operations for M channels. We address common overclaims in quantum machine learning literature by clarifying the distinction between Hilbert space dimensionality and extractable classical information, bounded by the Holevo limit. The architecture shows theoretical promise for seizure prediction applications, though practical advantage depends on hardware maturation beyond the current noisy intermediate-scale quantum (NISQ) era.

**Keywords:** quantum computing, neural networks, EEG analysis, positive-negative neuron, entanglement, seizure prediction

---

## 1. Introduction

Seizure prediction from electroencephalogram (EEG) recordings remains a challenging problem in computational neuroscience. Pre-ictal states exhibit complex spatiotemporal dynamics characterized by increased synchronization across channels, cross-frequency coupling, and subtle phase relationships that precede seizure onset (Mormann et al., 2007). Classical approaches require explicit computation of pairwise correlations, scaling quadratically with the number of channels.

The Positive-Negative (PN) neuron model, introduced by Gupta et al. (2024), provides a biologically-inspired framework that captures excitatory-inhibitory (E-I) dynamics fundamental to neural computation. We propose a quantum implementation of this architecture that leverages superposition and entanglement to encode E-I dynamics and inter-channel correlations more efficiently than classical methods.

This paper makes three contributions:

1. A quantum circuit architecture mapping PN neuron parameters to qubit rotations
2. Rigorous complexity analysis with corrections to common quantum advantage overclaims
3. A theoretical framework for template-based seizure prediction using quantum fidelity estimation

---

## 2. Background

### 2.1 The Positive-Negative Neuron Model

The PN neuron model (Gupta et al., 2024) describes neural dynamics through three parameters per unit:

- **a**: Amplitude parameter governing activation magnitude
- **b**: Phase parameter encoding temporal dynamics and inter-unit coupling
- **c**: Coupling strength between excitatory and inhibitory components

The E-I interaction is central to the model, reflecting the biological balance between excitation and inhibition that underlies healthy neural function and whose disruption characterizes pathological states including epilepsy.

### 2.2 Quantum Computing Preliminaries

A quantum system of n qubits exists in a 2ⁿ-dimensional complex Hilbert space. The state |ψ⟩ is described by 2ⁿ complex amplitudes, though measurement collapses this superposition to a classical outcome. This distinction between representational capacity and extractable information is crucial for honest assessment of quantum advantage.

Key quantum operations relevant to this work include:

- **Hadamard gate (H)**: Creates superposition from basis states
- **Phase gate P(θ)**: Applies phase rotation e^(iθ) to |1⟩ component
- **Rotation gates Rₓ(θ), Rᵧ(θ), Rᵤ(θ)**: Rotate qubit state around X, Y, or Z axis
- **Controlled rotations CRᵧ(θ), CRᵤ(θ)**: Conditional rotation based on control qubit state
- **CNOT, CZ**: Two-qubit entangling gates for multi-channel coupling

---

## 3. Quantum PN Neuron Architecture

### 3.1 Single-Channel Encoding (A-Gate)

For each EEG channel, we allocate two qubits representing excitatory (E) and inhibitory (I) components. The PN parameters (a, b, c) are encoded through a two-layer circuit we term the "A-Gate."

**Layer 1: Per-Qubit Encoding**

Each qubit undergoes an H-P-R-P-H sandwich structure:

**E qubit (q₀):** H → P(b) → Rₓ(2a) → P(b) → H

**I qubit (q₁):** H → P(b) → Rᵧ(2c) → P(b) → H

The shared phase parameter b appears on all four P gates (two per qubit), encoding the temporal coupling intrinsic to the PN model. The factor of 2 on rotation angles follows Qiskit convention where Rₓ(θ) rotates by θ/2. The use of Rₓ for excitatory and Rᵧ for inhibitory components creates orthogonal dynamics in the Bloch sphere representation.

**Layer 2: E-I Coupling**

After the encoding layer, bidirectional coupling gates entangle the E and I qubits:

**CRᵧ(π/4):** Control on E (q₀), target rotation on I (q₁) — excitatory influences inhibitory

**CRᵤ(π/4):** Control on I (q₁), target rotation on E (q₀) — inhibitory influences excitatory

This bidirectional structure reflects the biological reality that excitation and inhibition mutually regulate each other.

**Complete single-channel circuit:**
```
q₀ (E): ─H─P(b)─Rₓ(2a)─P(b)─H───●────────Rᵤ(π/4)──
                                │           │
q₁ (I): ─H─P(b)─Rᵧ(2c)─P(b)─H───Rᵧ(π/4)────●───────
```

**Gate count per channel:** 14 gates (4 H, 4 P, 2 R, 2 CR)

**Circuit depth:** 7

### 3.2 Multi-Channel Entanglement

For M channels, we employ 2M qubits plus one ancilla qubit. Inter-channel correlations are captured through two entanglement strategies:

**Ring topology:** Sequential CNOT gates connect adjacent E qubits and adjacent I qubits:

```
E₁ → E₂ → E₃ → ... → Eₘ
I₁ → I₂ → I₃ → ... → Iₘ
```

This requires 2(M-1) CNOT gates, encoding nearest-neighbor correlations.

**Global ancilla:** A single ancilla qubit connects to all channels via controlled-Z gates, enabling detection of global coherence patterns:

```
Ancilla —CZ— E₁, E₂, ..., Eₘ
```

This requires M additional CZ gates.

### 3.3 Total Gate Count

For M channels, the quantum circuit requires:

| Component | Gates |
|-----------|-------|
| Per-channel A-Gate | 14M (14 gates × M channels) |
| Ring topology (E qubits) | M-1 CNOT |
| Ring topology (I qubits) | M-1 CNOT |
| Global ancilla | M CZ |

**Total: 14M + 3M - 2 = 17M - 2 = O(M) gates**

---

## 4. Complexity Analysis

### 4.1 Classical Baseline

For M EEG channels with T time samples, classical PN analysis requires:

- **Feature extraction:** O(M·T) for per-channel PN dynamics
- **Pairwise correlations:** O(M²·T) for cross-channel phase locking values
- **Template matching:** O(M²) for similarity computation against stored templates

**Total classical complexity: O(M²·T)**

### 4.2 Quantum Complexity

The quantum approach requires:

- **Classical preprocessing:** O(M·T) for PN parameter extraction (unchanged)
- **Quantum encoding:** O(M) gates
- **Template matching:** O(M) gates for SWAP test circuit

**Total quantum complexity: O(M·T) + O(M)**

The quadratic term is eliminated in the correlation and matching phases.

### 4.3 Advantage Factor

| Operation | Classical | Quantum | Theoretical Advantage |
|-----------|-----------|---------|----------------------|
| Correlation encoding | O(M²) | O(M) | M× |
| Template matching | O(M²) | O(M) | M× |
| Parameter storage | O(M²) | O(M) | M× |

For 19-channel clinical EEG, this represents a theoretical 19× reduction in correlation and matching complexity.

---

## 5. Clarifications on Quantum Information Capacity

### 5.1 Hilbert Space vs. Extractable Information

A common overclaim in quantum machine learning literature conflates Hilbert space dimensionality with information capacity. While 2M qubits span a 2^(2M)-dimensional space, the Holevo bound (Holevo, 1973) limits extractable classical information to 2M bits per measurement.

**Incorrect claim:** "Quantum processes 2^(2M) dimensions simultaneously, providing exponential advantage."

**Correct statement:** The quantum state *represents* information in a 2^(2M)-dimensional space, but a single measurement yields at most 2M classical bits. The advantage lies in how interference and entanglement process correlations during computation, not in raw information throughput.

### 5.2 Measurement Statistics

The SWAP test for template matching yields fidelity |⟨ψ|φ⟩|² through measurement of an ancilla qubit. However, estimating this fidelity to precision ε requires O(1/ε²) repeated measurements (shots). This statistical overhead must be included in practical complexity analysis.

For ε = 0.01 (1% precision), approximately 10,000 shots are required per fidelity estimate. This does not negate quantum advantage for correlation encoding but tempers claims about measurement efficiency.

### 5.3 Gate Time Considerations

Comparing quantum gate counts to classical operations requires caution. Current NISQ hardware executes gates orders of magnitude slower than classical operations, with typical two-qubit gate times of 100-1000 nanoseconds versus sub-nanosecond classical operations. Quantum advantage in wall-clock time requires either:

- Fault-tolerant quantum computers with fast, reliable gates
- Problems where the classical scaling is sufficiently unfavorable

---

## 6. Application to Seizure Prediction

### 6.1 Pre-Ictal Biomarkers

Pre-ictal states exhibit characteristics naturally encoded by the quantum PN architecture:

1. **Increased synchronization:** Captured by shared phase parameter b across channels
2. **Cross-frequency coupling:** Encoded in entangled E-I dynamics
3. **Amplitude modulation:** Represented by parameter a
4. **Global coherence:** Detected via ancilla measurement

### 6.2 Template-Based Classification

The approach stores quantum states representing typical pre-ictal and inter-ictal patterns. Classification proceeds by:

1. Encoding current EEG window as quantum state |ψ_current⟩
2. Computing fidelity against pre-ictal template |ψ_preictal⟩
3. Computing fidelity against inter-ictal template |ψ_interictal⟩
4. Classifying based on fidelity comparison

The quantum fidelity naturally captures subtle phase relationships that distinguish pre-ictal from normal activity.

---

## 7. Limitations and Future Directions

### 7.1 Current Limitations

1. **NISQ constraints:** Current hardware coherence times (~100 μs) limit circuit depth
2. **Gate fidelity:** Two-qubit gate errors (~1%) accumulate across the circuit
3. **Classical preprocessing:** PN parameter extraction remains classical, limiting end-to-end speedup
4. **Simulation bottleneck:** Classical simulation for algorithm development limited to ~30 qubits

### 7.2 Mitigation Strategies

- **Error mitigation:** Zero-noise extrapolation for NISQ-era implementations
- **Shallow circuits:** Architecture designed for minimal depth
- **Hybrid workflow:** Classical preprocessing combined with quantum inference
- **Noise-aware training:** Template optimization accounting for hardware characteristics

### 7.3 Path to Practical Advantage

- **Near-term (NISQ):** Proof-of-concept with 4-8 channels on current hardware
- **Fault-tolerant era:** Full 19+ channel clinical deployment
- **Current value:** Algorithm development and theoretical foundation

---

## 8. Conclusion

The quantum PN neuron architecture provides a theoretically grounded approach to multi-channel EEG analysis with O(M) scaling for correlation encoding compared to O(M²) classical complexity. We have presented the circuit architecture, rigorous complexity analysis, and honest assessment of limitations.

Key contributions include:

1. Mapping of PN neuron dynamics to parameterized quantum circuits
2. Entanglement strategy for capturing inter-channel correlations
3. Corrections to common quantum advantage overclaims regarding information capacity
4. Framework for template-based seizure prediction

Empirical validation on clinical EEG datasets and quantum hardware represents the critical next step toward establishing practical utility.

---

## References

Gupta, A., et al. (2024). Positive-negative neuron model for excitatory-inhibitory neural dynamics. *[Journal details to be added]*.

Holevo, A. S. (1973). Bounds for the quantity of information transmitted by a quantum communication channel. *Problems of Information Transmission*, 9(3), 177-183.

Mormann, F., Andrzejak, R. G., Elger, C. E., & Lehnertz, K. (2007). Seizure prediction: The long and winding road. *Brain*, 130(2), 314-333.

Nielsen, M. A., & Chuang, I. L. (2010). *Quantum computation and quantum information* (10th anniversary ed.). Cambridge University Press.

Preskill, J. (2018). Quantum computing in the NISQ era and beyond. *Quantum*, 2, 79.

Schuld, M., & Petruccione, F. (2021). *Machine learning with quantum computers* (2nd ed.). Springer.

---

## Appendix A: Gate Definitions

### A.1 Single-Qubit Gates

| Gate | Matrix | Action |
|------|--------|--------|
| H | (1/√2)[[1,1],[1,-1]] | Creates equal superposition |
| P(θ) | [[1,0],[0,e^(iθ)]] | Phase rotation on \|1⟩ |
| Rₓ(θ) | [[cos(θ/2),-i·sin(θ/2)],[-i·sin(θ/2),cos(θ/2)]] | X-axis rotation |
| Rᵧ(θ) | [[cos(θ/2),-sin(θ/2)],[sin(θ/2),cos(θ/2)]] | Y-axis rotation |
| Rᵤ(θ) | [[e^(-iθ/2),0],[0,e^(iθ/2)]] | Z-axis rotation |

### A.2 Two-Qubit Gates

| Gate | Action |
|------|--------|
| CRᵧ(θ) | Applies Rᵧ(θ) to target when control is \|1⟩ |
| CRᵤ(θ) | Applies Rᵤ(θ) to target when control is \|1⟩ |
| CNOT | Flips target when control is \|1⟩ (ring topology) |
| CZ | Applies Z to target when control is \|1⟩ (global ancilla) |

---

## Appendix B: Concrete Example (M=4 Channels)

For a 4-channel subset (e.g., Fp1, Fp2, F3, F4):

- **Qubits:** 2×4 + 1 = 9 qubits (8 for E-I pairs + 1 ancilla)
- **A-Gate encoding:** 14 × 4 = 56 gates
- **Ring topology:** 3 + 3 = 6 CNOT gates
- **Global ancilla:** 4 CZ gates
- **Total:** 66 gates
- **Circuit depth:** ~15 (encoding parallel across channels + coupling layers)

**Classical equivalent:**

- Pairwise correlations: 4² = 16 computations
- Phase locking values: 6 pairs × T samples

The quantum circuit encodes all correlations in a single state preparation, with advantage growing as M increases.

## Appendix C: Qiskit Implementation

```python
from qiskit import QuantumCircuit
import numpy as np

def create_agate_circuit(a, b, c):
    """
    Create single-channel A-Gate circuit for PN neuron.

    Parameters:
        a: Excitatory state [0, 1]
        b: Shared phase parameter
        c: Inhibitory state [0, 1]

    Returns:
        QuantumCircuit with 2 qubits
    """
    qc = QuantumCircuit(2)

    # === Layer 1: Per-Qubit Encoding ===

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

    # === Layer 2: E-I Coupling ===
    qc.cry(np.pi/4, 0, 1)  # E controls I
    qc.crz(np.pi/4, 1, 0)  # I controls E

    return qc
```
