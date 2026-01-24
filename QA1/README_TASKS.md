# Quantum Seizure Prediction: Task Execution Guide

## Overview
This project implements a quantum machine learning system for EEG-based seizure prediction using Positive-Negative Dynamic Neural Units (PNDNU) mapped to quantum gates.

## Key Innovation
**Quantum advantage**: Multi-channel EEG correlations encoded in entangled quantum state using 2M qubits. Fidelity measurement captures O(M²) classical correlations with O(M) gate complexity (note: O(1/ε²) shots required for precision ε).

## Task Execution Order

Execute tasks sequentially. Each builds on previous components.

### Task 1: PN Dynamics Evolution ⏱️ 10 min
**File**: `task_01_pn_dynamics.md`

**What it does**: Implements differential equations that evolve EEG signals into quantum gate parameters (a, b, c).

**Dependencies**: numpy

**Output**: `pn_dynamics.py` with `PNDynamics` class

**Key validation**: Parameters stay bounded [0, 1] for normalized input

---

### Task 2: Single-Channel A-Gate ⏱️ 10 min
**File**: `task_02_single_agate.md`

**What it does**: Creates quantum circuit for single EEG channel with:
- Excitatory dynamics: H → P(b) → Rx(2a) → P(b) → H
- Inhibitory dynamics: H → P(b) → Ry(2c) → P(b) → H
- E-I coupling: CRy(π/4) and CRz(π/4)

**Dependencies**: qiskit, numpy, matplotlib

**Output**: `quantum_agate.py` with circuit creation functions

**Key validation**: Statevector norm = 1.0 (unitarity preserved)

---

### Task 3: Multi-Channel Architecture ⏱️ 12 min
**File**: `task_03_multichannel.md`

**What it does**: Extends to M channels with three entanglement layers:
1. Per-channel encoding (2M qubits)
2. Nearest-neighbor ring coupling
3. Global synchronization via ancilla

**Dependencies**: qiskit, quantum_agate.py

**Output**: `multichannel_circuit.py`

**Key validation**: Synchronized channels produce different state than unsynchronized

---

### Task 4: Template Training ⏱️ 10 min
**File**: `task_04_template_training.md`

**What it does**: Trains quantum template from pre-ictal EEG data:
- EEG normalization
- PN dynamics integration
- Template circuit creation
- Save/load functionality

**Dependencies**: pn_dynamics.py, multichannel_circuit.py

**Output**: `template_trainer.py` with `TemplateTrainer` class

**Key validation**: Pre-ictal data shows lower phase std than normal (synchronization)

---

### Task 5: Prediction Pipeline ⏱️ 12 min
**File**: `task_05_prediction.md`

**What it does**: Implements seizure prediction via quantum fidelity:
- F = |⟨ψ_template|ψ_test⟩|²
- Threshold-based classification
- Batch evaluation
- Performance metrics

**Dependencies**: template_trainer.py, sklearn

**Output**: `seizure_predictor.py` with `SeizurePredictor` class

**Key validation**: Pre-ictal achieves higher fidelity than normal EEG

---

### Task 6: Integration Testing ⏱️ 15 min
**File**: `task_06_integration.md`

**What it does**: End-to-end pipeline validation:
- Training phase
- Prediction phase
- Batch evaluation
- Parameter sensitivity
- Synthetic data generation

**Dependencies**: All previous modules

**Output**: `main_pipeline.py` with complete workflow

**Key validation**: >70% accuracy on synthetic pre-ictal vs normal classification

---

## Total Estimated Time
**69 minutes** (~1.2 hours)

## Project Structure After Completion

```
project/
├── pn_dynamics.py              # Task 1
├── quantum_agate.py            # Task 2
├── multichannel_circuit.py     # Task 3
├── template_trainer.py         # Task 4
├── seizure_predictor.py        # Task 5
├── main_pipeline.py            # Task 6
├── trained_template.npy        # Generated during training
└── task_*.md                   # Task specifications
```

## Quick Start After Completion

```python
from template_trainer import TemplateTrainer
from seizure_predictor import SeizurePredictor
import numpy as np

# 1. Train on pre-ictal EEG
trainer = TemplateTrainer(num_channels=4, lambda_a=0.1, lambda_c=0.05)
preictal_eeg = np.load('preictal_data.npy')  # Shape: (4, time_steps)
trainer.train(preictal_eeg)
trainer.save_template('my_template.npy')

# 2. Predict on new EEG
predictor = SeizurePredictor(trainer, threshold=0.7)
test_eeg = np.load('test_data.npy')
prediction, fidelity, metrics = predictor.predict(test_eeg)

print(f"Seizure predicted: {prediction}")
print(f"Confidence (fidelity): {fidelity:.3f}")
```

## Key Concepts

### PN Dynamics Parameters
- **a (excitatory)**: Decays naturally (−λ_a), driven by signal
- **b (phase)**: Pure integration, creates E-I coupling
- **c (inhibitory)**: Grows naturally (+λ_c), driven by signal

### Quantum Encoding (A-Gate)
Each channel uses 2 qubits with H-P-R-P-H sandwich structure:
- **E qubit**: H → P(b) → Rx(2a) → P(b) → H
- **I qubit**: H → P(b) → Ry(2c) → P(b) → H
- **E-I coupling**: CRy(π/4) from E→I, CRz(π/4) from I→E
- Shared **b** on all 4 P gates creates phase-sensitive interference

### Why Quantum Advantage
1. **Entanglement**: Encodes M² channel correlations in 2M qubits
2. **Interference**: Shared phase b detects synchronization automatically
3. **Hilbert space**: 2^(2M) dimensional state enables rich correlation encoding
4. **Gate complexity**: O(M) gates vs O(M²) classical correlation computations

## Troubleshooting

### Common Issues

**Issue**: Parameters overflow (>1.0)
- **Fix**: Increase normalization in `TemplateTrainer.normalize_eeg()`

**Issue**: Low prediction accuracy
- **Fix**: Tune lambda_a, lambda_c, threshold parameters

**Issue**: Fidelity always near 0.5
- **Fix**: Check EEG normalization, verify template training used correct data

**Issue**: Circuit too deep
- **Fix**: Reduce num_channels or simplify entanglement topology

## Next Development Steps

1. **Real EEG Integration**: Add .edf file loading
2. **Hardware Deployment**: Test on IBM Quantum backends
3. **Classical Baseline**: Implement classical PN network for comparison
4. **Hyperparameter Optimization**: Systematic search for optimal λ_a, λ_c
5. **Multi-Patient Validation**: Train/test across different patients

## References

- Gupta, A., et al. (2024). Positive-negative neuron model for excitatory-inhibitory neural dynamics.
- Holevo, A. S. (1973). Bounds for the quantity of information transmitted by a quantum communication channel. *Problems of Information Transmission*, 9(3), 177-183.
- Mormann, F., et al. (2007). Seizure prediction: The long and winding road. *Brain*, 130(2), 314-333.

## Contact & Questions

For questions about:
- **PN dynamics**: See Task 1 documentation
- **Quantum circuits**: See Tasks 2-3 documentation  
- **Training/prediction**: See Tasks 4-5 documentation
- **Integration**: See Task 6 documentation

---

**Status**: Ready for sequential execution
**Author**: System design based on quantum PN neuron architecture
**License**: Research/Academic Use
