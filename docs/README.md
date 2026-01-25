# Quantum Positive-Negative Neuron (QDNU)

A quantum computing architecture for multi-channel EEG seizure prediction based on the Positive-Negative (PN) neuron model.

**Author:** James Appenzeller, Independent Researcher

[![Paper](https://img.shields.io/badge/Paper-quantum__pn__neuron__paper.md-blue)](QA1/quantum_pn_neuron_paper.md)

---

## Abstract

This project presents a quantum computing architecture based on the Positive-Negative (PN) neuron model for multi-channel electroencephalogram (EEG) analysis. The proposed quantum PN neuron encodes excitatory-inhibitory dynamics using paired qubits with parameterized rotation gates, leveraging quantum entanglement to capture inter-channel correlations efficiently.

A rigorous complexity analysis demonstrates that correlation encoding scales as **O(M) quantum gates** compared to **O(M²) classical operations** for M channels. The architecture shows theoretical promise for seizure prediction applications, with empirical validation demonstrating clear separation between ictal and interictal EEG patterns.

---

## Results

Validation on the Kaggle American Epilepsy Society Seizure Prediction Challenge dataset (Dog_1 subject):

| Metric | Value |
|--------|-------|
| **Accuracy** | **72.4%** |
| Ictal Fidelity | 0.894 ± 0.090 |
| Interictal Fidelity | 0.722 ± 0.193 |
| Fidelity Separation | 0.173 |
| Precision | 66.7% |
| Recall | 85.7% |
| Specificity | 60.0% |
| F1 Score | 75.0% |

*Results obtained using 4-channel quantum circuit (9 qubits, 60 gates), symmetric PN dynamics, and optimized threshold (0.82).*

### Dashboard

![QDNU Dashboard](QA1/figures/qdnu_dashboard.png)

### Quantum Fidelity Distribution

![Fidelity Distribution](QA1/figures/fidelity_distribution.png)

---

## Architecture

### The A-Gate

The core component is the **A-Gate**, a 2-qubit circuit encoding a single PN neuron channel:

![A-Gate Circuit](QA1/figures/agate_circuit.png)

**Parameters:**
- `a`: Excitatory state amplitude [0, 1]
- `b`: Shared phase (E-I coupling) [0, 2π]
- `c`: Inhibitory state amplitude [0, 1]

**Circuit properties:**
- 14 gates per channel (4 H, 4 P, 2 R, 2 CR)
- Bidirectional E-I coupling via CRy and CRz
- Shared phase parameter encodes temporal dynamics

### Multi-Channel Architecture

For M EEG channels, the quantum circuit uses:
- **Qubits:** 2M + 1 (channel qubits + ancilla)
- **Gates:** 17M - 2
- **Depth:** O(M)

![Multi-Channel Circuit](QA1/figures/multichannel_circuit.png)

### Complexity Advantage

| Operation | Classical | Quantum | Advantage Factor |
|-----------|-----------|---------|------------------|
| Correlation encoding | O(M²) | O(M) | M× |
| Template matching | O(M²) | O(M) | M× |
| Parameter storage | O(M²) | O(M) | M× |

For 19-channel clinical EEG: **19× reduction** in correlation complexity.

---

## PN Dynamics

The Positive-Negative neuron model captures excitatory-inhibitory dynamics:

$$\frac{da}{dt} = -\lambda_a \cdot a + f(t)(1 - a)$$

$$\frac{dc}{dt} = +\lambda_c \cdot c + f(t)(1 - c)$$

Where:
- `f(t)` = normalized EEG input (RMS envelope)
- `λ_a` = excitatory decay rate (default: 0.1)
- `λ_c` = inhibitory growth rate (default: 0.05)

![PN Dynamics](QA1/figures/pn_dynamics.png)

---

## Installation

```bash
git clone https://github.com/jappenzeller/QDNU.git
cd QDNU
pip install -r requirements.txt
```

**Requirements:**
- Python 3.9+
- Qiskit 1.0+
- NumPy, SciPy, Matplotlib
- (Optional) Kaggle EEG dataset for validation

## Quick Start

```python
from QA1.quantum_agate import create_single_channel_agate
from QA1.visualization import extract_visualization_data

# Create A-Gate circuit
circuit = create_single_channel_agate(a=0.6, b=1.2, c=0.4)

# Extract quantum state visualization
viz = extract_visualization_data(circuit)
print(f"Concurrence: {viz['concurrence']:.4f}")
print(f"Bloch E: {viz['bloch_E']}")
print(f"Bloch I: {viz['bloch_I']}")
```

### Run Dashboard

```bash
cd QA1
python dashboard.py
```

### Run Harmonic Oscillator Visualization

```python
from QA1.visualization import run_oscillator_demo

config, history, viz, figures = run_oscillator_demo(
    duration=20.0,
    drive_frequency=0.2,
    save_dir='figures/harmonic_oscillator'
)
```

---

## Project Structure

```
QDNU/
├── QA1/
│   ├── quantum_pn_neuron_paper.md   # Full paper
│   ├── quantum_agate.py             # A-Gate circuit implementation
│   ├── pn_dynamics.py               # PN neuron dynamics
│   ├── template_trainer.py          # Template-based classifier
│   ├── seizure_predictor.py         # Fidelity-based prediction
│   ├── dashboard.py                 # Results visualization
│   ├── eeg_loader.py                # Kaggle dataset loader
│   ├── visualization/               # Visualization module
│   │   ├── agate_visualization.py   # Bloch spheres, entanglement
│   │   ├── fractal_generator.py     # Julia set fingerprints
│   │   ├── phase_analysis.py        # Dynamical systems analysis
│   │   ├── harmonic_oscillator.py   # 4D limit cycle visualization
│   │   └── interactive_explorer.py  # Real-time parameter explorer
│   └── figures/                     # Generated figures
│       ├── qdnu_dashboard.png
│       ├── fidelity_distribution.png
│       ├── phase_analysis/          # Phase space analysis
│       └── harmonic_oscillator/     # Limit cycle analysis
├── Diagrams/                        # Circuit diagrams
└── README.md
```

---

## Visualization Gallery

### Bloch Spheres & Julia Fractals

The quantum state of each PN neuron can be visualized as:
- **Bloch spheres** for E (excitatory) and I (inhibitory) qubits
- **Julia set fractals** as unique "fingerprints" of the quantum state
- **Concurrence** measuring E-I entanglement

### Phase Space Analysis

- 2D/3D phase portraits
- Fixed point analysis with stability classification
- Nullclines and vector fields
- Bifurcation diagrams

### 4D Harmonic Oscillator

Under periodic driving, the system traces limit cycles in 4D space (a, b, c, concurrence):
- 3D trajectory visualization
- Poincaré sections
- Lissajous phase relationships

---

## References

- Gupta, A., et al. (2024). Positive-negative neuron model for excitatory-inhibitory neural dynamics.
- Holevo, A. S. (1973). Bounds for the quantity of information transmitted by a quantum communication channel.
- Mormann, F., et al. (2007). Seizure prediction: the long and winding road. Brain.
- American Epilepsy Society Seizure Prediction Challenge. Kaggle Competition.

---

## License

Research use only. Contact author for collaboration.

---

## Citation

```bibtex
@article{appenzeller2025qdnu,
  title={Quantum Positive-Negative Neuron Architecture for Multi-Channel EEG Analysis},
  author={Appenzeller, James},
  year={2025},
  url={https://github.com/jappenzeller/QDNU}
}
```
