# Quantum Positive-Negative Neuron (QDNU)

A quantum computing architecture for multi-channel EEG seizure prediction based on the Positive-Negative (PN) neuron model.

**Author:** James Appenzeller, Independent Researcher

[![Platform](https://img.shields.io/badge/Platform-qdnu.ai-00d4ff)](https://qdnu.ai)
[![Visualization](https://img.shields.io/badge/Viz-Interactive_3D-f59e0b)](https://qdnu.ai/viz/)
[![Paper](https://img.shields.io/badge/Paper-quantum__pn__neuron__paper.md-blue)](paper/quantum_pn_neuron_paper.md)

---

## Abstract

This project presents a quantum computing architecture based on the Positive-Negative (PN) neuron model for multi-channel electroencephalogram (EEG) seizure prediction. The proposed A-Gate circuit encodes excitatory-inhibitory dynamics using paired qubits with parameterized rotation gates, leveraging quantum entanglement to capture inter-channel phase synchronization efficiently.

Validated on the CHB-MIT Scalp EEG Database using Leave-One-Subject-Out (LOSO) cross-validation, the 8-channel quantum circuit (17 qubits) achieves **0.637 AUC** on IBM Heron r2 hardware after polarity calibration, compared to **0.7419 AUC** for the strongest classical baseline.

---

## Live Platform

**[https://qdnu.ai](https://qdnu.ai)** — Project portal with research overview

**[https://qdnu.ai/viz/](https://qdnu.ai/viz/)** — Interactive 3D visualization

The visualization shows gate-by-gate quantum state trajectories on a PCA projection of the SPD covariance manifold:

- **Circuit Diagram** — Real-time gate execution highlighting
- **SPD Manifold** — 3D trajectory with polarity divergence between patients
- **IBM Heron Topology** — Physical qubit layout (ibm_torino, 17 qubits)
- **Audio Sonification** — Boundary distance mapped to pitch

---

## Key Results

### Hardware Validation (IBM Heron r2, CHB-MIT, LOSO)

| Metric | Value |
|--------|-------|
| **Calibrated Hardware AUC** | **0.637** |
| Raw Hardware AUC | 0.531 |
| Classical Baseline (XGBoost) | 0.7419 |
| Hardware-Classical Gap | 0.105 (14% relative) |
| Gap Reduction from Calibration | 56.4% |

### Per-Patient Hardware Results

| Patient | Raw AUC | Calibrated AUC | Polarity |
|---------|---------|----------------|----------|
| chb01   | 0.686   | 0.686          | Standard |
| chb03   | 0.436   | 0.564          | Inverted |
| chb05   | 0.610   | 0.610          | Standard |
| chb07   | 0.667   | 0.667          | Standard |
| chb11   | 0.283   | 0.717          | Inverted |
| chb14   | 0.600   | 0.600          | Standard |
| chb21   | 0.388   | 0.613          | Inverted |

### Key Findings

1. **Polarity Calibration**: Patient-specific polarity inversion corrects for individual differences in seizure manifestation, reducing the hardware-classical gap by 56%.

2. **Encoding Geometry**: PLV-based phase encoding captures temporal dynamics that correlation eigenvalues miss. CC_freq encoding produces 0.87 fidelity between ictal/interictal states (non-discriminative).

3. **O(M) vs O(M²) Scaling**: Quantum architecture encodes M-channel correlations in O(M) gates versus O(M²) classical operations.

---

## Architecture

### The A-Gate

The core component is the **A-Gate**, a 2-qubit circuit encoding a single PN neuron channel:

```text
     ┌───┐┌──────────┐     ┌───────────┐
E: ──┤ H ├┤ P(b)     ├──■──┤ Ry(a·π/2) ├──
     └───┘└──────────┘  │  └───────────┘
     ┌───┐┌──────────┐┌─┴─┐┌───────────┐
I: ──┤ H ├┤ P(b)     ├┤ X ├┤ Ry(c·π/2) ├──
     └───┘└──────────┘└───┘└───────────┘
```

**Parameters:**
- `a`: Excitatory amplitude (PLV theta-alpha)
- `b`: Shared phase (E-I coupling)
- `c`: Inhibitory amplitude (PLV theta-alpha)

### 8-Channel Circuit (17 Qubits)

| Property       | Value |
|----------------|-------|
| Logical qubits | 17    |
| CZ gates       | 97    |
| Total gates    | ~200  |
| Circuit depth  | O(M)  |

### Complexity Advantage

| Operation            | Classical | Quantum | Advantage |
|----------------------|-----------|---------|-----------|
| Correlation encoding | O(M²)     | O(M)    | M×        |
| Template matching    | O(M²)     | O(M)    | M×        |
| Parameter storage    | O(M²)     | O(M)    | M×        |

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
- qiskit-ibm-runtime (for hardware execution)
- NumPy, SciPy, scikit-learn, XGBoost
- pyedflib (for CHB-MIT EDF files)

## Quick Start

```python
from qdnu import create_single_channel_agate

# Create A-Gate circuit with PLV parameters
circuit = create_single_channel_agate(a=0.6, b=1.2, c=0.4)
print(circuit.draw())
```

### Run Hardware Validation

```bash
# Requires IBM Quantum credentials
python scripts/hardware_validation.py --patient chb01 --backend ibm_torino
```

### Generate 3D Assets

```bash
export MESHY_API_KEY="your_key"
python scripts/generate_meshy_assets.py
```

---

## Project Structure

```
QDNU/
├── qdnu/                      # Core quantum library
│   ├── quantum_agate.py       # A-Gate circuit implementation
│   ├── pn_dynamics.py         # PN neuron dynamics
│   └── multichannel_circuit.py
├── scripts/                   # Executable scripts
│   ├── hardware_validation.py # IBM Heron execution
│   ├── tier3_combined_loso.py # Classical baseline
│   ├── generate_meshy_assets.py # 3D asset generation
│   └── generate_figures.py    # Publication figures
├── sagemaker/                 # AWS SageMaker training
│   └── train_chbmit.py        # CHB-MIT preprocessing
├── qdnu-infra/                # Web infrastructure
│   └── static/
│       ├── index.html         # Portal page
│       └── viz/               # 3D visualization
├── results/                   # Experiment outputs
│   ├── hardware_validation/   # IBM hardware results
│   └── patient_analysis/      # Per-patient profiles
├── paper/                     # Publication materials
│   ├── quantum_pn_neuron_paper.md
│   └── figures/
└── .aws/                      # AWS configuration
    └── config                 # SSO profile for deployment
```

---

## Dataset

**CHB-MIT Scalp EEG Database** (PhysioNet)

- 22 patients (pediatric subjects with intractable seizures)
- 8 EEG channels (bipolar montage): FP1-F7, F7-T7, FP1-F3, F3-C3, FP2-F8, F8-T8, FP2-F4, F4-C4
- 256 Hz sampling rate
- Leave-One-Subject-Out cross-validation

---

## Classical Baseline

**Tier 3 Combined (XGBoost):** 0.7419 AUC

| Feature Set                           | AUC        |
|---------------------------------------|------------|
| log-FFT (1-47Hz) + CC eigenvalues     | **0.7419** |
| Correlation eigenvalues (MAX pooling) | 0.7234     |
| Riemannian tangent space              | 0.6314     |
| CC_freq eigenvalues only              | 0.5445     |

---

## References

- Shoeb, A. H. (2009). Application of machine learning to epileptic seizure onset detection and treatment. MIT PhD thesis.
- Gupta, A., et al. (2024). Positive-negative neuron model for excitatory-inhibitory neural dynamics.
- IBM Quantum. (2024). Heron processor architecture.

---

## License

Research use only. Contact author for collaboration.

---

## Citation

```bibtex
@article{appenzeller2026qdnu,
  title={Quantum Positive-Negative Neuron Architecture for Multi-Channel EEG Seizure Prediction},
  author={Appenzeller, James},
  year={2026},
  url={https://qdnu.ai},
  note={IBM Heron r2 hardware validation, CHB-MIT dataset}
}
```
