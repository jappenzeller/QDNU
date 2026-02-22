# Quantum Positive-Negative Neuron (QPNN)

**A quantum computing architecture for multi-channel EEG seizure prediction with IBM Heron hardware validation.**

[![Platform](https://img.shields.io/badge/Platform-qdnu.ai-00d4ff)](https://qdnu.ai)
[![Visualization](https://img.shields.io/badge/Viz-Interactive_3D-f59e0b)](https://qdnu.ai/viz/)
[![arXiv](https://img.shields.io/badge/arXiv-cs.LG-b31b1b)](arxiv_submission/main_v2.pdf)

**Author:** James Appenzeller, Independent Researcher

---

## Abstract

We present a quantum computing architecture based on the Positive-Negative (PN) neuron model for multi-channel electroencephalogram (EEG) seizure prediction. The proposed A-Gate circuit encodes excitatory-inhibitory dynamics using paired qubits with parameterized rotation gates, leveraging quantum entanglement to capture inter-channel phase synchronization efficiently.

Validated on the **CHB-MIT Scalp EEG Database** using Leave-One-Subject-Out (LOSO) cross-validation, our 8-channel quantum circuit (17 qubits) achieves **0.637 AUC** on IBM Heron r2 hardware after polarity calibration, compared to **0.7419 AUC** for the strongest classical baseline. The hardware-classical gap of 0.105 AUC (14% relative) represents honest assessment of current quantum limitations.

---

## Key Findings

1. **Polarity Calibration**: Patient-specific polarity inversion corrects for individual differences in seizure manifestation, reducing the raw hardware-classical gap from 0.24 to 0.105 (56% reduction).

2. **Encoding Geometry Matters**: PLV-based phase encoding captures temporal dynamics that correlation eigenvalues miss. Classical eigenvalue-only features achieve 0.54 AUC vs 0.74 AUC with spectral context—paralleling the quantum finding that encoding strategy constitutes the primary performance bottleneck.

3. **O(M) vs O(M²) Scaling**: The quantum architecture encodes M-channel correlations in O(M) gates versus O(M²) classical pairwise operations, with theoretical advantage realized at scale.

---

## Live Platform

| URL | Description |
|-----|-------------|
| [qdnu.ai](https://qdnu.ai) | Project portal with research overview |
| [qdnu.ai/viz/](https://qdnu.ai/viz/) | Interactive 3D quantum state visualization |

The visualization shows gate-by-gate quantum state trajectories on a PCA projection of the SPD covariance manifold, including:
- Real-time circuit diagram execution
- 3D manifold trajectory with polarity divergence
- IBM Heron physical qubit topology (ibm_torino)
- Audio sonification of boundary distance

---

## Results

### Hardware Validation (IBM Heron r2, CHB-MIT, LOSO)

| Metric                       | Value                 |
|------------------------------|-----------------------|
| **Calibrated Hardware AUC**  | **0.637**             |
| Raw Hardware AUC             | 0.531                 |
| Classical Baseline (XGBoost) | 0.7419                |
| Hardware-Classical Gap       | 0.105 (14% relative)  |
| Gap Reduction from Calibration | 56.4%               |

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

### Classical Baselines (LOSO, 8-channel)

| Method                                  | AUC        |
|-----------------------------------------|------------|
| **Tier 3 Combined** (log-FFT + CC eig)  | **0.7419** |
| Tier 2 MAX (correlation eigenvalues)    | 0.7234     |
| Riemannian tangent space + XGBoost      | 0.6314     |
| CC_freq eigenvalues only                | 0.5445     |
| CC_time eigenvalues only                | 0.5106     |

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

**Parameters (PLV theta-alpha encoding):**
- `a`: Excitatory amplitude — PLV between theta (4-8 Hz) and alpha (8-13 Hz) bands
- `b`: Shared phase — E-I coupling, encodes temporal dynamics
- `c`: Inhibitory amplitude — PLV between complementary frequency pairs

### Multi-Channel Circuit (8 channels, 17 qubits)

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

For 19-channel clinical EEG: theoretical 19× reduction in correlation complexity.

---

## Dataset

**CHB-MIT Scalp EEG Database** (PhysioNet)

- 22 pediatric subjects with intractable seizures
- 8 EEG channels (bipolar montage): FP1-F7, F7-T7, FP1-F3, F3-C3, FP2-F8, F8-T8, FP2-F4, F4-C4
- 256 Hz sampling rate
- 10-second windows, 1-second sub-windows with MAX pooling
- Leave-One-Subject-Out cross-validation

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
- pyriemann (for Riemannian geometry baselines)

---

## Quick Start

```python
from qdnu import create_single_channel_agate

# Create A-Gate circuit with PLV parameters
circuit = create_single_channel_agate(a=0.6, b=1.2, c=0.4)
print(circuit.draw())
```

### Run Classical Baseline

```bash
python scripts/tier3_combined_loso.py
```

### Run Hardware Validation

```bash
# Requires IBM Quantum credentials
python scripts/hardware_validation.py --patient chb01 --backend ibm_torino
```

---

## Project Structure

```text
QDNU/
├── qdnu/                      # Core quantum library
│   ├── quantum_agate.py       # A-Gate circuit implementation
│   ├── pn_dynamics.py         # PN neuron dynamics
│   └── multichannel_circuit.py
├── scripts/                   # Executable scripts
│   ├── hardware_validation.py # IBM Heron execution
│   ├── tier3_combined_loso.py # Classical baseline (0.7419 AUC)
│   ├── generate_figures.py    # Publication figures
│   └── generate_meshy_assets.py # 3D asset generation
├── sagemaker/                 # AWS SageMaker training
│   └── train_chbmit.py        # CHB-MIT preprocessing
├── arxiv_submission/          # Paper LaTeX source
│   ├── main_v2.tex            # Current manuscript
│   ├── main_v2.pdf            # Compiled PDF
│   └── figures/               # Publication figures
├── qdnu-infra/                # Web infrastructure (gitignored)
│   └── static/
│       ├── index.html         # Portal page
│       └── viz/               # 3D visualization
├── results/                   # Experiment outputs
│   ├── hardware_validation/   # IBM hardware results
│   └── patient_analysis/      # Per-patient profiles
└── .aws/                      # AWS SSO configuration
```

---

## Supplementary Material

**Interactive Visualization:** [qdnu.ai/viz/](https://qdnu.ai/viz/)

The visualization renders A-Gate circuit evolution on a PCA projection of the SPD covariance manifold for two representative patients:
- **chb01** (standard polarity): Hardware AUC 0.686, boundary distance -0.190
- **chb11** (inverted polarity): Raw AUC 0.283 → Calibrated 0.717, boundary distance +0.211

The polarity inversion is rendered as both visual trajectory divergence and auditory sonification (descending pitch for chb01 ictal, ascending for chb11). Three synchronized panels show the logical circuit diagram, 3D manifold trajectory, and IBM Heron physical qubit topology.

---

## References

- Shoeb, A. H. (2009). Application of machine learning to epileptic seizure onset detection and treatment. MIT PhD thesis.
- Gupta, A., et al. (2003). The Positive-Negative neuron model for neural computation.
- IBM Quantum. (2024). Heron processor architecture.
- Mormann, F., et al. (2007). Seizure prediction: the long and winding road. Brain.

---

## License

Research use only. Contact author for collaboration.

---

## Citation

```bibtex
@article{appenzeller2026qpnn,
  title={Quantum Positive-Negative Neuron Architecture for Multi-Channel EEG Analysis: Hardware Validation and Empirical Limits},
  author={Appenzeller, James},
  year={2026},
  journal={arXiv preprint},
  url={https://qdnu.ai},
  note={IBM Heron r2 hardware validation, CHB-MIT dataset, LOSO cross-validation}
}
```
