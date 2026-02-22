# QDNU Paper Abstract — Draft

**Title:** Quantum Positive-Negative Neuron Architecture for Multi-Channel EEG Seizure Prediction

## Abstract

We present a quantum computing architecture based on the Positive-Negative (PN) neuron model for multi-channel electroencephalogram (EEG) seizure prediction. The proposed A-Gate circuit encodes excitatory-inhibitory dynamics using paired qubits with parameterized rotation gates, leveraging quantum entanglement to capture inter-channel phase synchronization efficiently.

Validated on the CHB-MIT Scalp EEG Database using Leave-One-Subject-Out (LOSO) cross-validation, our 8-channel quantum circuit (17 qubits) achieves **0.637 AUC** on IBM Heron r2 hardware after polarity calibration, compared to **0.7419 AUC** for the strongest classical baseline (combined log-FFT spectral power + correlation eigenvalues, XGBoost classifier). The hardware-classical gap of **0.105 AUC** represents a 14% relative difference.

Key findings:
- **Encoding geometry matters**: Correlation eigenvalues alone achieve only 0.54 AUC, demonstrating that spectral context is essential for discrimination. This parallels the quantum finding that PLV-based phase encoding (which captures temporal dynamics) outperforms static eigenvalue encoding.
- **Polarity calibration reduces gap by 56%**: Patient-specific polarity inversion corrects for individual differences in seizure manifestation, reducing the raw hardware-classical gap from 0.24 to 0.105.
- **O(M) vs O(M²) scaling**: The quantum architecture encodes M-channel correlations in O(M) gates versus O(M²) classical operations, with theoretical advantage realized at scale.

---

## Key Results

### Classical Baselines (LOSO, 8-channel, CHB-MIT)

| Method | AUC | Description |
|--------|-----|-------------|
| **Tier 3 Combined** | **0.7419** | log-FFT (1-47Hz) + CC_freq eigenvalues + CC_time eigenvalues |
| Tier 2 MAX | 0.7234 | Correlation eigenvalues, 1s sub-windows, MAX pooling |
| Tier 3 TS+XGBoost | 0.6314 | Riemannian tangent space + XGBoost |
| CC_freq eigenvalues only | 0.5445 | Frequency-domain correlation eigenvalues |
| CC_time eigenvalues only | 0.5106 | Time-domain correlation eigenvalues |

### Quantum Hardware (IBM Heron r2, 8-channel)

| Metric | Value |
|--------|-------|
| Raw Hardware AUC | 0.531 |
| Calibrated Hardware AUC | 0.637 |
| Gap vs Classical (0.7419) | 0.105 |
| Gap Reduction from Calibration | 56.4% |

### Encoding Insight

The finding that eigenvalues alone underperform spectral features (0.54 vs 0.74 AUC) is consistent with quantum simulation results showing that CC_freq encoding produces high fidelity (0.87) between ictal and interictal states—indicating poor discriminability. The PLV-based encoding, which captures phase synchronization dynamics rather than static correlation structure, remains the preferred quantum encoding.

---

## Baseline Configuration

**Primary Classical Baseline (Tier 3 Combined):**
- Features: log10 FFT magnitudes (1-47 Hz, 5 bands) + CC_freq eigenvalues + CC_time eigenvalues
- Total features: 56 (40 spectral + 8 freq eigenvalues + 8 time eigenvalues)
- Classifier: XGBoost (200 trees, depth 6, 5-bag ensemble)
- Temporal aggregation: MAX pooling over 1-second sub-windows
- Validation: Leave-One-Subject-Out cross-validation
- Dataset: CHB-MIT, 8 patients (chb01-chb11 subset)
- Channels: 8 (FP1-F7, F7-T7, FP1-F3, F3-C3, FP2-F8, F8-T8, FP2-F4, F4-C4)

---

## Version History

- **v6 (2026-02-22)**: Updated baseline to 0.7419 AUC using combined features
- **v5**: Tier 2 MAX baseline at 0.7234 AUC
- **v4**: Initial hardware validation on IBM Heron

---

*Last updated: 2026-02-22*
