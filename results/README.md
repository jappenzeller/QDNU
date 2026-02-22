# QDNU Results Directory

This directory contains experimental results for the Quantum Positive-Negative Neuron (QDNU) seizure prediction project.

## Primary Classical Baseline

**AUC: 0.7419** (Tier 3 Combined Features)

Configuration:
- **Features**: log-FFT (1-47 Hz, 5 bands) + CC_freq eigenvalues + CC_time eigenvalues
- **Total features**: 56 (40 spectral + 8 + 8 eigenvalues)
- **Classifier**: XGBoost (200 trees, depth 6, 5-bag ensemble)
- **Aggregation**: MAX pooling over 1-second sub-windows
- **Validation**: Leave-One-Subject-Out (LOSO) cross-validation
- **Dataset**: CHB-MIT, 8 patients
- **Channels**: 8 (standard 10-20 montage subset)

## Hardware-Classical Gap

| Metric | Value |
|--------|-------|
| Classical Baseline | 0.7419 |
| Quantum Hardware (calibrated) | 0.637 |
| **Gap** | **0.105** |

---

## Directory Structure

### `/loso_cv/`
LOSO cross-validation results comparing classical feature extraction approaches.

| File | Description |
|------|-------------|
| `tier1_loso_results.json` | FFT band power features |
| `tier2_loso_max_results.json` | Correlation eigenvalues, MAX pooling (0.7234 AUC) |
| `tier2_loso_mean_results.json` | Correlation eigenvalues, MEAN pooling |
| `tier3_*.json` | Riemannian geometry approaches (MDM, TS+LDA, TS+XGBoost) |
| `loso_comparison_final.md` | Summary comparison table |

### `/tier3_cc_freq/`
V6 CC_freq baseline experiments (combined features).

| File | Description |
|------|-------------|
| `tier3_cc_freq_loso.json` | CC_freq eigenvalues only (0.5445 AUC) |
| `tier3_cc_time_loso.json` | CC_time eigenvalues only (0.5106 AUC) |
| `tier3_combined_loso.json` | Combined features (**0.7419 AUC**) |
| `tier3_combined_selected_loso.json` | With RF feature selection (0.7231 AUC) |
| `tier3_cc_freq_baseline.json` | All results combined |

### `/hardware_validation/`
IBM Heron r2 quantum hardware validation results.

| File | Description |
|------|-------------|
| `heron_8ch_validation.json` | 8-channel hardware run |
| `encoding_audit.json` | Encoding parameter verification |

### `/calibration/`
Polarity calibration results for patient-specific correction.

### `/scaling/`
Channel scaling benchmarks (4, 8, 12, 16 channels).

### `/patient_analysis/`
Per-patient quantum boundary analysis.

---

## Key Findings

### Eigenvalues Alone Underperform

| Method | AUC | vs Combined |
|--------|-----|-------------|
| CC_freq eigenvalues | 0.5445 | -0.197 |
| CC_time eigenvalues | 0.5106 | -0.231 |
| Combined | 0.7419 | baseline |

**Insight**: Correlation eigenvalues capture network structure but lack the spectral power context needed for seizure discrimination. This is consistent with quantum findings: static eigenvalue encoding produces high state fidelity (0.87) between classes, indicating poor discriminability. Phase-based encoding (PLV) captures temporal dynamics and remains the preferred quantum approach.

### Aggregation Matters

MAX pooling consistently outperforms MEAN pooling, consistent with the transient nature of seizure precursors.

---

## Reproducing Results

```bash
# Tier 3 Combined (primary baseline)
python experiments/v6_cc_freq_baseline.py

# Tier 1/2 comparison
python scripts/tier1_tier2_loso_cv.py

# Tier 3 Riemannian
python scripts/tier3_loso_cv.py
```

---

*Last updated: 2026-02-22*
