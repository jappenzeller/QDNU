# LOSO Cross-Patient CV Results — Final Comparison

## Primary Baseline

**Tier 3 Combined: 0.7419 AUC** (log-FFT + CC_freq eigenvalues + CC_time eigenvalues)

## All Tiers Comparison

| Classifier                | Channels | Window | Overall AUC | Delta vs Combined |
|---------------------------|----------|--------|-------------|-------------------|
| **Tier 3 Combined**       | 8        | 30s    | **0.7419**  | baseline          |
| Tier 2 Corr-Eig MAX       | 8        | 30s    | 0.7234      | -0.019            |
| Tier 3 TS+XGBoost MAX     | 8        | 1s     | 0.6314      | -0.111            |
| XGBoost-8ch-V2 (paper)    | 8        | 30s    | 0.6253      | -0.117            |
| XGBoost-8ch (paper)       | 8        | 30s    | 0.5989      | -0.143            |
| Tier 1 FFT Bands          | 8        | 30s    | 0.5869      | -0.155            |
| CC_freq eigenvalues only  | 8        | 30s    | 0.5445      | -0.197            |
| CC_time eigenvalues only  | 8        | 30s    | 0.5106      | -0.231            |

## Eigenvalue-Only Performance

| Method                   | AUC    | Note |
|--------------------------|--------|------|
| CC_freq eigenvalues      | 0.5445 | Frequency-domain correlation |
| CC_time eigenvalues      | 0.5106 | Time-domain correlation |
| Combined (log-FFT + both)| 0.7419 | Spectral context restores performance |

**Key insight**: Eigenvalues alone underperform spectral features (0.54 AUC vs 0.74 AUC). This is consistent with quantum findings: static eigenvalue encoding produces high state fidelity (0.87) between ictal/interictal, indicating poor discriminability. Encoding geometry matters for discrimination.

## Hardware-Classical Gap

| Metric | Value |
|--------|-------|
| Classical Baseline (Tier 3 Combined) | 0.7419 |
| Quantum Hardware (IBM Heron, calibrated) | 0.637 |
| **Gap** | **0.105** |
| Gap Reduction from Calibration | 56.4% |

## Sub-Window Aggregation Comparison

| Classifier                  | Window unit | Aggregation | Overall AUC | Delta vs Combined |
|-----------------------------|-------------|-------------|-------------|-------------------|
| **Tier 3 Combined**         | 30s         | -           | **0.7419**  | baseline          |
| Tier 2 Corr-Eig MAX         | 1s          | MAX         | 0.7234      | -0.019            |
| Tier 3 TS+XGBoost MAX       | 1s          | MAX         | 0.6314      | -0.111            |
| Tier 3 TS+XGBoost MEAN      | 1s          | MEAN        | 0.6087      | -0.133            |
| Tier 3 MDM MAX              | 1s          | MAX         | 0.5728      | -0.169            |

## Configuration

- **Channels**: 8 (FP1-F7, F7-T7, FP1-F3, F3-C3, FP2-F8, F8-T8, FP2-F4, F4-C4)
- **Segment Window**: 30 seconds
- **Sub-window**: 1 second (for MAX aggregation approaches)
- **Features (Tier 3 Combined)**: 56 total (40 log-FFT bands + 8 CC_freq eigenvalues + 8 CC_time eigenvalues)
- **Classifier**: XGBoost 5-bag ensemble (200 trees, depth 6)
- **Validation**: Leave-One-Subject-Out (LOSO)
- **Dataset**: CHB-MIT, 8 patients

Generated: 2026-02-22 08:45:00
