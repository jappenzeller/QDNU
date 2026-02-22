# Window Size Impact on Seizure Detection Performance

Generated: 2026-02-22 09:13:43

## Configuration

- **Segment duration**: 30s
- **Channels**: 8 (CH8)
- **Subjects**: chb01, chb03, chb05, chb07, chb11, chb14, chb21
- **Window sizes tested**: [1, 2, 5, 10, 15, 20, 30]

---

## Tier 2 MAX (Correlation Eigenvalues + XGBoost)

| Window (s) | Sub-windows | MAX AUC | MEAN AUC | STD AUC |
|------------|-------------|---------|----------|---------|
| 1 | 52 | 0.8026 | 0.6551 | 0.6817 |
| 2 | 26 | 0.7847 | 0.6332 | 0.8004 |
| 5 | 10 | 0.7954 | 0.6722 | 0.8586 |
| 10 | 5 | 0.7730 | 0.6823 | N/A |
| 15 | 3 | 0.7779 | 0.7121 | N/A |
| 20 | 2 | 0.8726 | 0.7431 | N/A |
| 30 | 2 | 0.8342 | 0.7205 | N/A |

---

## Tier 3 Riemannian (Tangent Space + LDA)

| Window (s) | Sub-windows | AUC |
|------------|-------------|-----|
| 1 | 30 | 0.4408 |
| 2 | 15 | 0.4461 |
| 5 | 6 | 0.4481 |
| 10 | 3 | 0.4579 |
| 15 | 2 | 0.4626 |
| 20 | 1 | 0.4930 |
| 30 | 1 | 0.4598 |

---

## Covariance Matrix Condition Number

| Window (s) | Median Condition (raw) | Median Condition (reg) | % Ill-conditioned |
|------------|------------------------|------------------------|-------------------|
| 1 | 106.6 | 67.3 | 4.1% |
| 2 | 77.5 | 60.1 | 2.4% |
| 5 | 63.5 | 55.7 | 1.0% |
| 10 | 55.3 | 51.5 | 0.6% |
| 15 | 53.3 | 50.8 | 0.2% |
| 20 | 51.0 | 49.0 | 0.3% |
| 30 | 49.1 | 47.4 | 0.0% |

---

## Key Findings

### 1. Tier 2 Performance by Window Size

- **Best window**: 20.0s (AUC = 0.8726)
- **Worst window**: 10.0s (AUC = 0.7730)
- **Improvement**: 0.0996 (12.9%)

### 2. Tier 3 Performance by Window Size

- **Best window**: 20.0s (AUC = 0.4930)
- **Worst window**: 1.0s (AUC = 0.4408)
- **Improvement**: 0.0521

### 3. Tier 2 vs Tier 3 Sensitivity to Window Size

- Tier 2 improves **more** with window size (+0.0996) than Tier 3 (+0.0521)
- Eigenvalue features benefit equally from longer windows

### 4. Quantum Improvement Projection

- Current Tier 2 AUC (at 1.95s): 0.7234
- Best Tier 2 AUC (at 20.0s): 0.8726
- **Projected improvement from window size alone**: +0.1492

If quantum encoding benefits proportionally from longer windows, this suggests up to +0.149 AUC improvement may be achievable by using 20.0s windows instead of 1.95s.

---

## SPD Manifold Argument Assessment

**Condition number decreases with window size**: 67.3 (at 1.0s) -> 47.4 (at 30.0s)

This supports the hypothesis that longer windows produce better-conditioned SPD matrices.

**NOT STRONGLY SUPPORTED**: Eigenvalue features benefit similarly from longer windows.
The quantum encoding bottleneck may not be primarily a window size / covariance quality issue.
