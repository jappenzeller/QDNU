# Channel Scaling Comparison

## Results

| Method          | 8ch AUC | 12ch AUC | 16ch AUC | Trend |
|-----------------|---------|----------|----------|-------|
| Classical Tier2 | 0.7234  | 0.7116   | 0.7184   | flat |
| Quantum QPNN    | TBD     | TBD      | TBD      | TBD   |

## Channel Sets (Nested)

- **CH8**: ['FP1-F7', 'F7-T7', 'FP1-F3', 'F3-C3', 'FP2-F8', 'F8-T8', 'FP2-F4', 'F4-C4']
- **CH12**: ['FP1-F7', 'F7-T7', 'FP1-F3', 'F3-C3', 'FP2-F8', 'F8-T8', 'FP2-F4', 'F4-C4', 'C3-P3', 'C4-P4', 'CZ-PZ', 'FZ-CZ']
- **CH16**: ['FP1-F7', 'F7-T7', 'FP1-F3', 'F3-C3', 'FP2-F8', 'F8-T8', 'FP2-F4', 'F4-C4', 'C3-P3', 'C4-P4', 'CZ-PZ', 'FZ-CZ', 'P3-O1', 'P4-O2', 'P7-O1', 'P8-O2']

## Configuration

- Method: Tier 2 MAX (correlation eigenvalues with MAX pooling)
- Classifier: XGBoost 5-bag ensemble
- CV: Leave-One-Subject-Out

Generated: 2026-02-21 10:02:17
