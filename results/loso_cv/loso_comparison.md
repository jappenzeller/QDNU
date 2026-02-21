# LOSO Cross-Patient CV Results

## Comparison with Paper Baselines

| Method | Overall AUC | Mean AUC +/- Std | Delta vs XGBoost-V2 |
|--------|-------------|------------------|---------------------|
| XGBoost-8ch-V1 (paper) | 0.5989 | - | -0.0264 |
| XGBoost-8ch-V2 (paper) | 0.6253 | - | +0.0000 |
| Tier 1 (FFT Power) | 0.5869 | 0.6611 +/- 0.2215 | -0.0384 |
| Tier 2 (MAX pool) | 0.7234 | 0.8200 +/- 0.1335 | +0.0981 |
| Tier 2 (MEAN pool) | 0.5674 | 0.6296 +/- 0.2101 | -0.0579 |

## Configuration

- **Channels**: 8 (matching paper)
- **Window**: 30 seconds
- **CV Method**: Leave-One-Subject-Out (LOSO)
- **Classifier**: XGBoost with 5-bag ensemble
- **Hyperparameters**: n_estimators=200, max_depth=6, learning_rate=0.1

Generated: 2026-02-21 09:18:10
