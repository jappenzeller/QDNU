# LOSO Cross-Patient CV Results — Final Comparison

## All Tiers Comparison

| Classifier                | Channels | Window | Overall AUC | Delta vs Paper |
|---------------------------|----------|--------|-------------|----------------|
| XGBoost-8ch (paper)       | 8        | 30s    | 0.5989      | baseline       |
| XGBoost-8ch-V2 (paper)    | 8        | 30s    | 0.6253      | baseline       |
| Tier 1 FFT Bands          | 8        | 30s    | 0.5869      | -0.038         |
| Tier 2 Corr-Eig MEAN      | 8        | 30s    | 0.5674      | -0.058         |
| Tier 2 Corr-Eig MAX       | 8        | 30s    | 0.7234      | +0.098         |
| Tier 3 MDM (30s)          | 8        | 30s    | 0.5664      | -0.059         |
| Tier 3 TS+LDA (30s)       | 8        | 30s    | 0.5686      | -0.057         |
| Tier 3 TS+XGBoost (30s)   | 8        | 30s    | 0.5658      | -0.060         |

## Sub-Window Aggregation Comparison

| Classifier                  | Window unit | Aggregation | Overall AUC | Delta vs Tier2 MAX |
|-----------------------------|-------------|-------------|-------------|--------------------|
| Tier 2 Corr-Eig MAX         | 1s          | MAX         | 0.7234      | baseline           |
| Tier 3 MDM (30s window)     | 30s         | none        | 0.5664      | -0.157             |
| Tier 3 TS+XGBoost (30s)     | 30s         | none        | 0.5658      | -0.158             |
| Tier 3 MDM MAX              | 1s          | MAX         | 0.5728      | -0.151             |
| Tier 3 MDM MEAN             | 1s          | MEAN        | 0.5699      | -0.154             |
| Tier 3 TS+XGBoost MAX       | 1s          | MAX         | 0.6314      | -0.092             |
| Tier 3 TS+XGBoost MEAN      | 1s          | MEAN        | 0.6087      | -0.115             |

## Configuration

- **Channels**: 8 (matching paper)
- **Segment Window**: 30 seconds
- **Sub-window**: 1 second (30 per segment)
- **Covariance Estimator**: Ledoit-Wolf regularization
- **XGBoost**: 5-bag ensemble, same hyperparameters as paper

Generated: 2026-02-21 09:47:41
