# LOSO Cross-Patient CV Results — Final Comparison

## All Tiers Comparison

| Classifier                | Channels | Window | Overall AUC | Delta vs Paper |
|---------------------------|----------|--------|-------------|----------------|
| XGBoost-8ch (paper)       | 8        | 30s    | 0.5989      | baseline       |
| XGBoost-8ch-V2 (paper)    | 8        | 30s    | 0.6253      | baseline       |
| Tier 1 FFT Bands          | 8        | 30s    | 0.5869      | -0.038         |
| Tier 2 Corr-Eig MEAN      | 8        | 30s    | 0.5674      | -0.058         |
| Tier 2 Corr-Eig MAX       | 8        | 30s    | 0.7234      | +0.098         |
| Tier 3 MDM                | 8        | 30s    | 0.5664      | -0.059          |
| Tier 3 TS+LDA             | 8        | 30s    | 0.5686      | -0.057          |
| Tier 3 TS+XGBoost         | 8        | 30s    | 0.5658      | -0.060          |

## Conclusion

**Best Tier 3 variant:** TS+LDA with AUC = 0.5686

**Tier 3 vs Tier 2 MAX (0.7234):**
- Tier 3 TS+LDA is BELOW Tier 2 MAX by -0.1548

**Strongest classical baseline overall:** Tier 2 MAX at 0.7234 AUC

**Riemannian SPD geometry assessment:**
- Riemannian geometry does NOT outperform correlation eigenvalue features cross-patient
- The eigenvalue summary (Tier 2 MAX) already captures the discriminative information
- The geometric argument for quantum approaches needs reexamination — simpler features may suffice

## Configuration

- **Channels**: 8 (matching paper)
- **Window**: 30 seconds (with 10s fallback for short seizures)
- **CV Method**: Leave-One-Subject-Out (LOSO)
- **Covariance Estimator**: Ledoit-Wolf regularization
- **XGBoost**: 5-bag ensemble, same hyperparameters as paper

Generated: 2026-02-21 09:28:14
