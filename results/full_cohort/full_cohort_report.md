# PROMPT 018 - Full Cohort Polarity Analysis

Generated: 2026-02-22T11:31:32.430612


## Configuration

- Channels: 8 (8-channel quantum configuration)
- Window: 20.0s
- Subjects: 22 (excluding ['chb12', 'chb24'])
- Total segments: 494

## Full Cohort Results

| Method | Features | Raw AUC | Calibrated | Lift | Inverted |
|--------|----------|---------|------------|------|----------|
| Tier 2A (eigenvals) | 8 | 0.521 | 0.628 | +0.107 | 10/22 |
| Tier 2B (full) | 336 | 0.952 | 0.952 | +0.000 | 0/22 |
| Tier 3 (Riemannian) | 36 | 0.536 | 0.659 | +0.123 | 6/22 |
| Quantum PLV CH8* | 8 | 0.524 | 0.637 | +0.113 | 3/7 |

*Quantum results from 7-patient subset only

## Feature Regime Analysis

| Regime | Tier 2 | Tier 3 Cal | Winner | Why |
|--------|--------|------------|--------|-----|
| Minimal (8) | 0.63 | 0.66 | Tier 3 | Eigenvalues lose info |
| Full (~336) | 0.95 | 0.66 | Tier 2B | Redundancy absorbs |

## Polarity Concordance Matrix

| Patient | T2A Pol | T2B Pol | T3 Pol |
|---------|---------|---------|--------|
| chb01 | i | s | s |
| chb02 | s | s | s |
| chb03 | i | s | i |
| chb04 | s | s | s |
| chb05 | s | s | s |
| chb06 | s | s | i |
| chb07 | i | s | s |
| chb08 | i | s | i |
| chb09 | s | s | s |
| chb10 | s | s | i |
| chb11 | i | s | s |
| chb13 | s | s | s |
| chb14 | i | s | s |
| chb15 | s | s | s |
| chb16 | i | s | s |
| chb17 | i | s | s |
| chb18 | s | s | s |
| chb19 | s | s | s |
| chb20 | s | s | i |
| chb21 | i | s | s |
| chb22 | i | s | i |
| chb23 | s | s | s |

**Concordance rates:**
- T2A vs T2B: 54.5%
- T2A vs T3: 54.5%
- T2B vs T3: 72.7%
- All three agree: 40.9%

## Per-Subject Details

| Subject | T2A Raw | T2A Cal | T2B Raw | T2B Cal | T3 Raw | T3 Cal |
|---------|---------|---------|---------|---------|--------|--------|
| chb01 | 0.421 | 0.579 | 0.864 | 0.864 | 0.864 | 0.864 |
| chb02 | 0.640 | 0.640 | 0.980 | 0.980 | 0.660 | 0.660 |
| chb03 | 0.293 | 0.707 | 0.993 | 0.993 | 0.300 | 0.700 |
| chb04 | 0.662 | 0.662 | 1.000 | 1.000 | 0.550 | 0.550 |
| chb05 | 0.600 | 0.600 | 0.980 | 0.980 | 0.760 | 0.760 |
| chb06 | 0.565 | 0.565 | 0.520 | 0.520 | 0.335 | 0.665 |
| chb07 | 0.467 | 0.533 | 1.000 | 1.000 | 0.500 | 0.500 |
| chb08 | 0.390 | 0.610 | 0.990 | 0.990 | 0.130 | 0.870 |
| chb09 | 0.600 | 0.600 | 1.000 | 1.000 | 0.512 | 0.512 |
| chb10 | 0.500 | 0.500 | 1.000 | 1.000 | 0.207 | 0.793 |
| chb11 | 0.267 | 0.733 | 1.000 | 1.000 | 0.517 | 0.517 |
| chb13 | 0.537 | 0.537 | 0.908 | 0.908 | 0.512 | 0.512 |
| chb14 | 0.438 | 0.562 | 0.913 | 0.913 | 0.550 | 0.550 |
| chb15 | 0.578 | 0.578 | 0.960 | 0.960 | 0.547 | 0.547 |
| chb16 | 0.445 | 0.555 | 1.000 | 1.000 | 0.645 | 0.645 |
| chb17 | 0.417 | 0.583 | 1.000 | 1.000 | 0.617 | 0.617 |
| chb18 | 0.800 | 0.800 | 1.000 | 1.000 | 0.517 | 0.517 |
| chb19 | 0.700 | 0.700 | 1.000 | 1.000 | 0.733 | 0.733 |
| chb20 | 0.675 | 0.675 | 0.962 | 0.962 | 0.463 | 0.537 |
| chb21 | 0.250 | 0.750 | 0.887 | 0.887 | 0.975 | 0.975 |
| chb22 | 0.433 | 0.567 | 1.000 | 1.000 | 0.217 | 0.783 |
| chb23 | 0.786 | 0.786 | 0.986 | 0.986 | 0.679 | 0.679 |

## Subgroup Analysis

**Tier 2A (eigenvalues) polarity:**
- Inverted (10): chb01, chb03, chb07, chb08, chb11, chb14, chb16, chb17, chb21, chb22
- Standard (12): chb02, chb04, chb05, chb06, chb09, chb10, chb13, chb15, chb18, chb19, chb20, chb23

**Tier 2B (full features) polarity:**
- Inverted (0): 
- Standard (22): chb01, chb02, chb03, chb04, chb05, chb06, chb07, chb08, chb09, chb10, chb11, chb13, chb14, chb15, chb16, chb17, chb18, chb19, chb20, chb21, chb22, chb23

**Tier 3 (Riemannian) polarity:**
- Inverted (6): chb03, chb06, chb08, chb10, chb20, chb22
- Standard (16): chb01, chb02, chb04, chb05, chb07, chb09, chb11, chb13, chb14, chb15, chb16, chb17, chb18, chb19, chb21, chb23

## Wilcoxon Signed-Rank Tests

| Comparison | Mean A | Mean B | p-value | Significant |
|------------|--------|--------|---------|-------------|
| T3 cal vs T2A cal | 0.659 | 0.628 | 0.4358 | No |
| T3 cal vs T2B cal | 0.659 | 0.952 | 0.0001 | Yes |
| T2A cal vs T2B cal | 0.628 | 0.952 | 0.0000 | Yes |

*Wilcoxon signed-rank test, two-sided, alpha=0.05*