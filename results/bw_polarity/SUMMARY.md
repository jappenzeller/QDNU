# BW-001: Bures-Wasserstein Polarity Replication Check

**Date:** 2026-04-19 14:57
**Subjects:** 22 CHB-MIT patients
**Channels:** 8 (FP1-F7, F7-T7, FP1-F3, F3-C3, FP2-F8, F8-T8, FP2-F4, F4-C4)
**Window:** 30s (10s fallback)

## Four Numbers

- **Concordance rate:** 0.773 (17/22 patients match)
- **Strongly-polar patients (SNR > 1.0):** affine=6, BW=4
- **Mean AUC (affine-invariant):** 0.6130
- **Mean AUC (Bures-Wasserstein):** 0.5660

## Verdict

**Polarity partially replicates under Bures-Wasserstein.**

## Polarity Comparison

| Patient | Pol (AI) | Pol (BW) | AUC (AI) | AUC (BW) | Match |
|---------|----------|----------|----------|----------|-------|
| chb01 | standard | standard | 0.877 | 0.931 | Y |
| chb02 | standard | standard | 0.880 | 0.900 | Y |
| chb03 | standard | inverted | 0.529 | 0.386 | **N** |
| chb04 | standard | standard | 0.938 | 0.825 | Y |
| chb05 | standard | standard | 0.980 | 0.730 | Y |
| chb06 | inverted | inverted | 0.380 | 0.230 | Y |
| chb07 | standard | standard | 0.667 | 0.700 | Y |
| chb08 | inverted | standard | 0.430 | 0.500 | **N** |
| chb09 | standard | standard | 0.900 | 0.962 | Y |
| chb10 | inverted | inverted | 0.436 | 0.371 | Y |
| chb11 | inverted | inverted | 0.380 | 0.400 | Y |
| chb13 | inverted | inverted | 0.460 | 0.445 | Y |
| chb14 | standard | inverted | 0.544 | 0.478 | **N** |
| chb15 | standard | standard | 0.535 | 0.542 | Y |
| chb16 | standard | standard | 0.540 | 0.560 | Y |
| chb17 | standard | standard | 0.650 | 0.867 | Y |
| chb18 | standard | standard | 0.733 | 0.675 | Y |
| chb19 | standard | standard | 0.750 | 0.717 | Y |
| chb20 | standard | inverted | 0.700 | 0.367 | **N** |
| chb21 | standard | inverted | 0.529 | 0.100 | **N** |
| chb22 | inverted | inverted | 0.117 | 0.183 | Y |
| chb23 | standard | standard | 0.533 | 0.583 | Y |

## Moments (M1, M2, SNR)

| Patient | M1(AI) | M2(AI) | SNR(AI) | M1(BW) | M2(BW) | SNR(BW) |
|---------|--------|--------|---------|--------|--------|---------|
| chb01 | 1.231 | 3.358 | 0.45 | 1.612 | 5.615 | 0.46 |
| chb02 | -0.645 | 1.087 | 0.38 | 0.084 | 1.981 | 0.00 |
| chb03 | 0.036 | 3.361 | 0.00 | 0.887 | 9.210 | 0.09 |
| chb04 | -1.162 | 7.662 | 0.18 | -0.550 | 2.835 | 0.11 |
| chb05 | 1.205 | 0.985 | 1.48 | 1.033 | 1.257 | 0.85 |
| chb06 | 1.608 | 0.899 | 2.88 | 1.497 | 2.252 | 0.99 |
| chb07 | 1.120 | 2.619 | 0.48 | 0.573 | 3.126 | 0.11 |
| chb08 | 3.126 | 4.056 | 2.41 | 1.718 | 5.787 | 0.51 |
| chb09 | 1.250 | 2.030 | 0.77 | 1.759 | 10.075 | 0.31 |
| chb10 | 0.898 | 0.580 | 1.39 | 1.489 | 1.464 | 1.51 |
| chb11 | -0.028 | 0.310 | 0.00 | -0.034 | 0.269 | 0.00 |
| chb13 | -1.088 | 1.374 | 0.86 | -0.953 | 0.778 | 1.17 |
| chb14 | -0.564 | 0.376 | 0.85 | -0.135 | 0.320 | 0.06 |
| chb15 | -0.544 | 1.172 | 0.25 | -0.474 | 2.304 | 0.10 |
| chb16 | -0.632 | 1.186 | 0.34 | -0.754 | 1.851 | 0.31 |
| chb17 | -0.814 | 6.551 | 0.10 | -0.173 | 0.796 | 0.04 |
| chb18 | 0.400 | 2.517 | 0.06 | 0.637 | 1.221 | 0.33 |
| chb19 | -1.540 | 1.823 | 1.30 | -0.275 | 0.397 | 0.19 |
| chb20 | -1.281 | 1.368 | 1.20 | -0.860 | 0.410 | 1.80 |
| chb21 | -0.105 | 0.521 | 0.02 | 0.537 | 0.947 | 0.30 |
| chb22 | 0.481 | 0.940 | 0.25 | -0.524 | 1.720 | 0.16 |
| chb23 | 0.527 | 0.405 | 0.69 | 0.919 | 0.738 | 1.14 |

## Discordant Patients

- **chb03**: AI=standard (AUC=0.529), BW=inverted (AUC=0.386), delta=0.143
- **chb08**: AI=inverted (AUC=0.430), BW=standard (AUC=0.500), delta=0.070
- **chb14**: AI=standard (AUC=0.544), BW=inverted (AUC=0.478), delta=0.067
- **chb20**: AI=standard (AUC=0.700), BW=inverted (AUC=0.367), delta=0.333
- **chb21**: AI=standard (AUC=0.529), BW=inverted (AUC=0.100), delta=0.429