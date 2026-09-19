# PROMPT 036 - Mean phase lag Delta_ij as a polarity correlate

**Generated:** 2026-09-02T19:18:04

Conventions: CH8 montage, 0.5-40 Hz broadband, theta 4-8 / alpha 8-15 Hz,
30 s blocks per segment (PROMPT 019 loader), strict ictal vs interictal
(preictal excluded) for both the shift analysis and the LOSO check.

## Polarity-discriminating pairs (1.95s windows)

- Flagged (agreement >= 6/7 OR |r_pb| > 0.6): **8/56**
  - by sign agreement >= 6/7: 5 (chance expectation ~7.0/56, since P(>=6/7 best-orientation) = 0.125)
  - by |r_pb| > 0.6: 5 (chance expectation ~8.7/56 at n=7)
- Label-permutation test over all C(7,3)=35 polarity assignments:
  - n flagged-by-agreement: observed 5, p = 0.914 (floor 0.029)
  - mean |r_pb| across 56 tests: observed 0.331, p = 0.657

| band | pair | agreement | r_pb | p | r_circlin | delta by patient (chb01 chb03 chb05 chb07 chb11 chb14 chb21) |
|------|------|-----------|------|---|-----------|------------------|
| alpha | FP1-F7|F3-C3 | 5/7 | -0.694 | 0.084 | 0.723 | +0.20 +1.57 -0.12 -0.36 +0.75 +0.03 +0.04 |
| theta | FP1-F7|F8-T8 | 7/7 | +0.690 | 0.086 | 0.879 | +0.12 -2.95 +0.32 +0.69 -0.64 +0.06 -0.20 |
| theta | FP1-F7|F7-T7 | 6/7 | +0.674 | 0.097 | 0.799 | +0.16 -2.32 +0.17 +0.44 +0.12 +0.58 -0.58 |
| theta | FP1-F7|F3-C3 | 5/7 | +0.663 | 0.105 | 0.663 | -0.35 -0.62 -0.03 +0.14 -0.34 +0.29 -0.17 |
| alpha | FP1-F7|FP2-F8 | 5/7 | -0.632 | 0.128 | 0.633 | -0.13 +0.83 +0.24 -0.24 +0.45 +0.13 +0.01 |
| theta | FP1-F7|FP2-F8 | 6/7 | +0.467 | 0.291 | 0.445 | +0.30 -1.70 +0.24 -0.77 -0.16 +0.11 -0.10 |
| alpha | FP1-F3|FP2-F4 | 6/7 | -0.443 | 0.320 | 0.656 | -0.07 +0.16 -0.09 -0.22 +0.02 +0.22 +0.08 |
| theta | FP1-F3|FP2-F4 | 6/7 | -0.010 | 0.983 | 0.953 | +0.11 -0.07 +0.14 -0.74 -0.04 +0.20 -0.09 |

## Polarity-discriminating pairs (20s windows)

- Flagged (agreement >= 6/7 OR |r_pb| > 0.6): **12/56**
  - by sign agreement >= 6/7: 9 (chance expectation ~7.0/56, since P(>=6/7 best-orientation) = 0.125)
  - by |r_pb| > 0.6: 6 (chance expectation ~8.7/56 at n=7)
- Label-permutation test over all C(7,3)=35 polarity assignments:
  - n flagged-by-agreement: observed 9, p = 0.229 (floor 0.029)
  - mean |r_pb| across 56 tests: observed 0.355, p = 0.400

| band | pair | agreement | r_pb | p | r_circlin | delta by patient (chb01 chb03 chb05 chb07 chb11 chb14 chb21) |
|------|------|-----------|------|---|-----------|------------------|
| theta | FP1-F7|F8-T8 | 7/7 | +0.748 | 0.053 | 0.854 | +0.14 -3.04 +0.34 +0.84 -1.22 +0.07 -0.19 |
| alpha | FP1-F7|FP2-F8 | 5/7 | -0.661 | 0.106 | 0.675 | -0.06 +1.22 +0.10 -0.72 +0.55 +0.16 +0.04 |
| theta | FP1-F7|F7-T7 | 6/7 | +0.657 | 0.109 | 0.786 | +0.19 -2.13 -0.04 +0.30 -0.03 +0.61 -0.36 |
| alpha | FP1-F3|F8-T8 | 6/7 | -0.650 | 0.114 | 0.578 | -0.10 +2.44 -3.12 +0.65 +0.41 -1.89 +0.45 |
| theta | F3-C3|F8-T8 | 5/7 | +0.645 | 0.118 | 0.684 | +0.06 -0.40 +0.25 +0.60 -0.43 -0.01 +0.19 |
| theta | FP1-F7|F4-C4 | 5/7 | +0.634 | 0.126 | 0.726 | -0.10 -2.68 +0.36 +0.50 -0.65 +0.35 +0.17 |
| alpha | FP1-F7|FP2-F4 | 6/7 | +0.600 | 0.155 | 0.618 | +0.35 -1.64 +1.56 +0.39 +0.42 +0.10 -0.19 |
| alpha | F3-C3|F8-T8 | 6/7 | -0.558 | 0.193 | 0.578 | -0.09 +0.12 -0.46 -0.67 +0.18 +0.30 +0.14 |
| theta | FP1-F3|FP2-F4 | 6/7 | +0.527 | 0.224 | 0.833 | +0.17 -0.19 +0.15 -0.28 -0.03 +0.25 -0.17 |
| alpha | F8-T8|FP2-F4 | 6/7 | +0.489 | 0.265 | 0.709 | +0.40 -0.77 +3.02 +2.08 -0.55 -1.51 -0.15 |
| alpha | FP2-F8|FP2-F4 | 6/7 | +0.473 | 0.283 | 0.613 | +0.09 -0.44 +1.55 -0.14 -0.10 +0.01 -0.04 |
| alpha | FP1-F7|FP1-F3 | 6/7 | -0.048 | 0.918 | 0.906 | +0.24 -0.71 +0.62 +0.03 +1.97 +0.12 -0.27 |

## LOSO AUC (20 s windows, XGBoost, strict ictal vs interictal)

| feature set | n features | pooled AUC | mean per-patient AUC |
|-------------|-----------|------------|----------------------|
| b alone (PROMPT 019 reference) | 8 | 0.5083 | - |
| r_alone | 56 | 0.6741 | 0.6732 |
| delta_alone | 112 | 0.3429 | 0.3713 |
| r_plus_delta | 168 | 0.5606 | 0.6376 |

Per-patient AUC:

| patient | r_alone | delta_alone | r_plus_delta |
|---------|---------|-------------|--------------|
| chb01 | 1.0000 | 0.3000 | 0.8857 |
| chb03 | 0.0714 | 0.4857 | 0.1714 |
| chb05 | 0.9200 | 0.6800 | 0.8600 |
| chb07 | 0.6000 | 0.1333 | 0.3333 |
| chb11 | 0.6333 | 0.5000 | 0.9000 |
| chb14 | 0.7125 | 0.0250 | 0.5375 |
| chb21 | 0.7750 | 0.4750 | 0.7750 |

Increment of r+Delta over r alone (pooled): -0.1135

## Strong vs weak polar |delta_ij| (Step 5)

| window | band | strong mean (chb03, chb21) | weak mean (chb01/05/07/14) | MW p (greater) | p (two-sided) |
|--------|------|---------------------------|----------------------------|---------------|---------------|
| 1.95s | theta | 0.5592 | 0.4710 | 0.6000 | 1.0000 |
| 1.95s | alpha | 0.5547 | 0.5591 | 0.6000 | 1.0000 |
| 1.95s | both | 0.5569 | 0.5150 | 0.6000 | 1.0000 |
| 20s | theta | 0.5646 | 0.4678 | 0.6000 | 1.0000 |
| 20s | alpha | 0.6283 | 0.6287 | 0.6000 | 1.0000 |
| 20s | both | 0.5964 | 0.5482 | 0.6000 | 1.0000 |

(n=2 vs n=4: the one-sided Mann-Whitney p-value floor is 1/15 = 0.067; this comparison is directional evidence only.)

## Verdict

Mean phase lag Delta_ij carries no cross-patient seizure signal and no polarity signature, and this null lands in the first branch of the interpretation guide: report and move on. Delta-alone LOSO AUC is 0.343 - not merely chance but *below* it, meaning XGBoost finds phase-lag structure in the training patients whose sign does not transfer to the held-out patient; phase lag is patient-specific, and appending it to PLV magnitude actively degrades the r-alone baseline (0.674 -> 0.561, increment -0.113). The apparent polarity-discriminating pairs (12/56 at 20 s, 9 by sign agreement >= 6/7 vs ~7 expected by chance) do not survive the exhaustive label-permutation test (p = 0.229 for the flag count, p = 0.400 for mean |r_pb|; floor 0.029), and inspection shows the headline pair (theta FP1-F7|F8-T8, 7/7 agreement, r_pb = +0.75) is driven almost entirely by chb03's single -3.0 rad shift while the other two inverted patients sit near zero - a one-patient effect on one frontal channel (FP1-F7 appears in most flagged pairs), not an opposite-sign network signature. Strong-polar patients do not show larger shifts than weak-polar ones (mean |delta| 0.596 vs 0.548 at 20 s, Mann-Whitney p = 0.60). Conclusion: seizure-related mean phase-lag shifts exist within individual patients but are idiosyncratic in direction, so polarity has no phase-lag (propagation-direction) reading on this subset; direction 1 loses its main payoff as the Paper 3 spine, and polarity remains best explained as a projection artifact. Caveat: n = 7 patients and only 3-8 ictal / 10 interictal 20 s windows per patient - this rules out a large, consistent effect, not a subtle one.