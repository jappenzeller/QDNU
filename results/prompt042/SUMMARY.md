# PROMPT 042 - The ictal entropy split and the E/I proxy

**Generated:** 2026-09-07
**Script:** `scripts/prompt042_entropy_ei_proxy.py`
**Depends on:** `results/prompt037/cache/W8`, `results/prompt038/q1_scale_shape.csv`, `docs/EI_TWO_LEVEL_NOTES.md`

Conventions: 22 patients, CH8, W = 8 s windows identical to 037 (asserted), `lwf` covariances, $\rho = \Sigma/\operatorname{tr}\Sigma$, S = von Neumann entropy. Per patient dS = median ictal S minus median interictal S; sign *determined* if Mann-Whitney p < 0.05 and 2000-resample bootstrap stability >= 0.9. Aperiodic slope per window = linear fit of log10 PSD (Welch, 1 s segments, 8-channel mean) on log10 f, in two ranges: 30-45 Hz ("hi") and 2-45 Hz with 6-15 Hz excluded ("broad_nopeak"). d_slope = median ictal minus median interictal. Pre-declared: **P1** purifiers and mixers differ in d_slope (MWU p < 0.05, same direction in both ranges); **P2** Spearman(d_slope, dS) significant over all 22 in both ranges.

---

## Part A - the split, formalised

| group | n | patients |
|---|---|---|
| purifier (dS < 0, determined) | 4 | chb08, chb13, chb14, chb21 |
| mixer (dS > 0, determined) | 10 | chb01, 04, 06, 07, 10, 16, 17, 18, 19, 23 |
| undetermined | 8 | chb02, 03, 05, 09, 11, 15, 20, 22 |

The 2026-09-06 count of "7 purifiers" included chb02, chb09, chb11, whose drops are real in sign (stability 0.94-0.99) but miss the Mann-Whitney criterion (p = 0.05-0.09). Under the stricter rule the split is 4 / 10 / 8. Within-patient effect sizes are large where determined (|dS| 0.09-0.75 nats against a maximum of ln 8 = 2.08). dS vs 038's power change alpha: Spearman +0.30 - the purifiers include the two strongest power-down patients (chb14, chb21) but also chb08 (alpha +5.8) and chb13 (~0), so the split is not the power split.

## Part B - the E/I proxy

| range | median d_slope purifiers | median d_slope mixers | direction | MWU p | Spearman(d_slope, dS), n = 22 | Spearman(d_slope, alpha) |
|---|---|---|---|---|---|---|
| hi (30-45 Hz) | -0.04 | +0.24 | purifiers steeper | **1.00** | -0.00 (p = 1.0) | +0.11 |
| broad, no peak | -0.34 | +0.20 | purifiers steeper | **0.45** | +0.21 (p = 0.34) | +0.12 |

Slope flattens ictally (excitation-ward) in 12/22 patients, steepens in 10 - no cohort-level direction either.

**P1 FAIL. P2 FAIL.** The direction is the one the note would want (purifiers move inhibition-ward, mixers excitation-ward) in both ranges, but with 4 vs 10 patients and per-patient slope changes of +/-1 to +/-3 the comparison has no power, and the pooled correlation is zero.

## Reading

Two things are true at once. The entropy split is a solid within-patient observable: 14 of 22 patients have a determined, large, stable change in the mixedness of their 8-channel state during seizures, and its sign is patient-specific and independent of the power sign. And the aperiodic slope, as measured here, does not explain that sign. The slope estimates are noisy on 8 s windows (interictal "hi" slopes range from -0.8 to -5.1 across patients, which is not physiology), the groups are tiny, and a median-over-window design throws away the within-seizure time course, which is where a mechanism would actually show. So this is an underpowered null, not a refutation: the E/I two-level note's regime reading of the split is neither supported nor contradicted.

What would test it properly: a time-resolved design inside each seizure - slope and $S(\rho)$ tracked window by window from onset - asking whether the *trajectory* of the slope precedes or accompanies the trajectory of the entropy, per seizure, with the 037 per-seizure structure. That uses 142 seizures instead of 22 medians. It is the right next test if the E/I mechanism is to be kept; it is a day's work; it was not pre-declared here and is not run.

## Verdict

Part A: the entropy split is real and now formalised (4 purifiers, 10 mixers, 8 undetermined). Part B: the aperiodic-slope proxy does not separate the groups; test underpowered; mechanism status unchanged (open). Nothing in the theory note changes. The next test, if pursued, is within-seizure time-resolved.

## Deliverables

```
scripts/prompt042_entropy_ei_proxy.py
results/prompt042/per_patient.csv, tests.json, SUMMARY.md
```
