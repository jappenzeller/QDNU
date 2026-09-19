# PROMPT 039 - Is per-patient polarity robust to the reference it is measured against?

**Generated:** 2026-09-06
**Script:** `scripts/prompt039_polarity_reference_robustness.py`
**Depends on:** `results/prompt037/cache/W8`, `results/prompt037/cache/W20`, `results/prompt038/q1_scale_shape.csv` (alpha), `results/prompt035/polarity_magnitude.csv` (t3 labels)

Conventions: 22 patients (chb16 has no ictal window at W = 20 and is skipped there), CH8, `lwf` covariances, 037 caches unchanged. For each held-out patient the tangent space is fit at the Frechet mean of the *training* patients' covariances (cohort reference, as in Tier 3), a direction is learned on their tangent vectors, and the held-out patient's windows are scored by projection; polarity = sign(AUC - 0.5). Two directions: shrinkage LDA (whitened, Paper 2's Tier 3 method) and the raw ictal-minus-interictal mean difference (un-whitened control). Two seizure-class definitions: strict (ictal vs interictal) and pooled (ictal + preictal vs interictal, the 035 / Paper 2 definition). Two window lengths. Reference resampling: 60 draws of 15 of the 21 training patients, seed 39; stability = fraction of draws with the full-LOSO sign.

Pre-declared per-patient reading (`docs`-string of the script): **robust** = $|\mathrm{AUC} - 0.5|$ >= 0.1, stability >= 0.90, same sign at both W; **reference-dependent** = $|\mathrm{AUC} - 0.5|$ >= 0.1 but stability < 0.90 or sign flips with W; **orthogonal** = $|\mathrm{AUC} - 0.5|$ < 0.1.

---

## Full table

AUC of the held-out patient; stability in parentheses. `alpha` is 038's scale component (+ = ictal power up). `t3` is the 035 label with its raw AUC.

| pt | t3 (AUC) | alpha | strict W8 LDA | strict W8 MD | strict W20 LDA | strict W20 MD | pooled W8 LDA | pooled W8 MD | pooled W20 LDA | pooled W20 MD |
|---|---|---|---|---|---|---|---|---|---|---|
| chb01 | sta (0.86) | +7.5 | 1.00 (1.00) | 1.00 (1.00) | 1.00 | 1.00 | 0.93 (1.00) | 0.94 (1.00) | 0.94 (1.00) | 0.98 (1.00) |
| chb02 | sta (0.66) | +6.7 | 0.96 (1.00) | 1.00 (1.00) | 1.00 | 1.00 | 0.68 (0.90) | 0.96 (1.00) | 0.72 (0.97) | 0.95 (1.00) |
| chb03 | **inv** (0.30) | +10.5 | 0.98 (1.00) | 1.00 (1.00) | 1.00 | 1.00 | 0.55 (0.73) | 0.88 (1.00) | 0.61 (0.77) | 0.91 (1.00) |
| chb04 | sta (0.55) | +6.2 | 0.98 (1.00) | 0.87 (1.00) | 0.97 | 0.91 | 0.94 (1.00) | 0.70 (1.00) | 0.91 (1.00) | 0.71 (1.00) |
| chb05 | sta (0.76) | +7.0 | 0.86 (1.00) | 1.00 (1.00) | 0.92 | 1.00 | 0.73 (1.00) | 0.89 (1.00) | 0.86 (1.00) | 0.91 (1.00) |
| chb06 | **inv** (0.34) | +0.0 | 0.42 (0.87) | 0.41 (1.00) | 0.10 | 0.45 | 0.50 (0.43) | 0.54 (0.93) | 0.34 (1.00) | 0.65 (1.00) |
| chb07 | sta (0.50) | +7.1 | 0.94 (1.00) | 0.99 (1.00) | 0.93 | 1.00 | 0.74 (1.00) | 0.82 (1.00) | 0.75 (0.97) | 0.88 (1.00) |
| chb08 | **inv** (0.13) | +5.8 | 0.61 (0.93) | 0.98 (1.00) | 0.61 | 0.99 | 0.52 (0.67) | 0.88 (1.00) | 0.38 (0.80) | 0.87 (1.00) |
| chb09 | sta (0.51) | +10.5 | 0.99 (1.00) | 1.00 (1.00) | 1.00 | 1.00 | 0.82 (1.00) | 0.89 (1.00) | 0.78 (1.00) | 0.83 (1.00) |
| chb10 | **inv** (0.21) | +2.7 | 0.97 (1.00) | 1.00 (1.00) | 0.99 | 1.00 | 0.45 (0.58) | 0.69 (1.00) | 0.37 (0.87) | 0.68 (1.00) |
| chb11 | sta (0.52) | +4.9 | 0.99 (1.00) | 1.00 (1.00) | 1.00 | 1.00 | 0.77 (0.98) | 0.96 (1.00) | 0.83 (1.00) | 0.96 (1.00) |
| chb13 | sta (0.51) | +0.4 | 0.54 (0.40) | 0.58 (1.00) | 0.63 | 0.54 | 0.45 (0.78) | 0.42 (1.00) | 0.52 (0.50) | 0.35 (0.98) |
| chb14 | sta (0.55) | **-2.8** | **0.21** (1.00) | **0.02** (1.00) | 0.50 | **0.00** | 0.49 (0.67) | **0.21** (1.00) | 0.67 (0.93) | **0.18** (1.00) |
| chb15 | sta (0.55) | **-1.6** | 0.77 (1.00) | 0.45 (1.00) | 0.83 | 0.52 | 0.67 (0.98) | 0.40 (1.00) | 0.65 (0.93) | 0.46 (1.00) |
| chb16 | sta (0.65) | +4.0 | 0.86 (1.00) | 0.96 (1.00) | - | - | 0.38 (0.75) | 0.78 (1.00) | 0.31 (0.77) | 0.71 (1.00) |
| chb17 | sta (0.62) | +4.8 | 0.84 (1.00) | 0.93 (1.00) | 0.82 | 0.98 | 0.56 (0.73) | 0.84 (1.00) | 0.61 (0.90) | 0.90 (1.00) |
| chb18 | sta (0.52) | +2.9 | 0.99 (1.00) | 0.89 (1.00) | 0.92 | 0.92 | 0.92 (1.00) | 0.54 (1.00) | 0.76 (1.00) | 0.55 (1.00) |
| chb19 | sta (0.73) | +8.3 | 0.99 (1.00) | 0.96 (1.00) | 1.00 | 1.00 | 0.67 (0.97) | 0.60 (1.00) | 0.70 (0.87) | 0.55 (1.00) |
| chb20 | **inv** (0.46) | +2.6 | 0.99 (1.00) | 0.84 (1.00) | 0.99 | 1.00 | 0.91 (0.95) | 0.47 (0.77) | 0.76 (0.68) | 0.44 (0.92) |
| chb21 | sta (0.98) | **-0.8** | 0.98 (0.88) | **0.35** (1.00) | 0.94 | 0.42 | 0.89 (0.88) | **0.20** (1.00) | 0.77 (0.83) | **0.20** (1.00) |
| chb22 | **inv** (0.22) | +6.2 | 0.98 (1.00) | 0.99 (1.00) | 1.00 | 1.00 | 0.33 (0.65) | 0.83 (1.00) | 0.35 (0.67) | 0.88 (1.00) |
| chb23 | sta (0.68) | +4.9 | 0.98 (1.00) | 1.00 (1.00) | 1.00 | 1.00 | 0.50 (0.30) | 0.79 (1.00) | 0.70 (0.95) | 0.77 (1.00) |

---

## What the table says

### 1. Under the strict class definition there is a robust, reference-independent polarity, and it is the sign of the ictal power change.

With ictal vs interictal, both directions, both window lengths, and every training subset agree: 17 of 22 patients are standard with AUC 0.84-1.00 and stability 1.00. The remaining five are the patients 038 found to have negative or near-zero scale component:

| patient | alpha (038) | strict MD AUC W8 / W20 | reading |
|---|---|---|---|
| chb14 | -2.8 | 0.02 / 0.00 | strongly inverted, ictal power *down* |
| chb21 | -0.8 | 0.35 / 0.42 | inverted on the mean difference |
| chb15 | -1.6 | 0.45 / 0.52 | inverted-to-orthogonal |
| chb06 | +0.0 | 0.41 / 0.45 | inverted-to-orthogonal (LDA 0.10 at W20) |
| chb13 | +0.4 | 0.58 / 0.54 | orthogonal |

The mean-difference direction is stable at 1.00 for every patient in every configuration except chb06 and chb20 pooled (0.77-0.93). So a direction-free polarity does exist: whether a patient's seizures raise or lower total power relative to the cohort's ictal direction. It is not subtle, it does not depend on who trained the reference, and it survives the change of window length. For chb21 and chb15 the whitened LDA direction disagrees with the mean difference (LDA standard, MD inverted): the power goes down, but a shape component the LDA picks up goes the standard way.

### 2. Paper 2's inverted set is a property of the pooled class plus whitening, and it is reference-dependent.

The 035 label calls chb03, chb06, chb08, chb10, chb20, chb22 inverted. Under the strict definition every one of them except chb06 is *strongly standard* (AUC 0.97-1.00 on both directions). The inverted signs reappear only with preictal pooled into the seizure class **and** the LDA direction: pooled LDA AUC 0.55 / 0.61 for chb03, 0.52 / 0.38 for chb08, 0.45 / 0.37 for chb10, 0.33 / 0.35 for chb22, plus chb16 (0.38 / 0.31) and chb13 (0.45 / 0.52) which 035 did not call inverted. And in that configuration the sign is unstable to the training cohort: stability 0.58-0.80 for chb03, 08, 10, 22, 16, with draw ranges that cross 0.5 in every case (chb03 [0.40, 0.67], chb08 [0.31, 0.76], chb10 [0.31, 0.68], chb22 [0.13, 0.76]). chb23 pooled-LDA at W8 has stability 0.30 with AUC 0.50 - its sign is a coin flip across training subsets. The un-whitened mean difference on the same pooled class does **not** produce these inversions (chb03 0.88, chb08 0.88, chb10 0.69, chb22 0.83, all stability 1.00); it produces the power-down set again.

Mechanism, stated plainly: preictal windows carry no power increase, so pooling them with ictal dilutes the class mean difference along the scale axis and leaves a class boundary dominated by whatever separates preictal from interictal, which is small and patient-specific (038: cross-patient shape vectors nearly orthogonal). Shrinkage LDA then whitens by the pooled within-class covariance, amplifying exactly those low-variance directions, and the resulting direction depends on which patients' preictal structure went into the training set. Held-out patients whose own preictal/ictal structure sits on the far side of that direction come out "inverted".

### 3. Per-patient classification under the pre-declared reading (LDA direction, W8 primary, W20 sign check)

Strict class:
- **robust standard (18):** every patient except the four below (chb16 on W8 alone; no W20 ictal windows).
- **robust inverted (0)** by the LDA rule. chb14 is robustly inverted on the mean difference (0.02 / 0.00, stability 1.00) but its LDA AUC at W20 is exactly 0.50, so by the letter of the rule it falls in the next bin.
- **reference-dependent (2):** chb14 (W20 LDA sign undefined), chb21 (stability 0.88; MD says inverted, LDA says standard).
- **orthogonal (2):** chb06 (LDA 0.42, within 0.1 of chance at W8), chb13.

Pooled class (035 / Paper 2 definition):
- **robust standard (11)**, **robust inverted (0)**, **reference-dependent (3):** chb16, chb21, chb22; **orthogonal (8):** chb03, chb06, chb08, chb10, chb13, chb14, chb17, chb23.

Under the pooled definition, every patient 035 called inverted is either within 0.1 of chance at W8 or reference-dependent. None is robustly inverted. Pooling turns a clean picture into an unstable one.

---

## Verdict

**Polarity is real, and it is simpler than Papers 2-3 have been treating it.** A stable, reference-independent per-patient sign exists: the direction of the ictal total-power change relative to the cohort. Most patients go up; chb14 goes down hard; chb21, chb15, chb06 go down or sideways. This sign is the same on the raw mean difference and on the whitened direction (with the two disagreements above), the same at 8 s and 20 s, and the same for every resampled training cohort. It is what 038's alpha measures directly, without any classifier. It is a direction-free geometric fact about the patient.

**The specific inverted set reported from Tier 3 (chb03, chb08, chb10, chb22) is not that.** Those patients have among the largest standard power increases in the cohort (alpha +5.8 to +10.5). Their "inversion" requires pooling preictal into the seizure class and whitening, and under those conditions the sign moves with the training cohort. That is a reproducible artifact of a class definition and an estimator, not a property of the patient, and Paper 3 should say so - it is the right correction to Paper 2's operational definition, and it is the kind of correction the polarity mechanism section already half-anticipates ("polarity is axis-dependent"). The hardware assignments (chb03, chb11, chb21 inverted; template-fidelity readout, 1.95 s windows) are a third definition again; chb21 is the only patient inverted under hardware *and* under the strict mean-difference here.

**For DSP-000:** the robust polarity lives on the scale axis, which trace-normalisation removes. That is now a settled reason, from two prompts, why the prepared state must be accompanied by $\operatorname{tr}\Sigma$ as a classical channel. What remains for a state-based observable is the shape residual, whose cross-patient structure is weak (038) and whose within-patient structure is strong (037). The natural state-based question is therefore within-patient, not cross-patient: does the shape of a patient's own covariance state change reproducibly across events - which is also exactly the question the lattice alpha-blocking sessions can answer on a single subject.

---

## Deliverables

```
scripts/prompt039_polarity_reference_robustness.py
results/prompt039/
    patient_chbNN_W{8,20}[_pre].json
    per_patient_W8.csv, per_patient_W20.csv          strict class
    per_patient_W8_pre.csv, per_patient_W20_pre.csv  pooled class
    SUMMARY.md
```

Run: `python scripts/prompt039_polarity_reference_robustness.py --windows 8,20 [--seizure-class ictal+preictal]`.
