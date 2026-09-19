# PROMPT 037 - Per-seizure decomposition of the geometric polarity measures

**Generated:** 2026-09-03
**Script:** `scripts/prompt037_per_seizure_decomposition.py` (+ `scripts/prompt037_step0_diagnostics.py`)
**Depends on:** `results/prompt035/` (geometric addendum), `results/prompt036/`

Conventions: 22 CHB-MIT patients (chb12, chb24 excluded), CH8 LOSO montage,
preprocess_eeg (demean, 0.5-128 Hz, 60 Hz notch), Covariances(estimator='lwf'),
TangentSpace(metric='riemann') fit on all of the patient's covariances
(reference = patient Frechet mean), d = 36. **New loader:** every segment is
tiled into consecutive non-overlapping W-second windows (W = 8 s primary,
W = 20 s confirmatory), each window tagged (patient, seizure_id, phase).
Interictal capped at 10 windows per patient, sampled with seed 37 from the 30
available. Ictal-only is the primary arm; preictal-only is the control arm.
Permutations: 2000, seed 37. Fixed-count bootstrap: 200 draws.

This was declared a disconfirmation run: Step 0 offered two closed-form
mechanisms for the addendum's flagged correlations, and the pre-declared
expectation was Branch A (cancellation). **That is not where it landed.**

---

## Step 0 - diagnostics reproduced

All six anchored values from the prompt reproduce within 0.0015
(`step0_diagnostics.csv`, exit 0):

| test | rho | p |
|---|---|---|
| s_norm vs n_seizures | -0.581 | 0.0046 |
| $s_{\rm norm}\cdot\sqrt{K}$ vs n_seizures | +0.108 | 0.633 |
| ictal_retention (035 loader) vs mean_seizure_duration | +0.723 | 0.0001 |
| partial: bw_dist vs mean_dur \| n_seizures | +0.400 | 0.072 |
| partial: s_norm vs n_seizures \| mean_dur | -0.467 | 0.033 |
| n_seizures vs mean_seizure_duration | -0.558 | 0.007 |

## Unit tests

1. **Decomposition identity** $K^{2}\,\|\bar s\|^{2} = \sum_k \|s^{(k)}\|^{2} + \sum_{j \neq k} \langle s^{(j)}, s^{(k)}\rangle$: max relative residual 3.0e-16 over 22 patients (W = 8), 3.5e-16 (W = 20). PASS.
2. **Legacy reproduction** (`--stage legacy`): the new loader in one-window-per-segment mode reproduces `results/prompt035/geometric_measures.csv` s_norm and bw_dist to the CSV's 6-decimal resolution for **22/22** patients (`legacy_reproduction.csv`). The new loader is a strict superset of the addendum's. PASS.
3. **Synthetic recovery** (`synthetic_recovery.csv`, 4000 draws, sigma = 1, d = 36, n_int = 10):

   | K | n_k | L | true c | Q_corr | P_corr/Q_corr | naive mean-cosine |
   |---|---|---|---|---|---|---|
   | 7 | 3 | 3 | 0.00 | 8.98 (9.0) | -0.002 (-0.02) | +0.142 (+0.14) |
   | 7 | 3 | 3 | 0.29 | 8.99 (9.0) | +0.289 (+0.28) | +0.249 (+0.25) |
   | 7 | 3 | 5 | 0.00 | 25.00 (25.0) | -0.001 (-0.01) | +0.086 (+0.09) |
   | 4 | 2 | 3 | 0.00 | 9.02 (9.0) | -0.001 (-0.01) | +0.115 (+0.12) |
   | 7 | 8 | 3 | 0.29 | 8.95 (9.0) | +0.287 (+0.28) | +0.356 (+0.36) |

   All within +/-0.03 of the prompt's expectations. The naive estimator invents +0.09 to +0.14 of coherence at c = 0 and its bias flips sign with n_k (understates at n_k = 3, overstates at n_k = 8), exactly as predicted. Note: $P_{\rm corr}/Q_{\rm corr}$ is evaluated as the ratio of draw-averaged P and Q; the mean of per-draw ratios is Jensen-biased (-0.114 at K = 4, n_k = 2) and is reported alongside in the CSV. PASS.
4. **Retention** at W = 8 s: 21/22 patients at 1.00. **chb16 is 0.70** (3 of its 10 seizures are 6-7 s and cannot supply an 8 s window). The prompt's statement that 8 s "retains chb16 (max seizure 14 s)" looked at the max, not the min. Handled by reporting the primary tests with and without chb16 (below) rather than by dropping to W = 6. FAIL on the letter of the test; no conclusion below changes with chb16 excluded.

---

## Window inventory (W = 8 s)

| patient | K | ictal win | preictal win | interictal kept/avail | ictal win/seizure min / med / max | retention |
|---|---|---|---|---|---|---|
| chb01 | 7 | 53 | 49 | 10/30 | 3 / 6 / 12 | 1.00 |
| chb02 | 3 | 21 | 21 | 10/30 | 1 / 10 / 10 | 1.00 |
| chb03 | 7 | 47 | 49 | 10/30 | 5 / 6 / 8 | 1.00 |
| chb04 | 4 | 45 | 28 | 10/30 | 6 / 12.5 / 14 | 1.00 |
| chb05 | 5 | 68 | 35 | 10/30 | 12 / 14 / 15 | 1.00 |
| chb06 | 10 | 14 | 70 | 10/30 | 1 / 1 / 2 | 1.00 |
| chb07 | 3 | 39 | 21 | 10/30 | 10 / 12 / 17 | 1.00 |
| chb08 | 5 | 113 | 35 | 10/30 | 16 / 21 / 33 | 1.00 |
| chb09 | 4 | 32 | 28 | 10/30 | 7 / 8 / 9 | 1.00 |
| chb10 | 7 | 53 | 49 | 10/30 | 4 / 8 / 11 | 1.00 |
| chb11 | 3 | 100 | 21 | 10/30 | 2 / 4 / 94 | 1.00 |
| chb13 | 12 | 62 | 84 | 10/30 | 2 / 6 / 8 | 1.00 |
| chb14 | 8 | 17 | 56 | 10/30 | 1 / 2 / 5 | 1.00 |
| chb15 | 20 | 237 | 140 | 10/30 | 3 / 10.5 / 25 | 1.00 |
| chb16 | 10 | 7 | 70 | 10/30 | 0 / 1 / 1 | **0.70** |
| chb17 | 3 | 36 | 21 | 10/30 | 11 / 11 / 14 | 1.00 |
| chb18 | 6 | 36 | 42 | 10/30 | 3 / 6 / 8 | 1.00 |
| chb19 | 3 | 28 | 21 | 10/30 | 9 / 9 / 10 | 1.00 |
| chb20 | 8 | 32 | 56 | 10/30 | 3 / 4 / 6 | 1.00 |
| chb21 | 4 | 24 | 28 | 10/30 | 1 / 6.5 / 10 | 1.00 |
| chb22 | 3 | 25 | 21 | 10/30 | 7 / 9 / 9 | 1.00 |
| chb23 | 7 | 49 | 49 | 10/30 | 2 / 7 / 14 | 1.00 |

Compared with the addendum (at most one ictal window per seizure, zero for
chb16, two for chb06), every patient now has ictal representation for every
seizure of >= 8 s. Diagnostic 2's dropout is removed.

---

## Step 2 - per-patient decomposition (W = 8 s, ictal arm)

$c_{\rm corr} = P_{\rm corr}/Q_{\rm corr}$ is the bias-corrected coherence; $\|\bar s\|^{2}/Q_{\rm corr}$
is the observed averaging ratio, to be compared with `1/K` (the pure
cancellation prediction) and `1` (perfectly aligned seizures). Noise source:
per-seizure within-covariance where n_k >= 2, pooled across the patient's
multi-window seizures otherwise; chb16 (every seizure 1 window) falls back to
the total within-ictal variance, which over-corrects Q for that patient.

| patient | K | Q_corr | L_corr | c (raw) | c_corr | naive cos | $\|\bar s\|^{2}/Q_{\rm corr}$ | 1/K | D_bar | d_pooled | BW shrinkage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chb01 | 7 | 62.3 | 7.90 | +0.98 | +0.98 | +0.98 | 0.99 | 0.14 | 234 | 233 | 0.99 |
| chb02 | 3 | 46.3 | 6.81 | +0.95 | +1.01 | +0.97 | 1.03 | 0.33 | 189 | 211 | 1.12 |
| chb03 | 7 | 117.9 | 10.86 | +0.96 | +0.96 | +0.97 | 0.97 | 0.14 | 284 | 266 | 0.94 |
| chb04 | 4 | 48.5 | 6.96 | +0.85 | +0.87 | +0.90 | 0.96 | 0.25 | 197 | 197 | 1.00 |
| chb05 | 5 | 51.4 | 7.17 | +0.97 | +0.98 | +0.98 | 1.00 | 0.20 | 468 | 457 | 0.98 |
| chb06 | 10 | 2.9 | 1.70 | +0.48 | +0.84 | +0.51 | 1.16 | 0.10 | 80 | 61 | 0.76 |
| chb07 | 3 | 61.0 | 7.81 | +0.94 | +0.95 | +0.95 | 0.98 | 0.33 | 362 | 349 | 0.96 |
| chb08 | 5 | 37.1 | 6.09 | +0.98 | +0.99 | +0.99 | 1.00 | 0.20 | 253 | 246 | 0.97 |
| chb09 | 4 | 111.6 | 10.56 | +1.00 | +1.00 | +1.00 | 1.00 | 0.25 | 630 | 625 | 0.99 |
| chb10 | 7 | 14.2 | 3.77 | +0.81 | +0.84 | +0.85 | 0.88 | 0.14 | 187 | 183 | 0.98 |
| chb11 | 3 | 30.5 | 5.52 | +0.93 | +0.94 | +0.95 | 0.98 | 0.33 | 143 | 186 | 1.30 |
| chb13 | 12 | 10.7 | 3.27 | +0.46 | +0.47 | +0.50 | 0.55 | 0.08 | 111 | 86 | 0.77 |
| chb14 | 8 | 13.2 | 3.64 | +0.68 | +0.90 | +0.70 | 0.98 | 0.13 | 99 | 88 | 0.89 |
| chb15 | 20 | 4.2 | 2.05 | +0.47 | +0.09 | +0.46 | 0.96 | 0.05 | 149 | 144 | 0.97 |
| chb16 | 7/10 | 17.8 | 4.22 | +0.78 | +1.00 | +0.77 | 1.07 | 0.14 | 106 | 88 | 0.83 |
| chb17 | 3 | 27.0 | 5.20 | +0.94 | +0.96 | +0.95 | 1.03 | 0.33 | 107 | 99 | 0.92 |
| chb18 | 6 | 28.6 | 5.35 | +0.50 | +0.51 | +0.52 | 0.62 | 0.17 | 114 | 85 | 0.74 |
| chb19 | 3 | 73.1 | 8.55 | +0.99 | +1.01 | +0.99 | 1.03 | 0.33 | 317 | 311 | 0.98 |
| chb20 | 8 | 8.5 | 2.92 | +0.84 | +1.06 | +0.85 | 1.12 | 0.13 | 73 | 66 | 0.91 |
| chb21 | 4 | 5.8 | 2.41 | +0.86 | +1.05 | +0.86 | 1.17 | 0.25 | 91 | 84 | 0.92 |
| chb22 | 3 | 43.7 | 6.61 | +0.99 | +1.00 | +0.99 | 1.03 | 0.33 | 195 | 191 | 0.98 |
| chb23 | 7 | 27.1 | 5.21 | +0.97 | +0.99 | +0.98 | 1.01 | 0.14 | 143 | 154 | 1.08 |

No patient has $Q_{\rm corr} \le 0$; none is excluded from the c tests. Values of
c_corr slightly above 1 (chb02, 19, 20, 21) are the corrected-Q denominator
being over-shrunk at small n_k; they are consistent with c = 1 within noise.

**Reading.** The cancellation mechanism is not operating. $\|\bar s\|^{2}/Q_{\rm corr}$
sits at 0.9-1.0 for 19 of 22 patients, not at 1/K; the exceptions (chb13
0.55, chb18 0.62, chb15 0.96 with c_corr 0.09 - chb15's shift is so small that
the coherence estimate is unstable) are still far above 1/K. Within a patient
the per-seizure shift vectors point in essentially the same tangent direction,
so the mean shift vector loses almost nothing to averaging and $\|\bar s\|$
tracks $\bar L$ closely. The BW shrinkage factor tells the same story
(0.74-1.30, median 0.97). Diagnostic 1's $\sqrt{K}$ annihilation was a
coincidence: rescaling by $\sqrt{K}$ cancels *any* $\|s\|$ vs `K` relation with
a log-log slope near -0.5, whether the slope comes from averaging or from a
genuine dependence of per-seizure magnitude on `K`. Step 0's Diagnostic 1 was
consistent with cancellation but was never able to distinguish it from a real
effect; the decomposition here does, and it is real.

---

## Step 3 - permutation of window -> seizure assignment (`permutation_coherence.csv`)

Null: shuffle the patient's ictal windows among its seizures holding each
n_k fixed, recompute c_corr. chb16 is excluded (every seizure has one window,
nothing to shuffle). 21 patients tested.

Null medians sit at $c_{\rm corr} \approx 1$.000 with IQR 0.002-0.20 (chb06 and chb15 have
degenerate nulls because Q_corr under shuffling is near zero; their p-values
are rank-based and remain valid, but their null spread is meaningless).
Observed c_corr is **below** the null median in 18/21 patients; two-sided
p < 0.05 in 9 (chb01, 03, 04, 07, 08, 10, 13, 17, 18). Cohort Stouffer
z = -7.04 (p < 1e-11); sign test 3/21 above null median, p = 0.001.

**Reading.** Seizures within a patient are more heterogeneous than
interchangeable draws from a common pool - but only slightly: the departures
are c_corr 0.84-0.98 against a null of 1.00 for most patients, and the three
patients with real structure (chb13, chb18, and to a lesser extent chb10) are
the ones whose averaging ratio fell below 1 in Step 2. **Caveat that must
travel with this result:** windows within one seizure are temporally
contiguous and autocorrelated; the shuffled null treats them as exchangeable,
so it is rejected by within-segment autocorrelation alone, independent of
between-seizure physiology. The preictal control arm rejects the same null
even more strongly (Stouffer z = -13.6, 1/21 above null) with no phenotype
signal at all, which shows the permutation is detecting segment structure,
not seizure identity. Step 3 therefore establishes only that per-seizure
vectors are *not* pure noise around a common direction; it does not license
event-level claims about individual seizures.

---

## Step 4 - cross-patient primary tests (n = 22, Holm over 4)

| id | test | rho | p | Holm p | partial \| other phenotype var | partial \| ictal retention |
|---|---|---|---|---|---|---|
| **P1** | L_corr vs n_seizures | **-0.566** | 0.0060 | **0.018** | -0.421 \| mean_dur (p = 0.057) | -0.559 (p = 0.008) |
| P2 | c_corr vs n_seizures | -0.379 | 0.082 | 0.082 | -0.531 \| mean_dur (p = 0.013) | -0.472 (p = 0.031) |
| **P3** | D_bar vs mean_seizure_duration | **+0.656** | 0.0009 | **0.0037** | +0.549 \| n_seizures (p = 0.010) | +0.632 (p = 0.002) |
| P4 | L_corr vs mean_seizure_duration | +0.458 | 0.032 | 0.064 | +0.207 \| n_seizures (p = 0.367) | +0.448 (p = 0.042) |

Sensitivity, chb16 excluded (n = 21, `sensitivity/primary_tests_excl_chb16.csv`):
P1 rho = -0.532 (Holm 0.039), P2 -0.464 (Holm 0.069), P3 +0.618 (Holm 0.011),
P4 +0.432 (Holm 0.069). Same pattern.

Preictal control arm (`primary_tests_preictal.csv`): P1 -0.230 (p = 0.30),
P2 +0.094 (p = 0.68), P3 +0.106 (p = 0.64), P4 +0.006 (p = 0.98). **All four
are null when the same machinery is run on the preictal minute.** Whatever the
ictal arm is measuring is ictal physiology, not a property of the loader, the
tangent reference, or the patient's recording.

W = 20 s confirmatory, 17 patients with >= 80 % ictal retention at 20 s
(chb02, 06, 14, 16, 21 excluded; `w20/`): P1 rho = -0.505 (p = 0.039, Holm
0.15; partial | mean_dur = -0.504, p = 0.046); P2 -0.389 (p = 0.12);
P3 +0.409 (p = 0.10; partial | K = +0.19); P4 +0.169 (p = 0.52). P1 replicates
in magnitude and its duration-partial strengthens; P3 weakens because the
exclusion removes exactly the short-seizure patients that anchor the low end
of the duration axis (range restriction: remaining mean durations are
37-269 s vs 8-269 s in the full cohort). Neither survives Holm at n = 17.

Exploratory table (`exploratory_correlations.csv`): 48 tests, Bonferroni
alpha = 0.00104, 3 survivors, all the same duration axis: D_bar vs mean_dur
(+0.656), d_pooled vs mean_dur (+0.668), D_bar vs median_dur (+0.665).
Nothing else is promotable. seizure_density_per_hour (which is n_seizures /
recording hours) tracks the P1 axis at rho -0.49 to -0.53.

---

## Step 5 - within-patient duration test (`within_patient_duration.csv`, `within_patient_combined.json`)

16 patients with K >= 4, 121 seizures. Per-patient Spearman of seizure
duration against d^(k):

| patient | K | $\rho$ ($d_k$) | p | $\rho$ ($\|s^{(k)}\|^{2}_{\rm corr}$) | p |
|---|---|---|---|---|---|
| chb01 | 7 | +0.34 | 0.45 | +0.34 | 0.45 |
| chb03 | 7 | +0.16 | 0.73 | -0.05 | 0.91 |
| chb04 | 4 | +0.40 | 0.60 | +0.40 | 0.60 |
| chb05 | 5 | -0.80 | 0.10 | -0.60 | 0.28 |
| chb06 | 10 | **-0.88** | 0.0009 | -0.68 | 0.030 |
| chb08 | 5 | -0.60 | 0.28 | -0.60 | 0.28 |
| chb09 | 4 | -0.20 | 0.80 | -0.40 | 0.60 |
| chb10 | 7 | **+0.93** | 0.0025 | +0.93 | 0.0025 |
| chb13 | 12 | +0.17 | 0.59 | +0.01 | 0.97 |
| chb14 | 8 | +0.70 | 0.052 | +0.90 | 0.0025 |
| chb15 | 20 | -0.37 | 0.11 | -0.26 | 0.28 |
| chb16 | 7 | -0.46 | 0.30 | -0.69 | 0.083 |
| chb18 | 6 | +0.06 | 0.91 | -0.81 | 0.050 |
| chb20 | 8 | -0.01 | 0.98 | +0.11 | 0.80 |
| chb21 | 4 | -0.40 | 0.60 | +0.80 | 0.20 |
| chb23 | 7 | +0.64 | 0.12 | +0.75 | 0.052 |

Combined:

| estimator | result |
|---|---|
| Stouffer z (weights $\sqrt{K-1}$), duration vs $d^{(k)}$ | z = -0.40, p = 0.69; mean per-patient rho = -0.02 |
| Stouffer z, duration vs \|\|s^(k)\|\|^2_corr | z = +0.17, p = 0.86 |
| Mixed model $\log d \sim \log(\text{dur}) + (1 \mid \text{patient})$ | slope +0.121 +/- 0.043, p = 0.005 (random slope: +0.201, p = 0.007) |
| Same model, within/between split (Mundlak) | **within** +0.096 +/- 0.044, p = 0.027; **between** +0.506 +/- 0.15, p = 0.0009 |
| Fixed-count control (1 window per seizure, 200 draws) | mean Stouffer z = -0.03 (SD 0.83), mean rho = +0.03 |
| W = 20 replication (17 patients) | Stouffer z = +1.13 (p = 0.26); within slope +0.143 (p = 0.004); fixed-count z = +1.36 |

**Reading.** The naive random-intercept slope (+0.12, p = 0.005) is *not* a
within-patient result - the Mundlak split shows it is a blend dominated by
the between-patient slope (+0.51), which is P3 restated at the seizure level.
The genuine within-patient component is +0.10 on the log-log scale (a
doubling of seizure duration moves d^(k) by ~7 %), nominally significant in
the mixed model but invisible to the rank-based Stouffer combination and to
the fixed-count control. Per-patient signs are split (chb10 +0.93 vs chb06
-0.88, both p < 0.003). The pre-declared Branch D criterion - combined
p < 0.05 **with** the fixed-count control holding - is not met. The design was
declared underpowered below rho ~ 0.25 and the point estimate is at or below
that bound; the honest statement is that any within-patient duration effect
on ictal displacement is small (|rho| <~ 0.2), not that it is absent.

---

## Verdict

**Branch B - genuine magnitude phenotype - with the duration axis (P3) also
surviving cross-patient, and Branches A, C and D not met.**

What is retired:

- **The cancellation explanation (Branch A).** The addendum's $\|s_p\|$ vs
  `n_seizures` correlation is not a $1/\sqrt{K}$ averaging artifact: per-seizure
  shift vectors within a patient are coherent (c_corr 0.84-1.00 for 19/22),
  $\|\bar s\|^{2}/Q_{\rm corr}$ sits near 1 rather than 1/K, and the correlation is
  reproduced on the K-independent per-seizure magnitude $L_{\rm corr}$ (rho -0.566,
  Holm p = 0.018). Step 0's Diagnostic 1 is retired as a diagnostic: $\sqrt{K}$
  rescaling cannot separate averaging from a real L-K dependence.
- **The ictal-dropout explanation of the duration correlation (Diagnostic 2)
  as the whole story.** Fixing the dropout did not remove the correlation;
  it strengthened it (bw_dist vs mean_dur +0.506 -> D_bar vs mean_dur +0.656,
  partial | K = +0.549). The dropout was real and did distort the addendum's
  values for chb06/chb14/chb16/chb02/chb21, but the corrected values keep
  those patients at the low end.
- **The coherence phenotype (Branch C).** P2 does not survive Holm (0.082).
  Lower coherence in the many-seizure patients (chb13, chb15, chb18) is an
  exploratory observation only.
- **The within-patient duration effect (Branch D).** Not supported at the
  pre-declared standard; at most a small effect (within slope +0.10 log-log).

What is promoted (to Paper 3):

- **Per-seizure ictal displacement magnitude is smaller in patients with more
  seizures** (P1, rho = -0.566, Holm 0.018; W = 20 replication -0.505; preictal
  control null). The count/duration partial is at the margin (-0.42, p = 0.057
  at W = 8; -0.50, p = 0.046 at W = 20) because n_seizures and mean duration
  are collinear in CHB-MIT (rho -0.558), so the two phenotype variables cannot
  be fully separated at n = 22. State it as a phenotype axis, not two
  independent effects.
- **Patients with longer seizures show larger ictal-interictal BW
  displacement** (P3, rho = +0.656, Holm 0.0037, partial | K = +0.549, 3/3
  Bonferroni survivors in the exploratory table are this axis). This is a
  between-patient property: within a patient, seizure duration does not
  predict displacement (Step 5).
- **Within-patient seizures are geometrically stereotyped.** $c_{\rm corr} \approx 1$ is
  the strongest single finding here and was not on the pre-declared list. It
  is what makes the per-patient polarity sign well defined in the first
  place, and it is the reason the addendum's mean-vector measure worked
  despite being the "wrong" estimator: with coherent seizures, $\|\bar s\|$
  and $\bar L$ coincide. Methodological note for Paper 3 still stands in a
  revised form - a mean-shift-vector magnitude is only interpretable once
  per-seizure coherence has been measured; here it happens to be ~1.

Caveats carried forward: the Step 3 permutation is confounded by
within-segment autocorrelation (see Step 3); chb16 retention is 0.70 at
W = 8 s (conclusions unchanged with chb16 excluded); the W = 20 confirmatory
subset is range-restricted on duration; short-seizure patients contribute 1-2
windows per seizure, so their $Q_{\rm corr}$ corrections lean on pooled noise
estimates and their $c_{\rm corr}$ values are the least precise in the table.

---

## Addendum (2026-09-07) - shape-only coherence

038 showed that a median 83 % of each patient's ictal shift lies along the pure-scaling (identity) direction. Since that direction is the same for every seizure, the full-vector $c_{\rm corr}$ above could in principle be high merely because power rises in every seizure. The same estimator was therefore re-run on the shape component alone (identity direction projected out of every tangent vector before `decompose`), W = 8, same cache.

| patient | K | $c_{\rm corr}$ full | $c_{\rm corr}$ shape | $\|\bar s\|^{2}/Q_{\rm corr}$ (shape) | 1/K |
|---|---|---|---|---|---|
| chb01 | 7 | +0.98 | +0.84 | 0.92 | 0.14 |
| chb02 | 3 | +1.01 | +1.06 | 1.33 | 0.33 |
| chb03 | 7 | +0.96 | +0.44 | 0.58 | 0.14 |
| chb04 | 4 | +0.87 | +0.78 | 0.96 | 0.25 |
| chb05 | 5 | +0.98 | +0.71 | 0.87 | 0.20 |
| chb06 | 10 | +0.84 | +1.06 | 1.19 | 0.10 |
| chb07 | 3 | +0.95 | +0.82 | 0.92 | 0.33 |
| chb08 | 5 | +0.99 | +0.93 | 0.99 | 0.20 |
| chb09 | 4 | +1.00 | +0.91 | 1.09 | 0.25 |
| chb10 | 7 | +0.84 | +0.88 | 0.94 | 0.14 |
| chb11 | 3 | +0.94 | +0.81 | 0.96 | 0.33 |
| chb13 | 12 | +0.47 | +0.67 | 0.72 | 0.08 |
| chb14 | 8 | +0.90 | +0.76 | 0.88 | 0.13 |
| chb15 | 20 | +0.09 | +0.20 | 0.55 | 0.05 |
| chb16 | 7 | +1.00 | +1.00 | 1.28 | 0.14 |
| chb17 | 3 | +0.96 | +0.78 | 0.98 | 0.33 |
| chb18 | 6 | +0.51 | +0.74 | 0.82 | 0.17 |
| chb19 | 3 | +1.01 | +0.97 | 1.06 | 0.33 |
| chb20 | 8 | +1.06 | +0.85 | 0.96 | 0.13 |
| chb21 | 4 | +1.05 | +1.01 | 1.09 | 0.25 |
| chb22 | 3 | +1.00 | +0.97 | 1.06 | 0.33 |
| chb23 | 7 | +0.99 | +0.93 | 1.01 | 0.14 |

Shape-only $c_{\rm corr}$: median +0.84; > 0.5 in 20/22, > 0.8 in 14/22, <= 0 in 0/22. The averaging ratio on the shape component stays at 0.55-1.33 against a 1/K floor of 0.05-0.33. **The coherence is not a power artifact: within a patient, seizures share a spatial pattern of covariance change, not only a direction of power change.** The patients whose coherence drops most from full to shape (chb03 0.96 -> 0.44, chb05 0.98 -> 0.71) are the ones where part of the full-vector coherence was the shared power rise; chb15 (twenty short seizures, small shifts) remains the one patient without measurable event-level structure.

This is the strong form of the "geometrically stereotyped" finding in the Verdict above and is the version to state in Paper 3.

---

## Deliverables

```
scripts/prompt037_per_seizure_decomposition.py
scripts/prompt037_step0_diagnostics.py
results/prompt037/
    step0_diagnostics.csv
    synthetic_recovery.csv               unit test 3
    legacy_reproduction.csv              unit test 2
    window_inventory.csv
    per_seizure_shifts.csv
    patient_decomposition.csv
    permutation_coherence.csv
    primary_tests.csv
    exploratory_correlations.csv
    within_patient_duration.csv
    within_patient_combined.json
    run_meta.json
    *_preictal.csv                       preictal control arm
    sensitivity/*_excl_chb16.csv         W = 8, chb16 dropped
    w20/*_w20.*                          W = 20 confirmatory, 17 patients
    scatters/P1..P4_*.png, within_patient_duration.png
    cache/W8/chbNN.npz, cache/W20/chbNN.npz
    SUMMARY.md
```

Run: `python scripts/prompt037_per_seizure_decomposition.py --stage {test,legacy,cache,analyze,all}`
(`--data-root` or `QDNU_DATA_ROOT` override the CHB-MIT path; `--window`,
`--arm`, `--exclude`, `--tag`, `--out` reproduce the secondary runs).
