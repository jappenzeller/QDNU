# PROMPT 035 — Polarity magnitude vs clinical severity (CHB-MIT)

Generated: 2026-08-02T09:21:28.502958

Cohort: 22 patients from CHB-MIT (chb12 and chb24 excluded); 20-s windows; Tier 2A eigenvalues + Tier 3 Riemannian. Source: `results\full_cohort\full_cohort_polarity.json`.

## Findings paragraph

Across 16 Spearman-correlation tests pairing polarity magnitude ($|\mathrm{AUC}-0.5|$ under Tier 2A eigenvalues and Tier 3 Riemannian) with eight CHB-MIT clinical severity variables (age, seizure count, mean/median/max seizure duration, seizure-carrying files, total recording hours, and seizure density per hour), no pair crossed the pre-declared exploratory threshold of $|\rho| > 0.4$ AND $p < 0.05$. One near-miss appeared for Tier 2A magnitude vs age years (Spearman $\rho = +0.416$, $p = 0.054$, $n=22$), suggesting a weak positive relation between age and how strongly the eigenvalue-basis classifier separates from chance, but the test does not clear the 0.05 threshold. The one flagged point-biserial finding was Tier 3 polarity sign (standard vs inverted) vs age ($r_{pb} = -0.473$, $p = 0.026$, $n=22$): under Tier 3, inverted-polarity patients skewed younger. Bonferroni-corrected alpha across all 32 tests is $\approx 0.0016$, so this hit is exploratory and could easily be a false positive at this sample size; it also is not corroborated by the Tier 2A sign or by Spearman on the magnitude itself. If real, it is more plausibly a signal of developmental cortical maturation affecting the Riemannian tangent projection than of seizure severity per se. Overall the hypothesis 'polarity magnitude scales with clinical seizure severity within CHB-MIT' is NOT supported by this dataset: the seven summary-derived severity variables show no reliable monotone relationship with $|\mathrm{AUC}-0.5|$ under either tier. This is consistent with a structural interpretation of polarity (a property of the covariance geometry, not of how many or how long a patient seized in the recording) but should be read as a failure to reject the null on $n=22$: a larger cohort, richer clinical annotations (seizure type, onset lateralization, medication load), or direct developmental / anatomical covariates could still surface an effect.

## Top |rho| pairs (Spearman)

| Polarity measure | Clinical variable | rho | p | n | flag |
|---|---|---|---|---|---|
| t2a_polarity_mag | age_years | +0.416 | 0.0541 | 22 |  |
| t2a_polarity_mag | n_seizure_files | -0.373 | 0.0875 | 22 |  |
| t2a_polarity_mag | n_seizures | -0.301 | 0.1730 | 22 |  |
| t3_polarity_mag | total_recording_hours | -0.226 | 0.3120 | 22 |  |
| t3_polarity_mag | age_years | -0.212 | 0.3438 | 22 |  |
| t2a_polarity_mag | seizure_density_per_hour | -0.161 | 0.4742 | 22 |  |

Total tests: 16. Bonferroni-adjusted alpha: 0.0031.

## Strong vs weak polar (7-patient hardware subset)

Strong polar: chb03, chb21 (from prompt).
Weak polar:   chb01, chb05, chb07, chb14 (from prompt).
Hardware subset also includes chb11 (not classified by prompt).

| Group | n | mean age | mean seizures | mean dur (s) | mean rec (h) | mean density (/h) | mean t2a mag | mean t3 mag |
|---|---|---|---|---|---|---|---|---|
| strong_polar | 2 | 13.5 | 5.5 | 53.6 | 35.4 | 0.153 | 0.229 | 0.338 |
| weak_polar | 4 | 10.4 | 5.8 | 76.1 | 43.2 | 0.163 | 0.069 | 0.169 |
| hardware_subset_all | 7 | 11.5 | 5.3 | 97.1 | 39.7 | 0.149 | 0.138 | 0.195 |

## Files in this directory

- `clinical_metadata.csv` — one row per patient, parsed from `chbNN-summary.txt` (age from PhysioNet documentation).
- `polarity_magnitude.csv` — Tier 2A / Tier 3 raw AUC, $|\mathrm{AUC}-0.5|$, and sign.
- `correlations.csv` — full Spearman table (16 pairs).
- `biserial.csv` — point-biserial for polarity sign vs clinical vars.
- `hardware_subset_profile.csv` — strong / weak / all hardware means.
- `scatters/*.png` — plotted only for FOLLOW-UP-flagged pairs.

## Reading guide

- This is exploratory, not confirmatory.
- Bonferroni is reported for context; the small n means the Bonferroni-corrected test has near-zero power against realistic effect sizes.
- A null result narrows interpretation: polarity direction and magnitude appear independent of the severity variables that CHB-MIT summaries expose. Structural / anatomical / medication-status variables were not tested here and remain plausible mediators.
---

## Geometric addendum (post PROMPT 036)

**Generated:** 2026-09-02T20:06:23

Two basis-independent geometric polarity measures per patient (primary; |AUC-0.5| is secondary): tangent-space shift magnitude ||s_p|| (TangentSpace metric='riemann', reference = patient's own Frechet mean, seizure-class mean tangent vector minus interictal mean) and the Bures-Wasserstein distance between the per-class Wasserstein barycenters of the 8x8 covariances. 20 s windows, LWF covariance estimator, seizure class = ictal + preictal, as in the Tier 3 full-cohort run.

| patient | ||s_p|| | ||s_p|| corr | BW dist | windows |
|---------|---------|--------------|---------|---------|
| chb01 | 4.946 | 4.829 | 108.1422 | 24 @ 20s |
| chb02 | 4.808 | 4.498 | 129.2335 | 15 @ 20s |
| chb03 | 7.642 | 7.493 | 162.5279 | 24 @ 20s |
| chb04 | 2.335 | 1.295 | 36.8576 | 18 @ 20s |
| chb05 | 2.823 | 2.641 | 128.9702 | 20 @ 20s |
| chb06 | 1.583 | 1.325 | 42.3749 | 22 @ 20s |
| chb07 | 3.623 | 3.237 | 95.0739 | 16 @ 20s |
| chb08 | 3.798 | 3.641 | 140.2262 | 20 @ 20s |
| chb09 | 5.942 | 5.600 | 334.5068 | 18 @ 20s |
| chb10 | 1.488 | 1.347 | 62.4000 | 24 @ 20s |
| chb11 | 3.415 | 3.174 | 79.4418 | 16 @ 20s |
| chb13 | 2.024 | 1.849 | 44.5203 | 32 @ 20s |
| chb14 | 1.730 | 1.525 | 42.5263 | 23 @ 20s |
| chb15 | 2.198 | 1.222 | 118.8221 | 50 @ 20s |
| chb16 | 1.996 | 1.680 | 44.4507 | 20 @ 20s |
| chb17 | 2.761 | 2.280 | 55.9451 | 16 @ 20s |
| chb18 | 3.093 | 2.727 | 63.6439 | 22 @ 20s |
| chb19 | 3.038 | 1.853 | 79.5774 | 16 @ 20s |
| chb20 | 1.888 | 1.706 | 36.9560 | 26 @ 20s |
| chb21 | 3.122 | 2.866 | 63.6509 | 17 @ 20s |
| chb22 | 4.020 | 3.777 | 97.7219 | 16 @ 20s |
| chb23 | 2.074 | 1.873 | 51.5363 | 24 @ 20s |

`s_norm_corr` subtracts the sampling-variance bias term tr(Cov)/n per class from ||s_p||^2 (raw norms are inflated for patients with few seizure windows). Confound diagnostic - Spearman vs n_seizure_windows: s_norm: rho=-0.440 (p=0.040), s_norm_corr: rho=-0.382 (p=0.079), bw_dist: rho=-0.221 (p=0.322).

Spearman correlations vs the 8 clinical variables (40 tests total incl. the Tier 2A/Tier 3 magnitudes; Bonferroni alpha = 0.0013):

| clinical variable | measure | rho | p | flag |
|-------------------|---------|-----|---|------|
| n_seizures | s_norm | -0.581 | 0.0046 | FLAG |
| n_seizures | s_norm_corr | -0.524 | 0.0122 | FLAG |
| mean_seizure_duration_sec | bw_dist | +0.506 | 0.0162 | FLAG |
| median_seizure_duration_sec | bw_dist | +0.475 | 0.0255 | FLAG |
| seizure_density_per_hour | s_norm | -0.464 | 0.0298 | FLAG |
| age_years | s_norm | +0.438 | 0.0416 | FLAG |
| mean_seizure_duration_sec | s_norm | +0.405 | 0.0616 |  |
| n_seizure_files | s_norm | -0.399 | 0.0659 |  |
| max_seizure_duration_sec | bw_dist | +0.387 | 0.0753 |  |
| seizure_density_per_hour | s_norm_corr | -0.360 | 0.1001 |  |
| n_seizure_files | s_norm_corr | -0.359 | 0.1004 |  |
| median_seizure_duration_sec | s_norm | +0.354 | 0.1059 |  |
| n_seizures | bw_dist | -0.350 | 0.1101 |  |
| max_seizure_duration_sec | s_norm | +0.291 | 0.1891 |  |
| seizure_density_per_hour | bw_dist | -0.256 | 0.2506 |  |
| mean_seizure_duration_sec | s_norm_corr | +0.237 | 0.2891 |  |
| age_years | bw_dist | +0.210 | 0.3473 |  |
| total_recording_hours | bw_dist | +0.196 | 0.3822 |  |
| total_recording_hours | s_norm | +0.184 | 0.4137 |  |
| median_seizure_duration_sec | s_norm_corr | +0.173 | 0.4403 |  |
| age_years | s_norm_corr | +0.149 | 0.5072 |  |
| n_seizure_files | bw_dist | -0.136 | 0.5458 |  |
| max_seizure_duration_sec | s_norm_corr | +0.136 | 0.5475 |  |
| total_recording_hours | s_norm_corr | +0.032 | 0.8869 |  |

Flagged geometric correlations (|rho| > 0.4, p < 0.05): **6/24**; surviving Bonferroni: 0.

Strong vs weak polar (geometric measures; n=2 vs n=4, two-sided Mann-Whitney):

| variable | strong mean | weak mean | p |
|----------|-------------|-----------|---|
| s_norm | 5.3822 | 3.2804 | 0.5333 |
| s_norm_corr | 5.1796 | 3.0581 | 0.5333 |
| bw_dist | 113.0894 | 93.6782 | 0.8000 |

### Addendum verdict

The basis-independent geometric quantities behave differently from |AUC - 0.5|: 6 of 40 exploratory tests flag, and every flag involves a geometric measure while the Tier 2A / Tier 3 magnitudes remain null. The pattern that survives scrutiny is a seizure-phenotype axis, not a simple severity axis. Tangent-shift magnitude ||s_p|| correlates NEGATIVELY with seizure count (rho = -0.58, p = 0.005; still rho = -0.52, p = 0.012 after subtracting the small-sample bias term tr(Cov)/n that inflates norms for patients with few seizure windows), and the Bures-Wasserstein distance correlates POSITIVELY with mean and median seizure duration (rho = +0.51 / +0.48, p = 0.016 / 0.026; this measure is nearly uncorrelated with window count, rho = -0.22, so the count confound does not explain it). Together: patients with fewer, longer seizures show larger geometric ictal-interictal separation on the covariance manifold. The age and seizure-density flags do NOT survive bias correction (age: +0.44 -> +0.15; density: -0.46 -> -0.36, n.s.) and should be treated as artifacts of the raw-norm bias. Caveats: 0/40 tests survive Bonferroni (alpha = 0.0013); and n_seizure_windows tracks n_seizures almost one-to-one, so even after the variance correction a mechanical explanation remains open - averaging over more, heterogeneous seizures can shrink a mean shift vector by cancellation without any per-seizure difference. Verdict: partial, exploratory support for the interpretation-guide's positive branch - the geometric separation reflects seizure phenotype (duration vs frequency), while polarity magnitude |AUC - 0.5| itself is severity-independent, consistent with the structural mechanism. Before promoting this to a Paper 3 finding, run the targeted follow-up: per-seizure shift vectors s_p^(k) per patient, testing whether few-seizure patients have genuinely larger per-seizure shifts or whether many-seizure patients lose magnitude to cancellation.

---

## Correction block - PROMPT 037 follow-up (2026-09-03)

The two flagged geometric correlations above were re-examined per seizure
event in `results/prompt037/` (tiled 8 s windows, every seizure represented,
bias-corrected per-seizure decomposition). Outcome:

- The `||s_p||` vs `n_seizures` correlation is **not** a 1/sqrt(K) averaging
  artifact: within-patient per-seizure shift vectors are coherent (c ~ 1) and
  the correlation reproduces on the K-independent per-seizure magnitude
  (L_corr vs n_seizures rho = -0.566, Holm p = 0.018). The sqrt(K) diagnostic
  in PROMPT 037 Step 0 cannot separate averaging from a real L-K dependence
  and should not be cited as evidence of either.
- The loader used here drops ictal windows for short-seizure patients (chb16
  0/10, chb06 2/10); the `bw_dist` values in this addendum for chb02, chb06,
  chb14, chb16, chb21 are affected. With the dropout fixed the duration
  correlation strengthens (D_bar vs mean_dur rho = +0.656, Holm 0.0037,
  partial | n_seizures +0.549).
- Preictal-only control arm: all four primary tests null.

The numbers in this addendum stand as a record of what was run; for Paper 3
use `results/prompt037/` values. See `results/prompt037/SUMMARY.md`.
