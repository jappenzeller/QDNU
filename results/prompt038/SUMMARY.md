# PROMPT 038 - Is polarity in the state or in the trace, and does the A-Gate's E/I balance read it?

**Generated:** 2026-09-05
**Scripts:** `scripts/prompt038_state_vs_trace.py` (Q1), `scripts/prompt038_agate_balance.py` (Q2)
**Depends on:** `results/prompt037/cache/W8/` (windows reused unchanged, asserted), `results/prompt035/polarity_magnitude.csv`, `docs/AGATE_OBSERVABLE_NOTES.md`, `docs/PROMPT_038.md`

Conventions: 22 CHB-MIT patients (chb12, chb24 excluded), CH8 LOSO montage, W = 8 s tiled windows, seeded 10-window interictal cap, `lwf` covariances, `TangentSpace(metric='riemann')` at the patient's own Frechet mean (d = 36) - all identical to 037. A-Gate encoding is `extract_plv_params` from `scripts/braket_ch16_simulation.py` (PROMPT 006 lineage), unchanged: a = min-max-normalised Hilbert envelope in [0.05, 0.95], b = circular-mean phase, c = PLV to the global analytic phase in [0.05, 0.95], band 4-13 Hz. Single-channel circuit from `qdnu/quantum_agate.py` including the CRy/CRz coupling; 8-channel circuit from `qdnu/multichannel_circuit.py` (closed CNOT rings, ancilla). Dead zone $|\sin b| < 0.3$ or $|\cos b| < 0.3$; amplitude bound pi/4; 2000 bootstrap resamples, seed 38.

Both questions were pre-declared in `docs/PROMPT_038.md`; the verdicts below use those branches.

---

## Q1 - Scale vs shape decomposition of the ictal shift

### Calibration (unit tests)

- Scale direction: $\operatorname{vec}(I_8)/\sqrt{8}$ in pyriemann's vectorisation; round-trip `inverse_transform(t*u)` returns $e^{t}\,C_{\rm ref}$ with max relative error 6e-15 across 22 patients. PASS.
- Exact identity $\sqrt{8}\,\alpha_p = \overline{\log\det\Sigma}_{\rm ictal} - \overline{\log\det\Sigma}_{\rm interictal}$: max error 2e-14. PASS (this is the stronger check; it is exact, not approximate).
- Log-trace ratio of the class Riemannian means vs `alpha_p`: same sign in 20/22, Spearman 0.928. The two misses are chb06 (alpha = +0.04, log-trace -0.26) and chb21 (alpha = -0.75, log-trace +0.26), both patients whose scale component is near zero, where log-det and log-trace can legitimately disagree in sign. The pre-declared "22/22" was too strict for near-zero cases; the identity test is the one that matters and it passes.

### Per-patient split

`f_scale` = fraction of $\|\bar s\|^{2}$ along the pure-scaling direction. `alpha` > 0 means ictal total power up.

| patient | K | alpha | \|\|s_bar\|\| | f_scale | per-seizure f_scale min / med / max | log-trace ratio | shape proj. | t3_polarity |
|---|---|---|---|---|---|---|---|---|
| chb01 | 7 | +7.50 | 7.87 | 0.907 | 0.84 / 0.89 / 0.93 | +2.40 | -1.82 | standard |
| chb02 | 3 | +6.67 | 6.91 | 0.933 | 0.87 / 0.92 / 0.95 | +2.60 | -0.47 | standard |
| chb03 | 7 | +10.55 | 10.71 | 0.970 | 0.92 / 0.94 / 0.96 | +3.57 | +1.12 | inverted |
| chb04 | 4 | +6.22 | 6.83 | 0.829 | 0.49 / 0.78 / 0.92 | +1.94 | -2.06 | standard |
| chb05 | 5 | +6.98 | 7.16 | 0.951 | 0.91 / 0.94 / 0.95 | +2.39 | -0.63 | standard |
| chb06 | 10 | +0.04 | 1.83 | 0.001 | 0.00 / 0.07 / 0.52 | -0.26 | +0.79 | inverted |
| chb07 | 3 | +7.11 | 7.74 | 0.845 | 0.70 / 0.86 / 0.87 | +1.55 | -2.83 | standard |
| chb08 | 5 | +5.79 | 6.09 | 0.903 | 0.87 / 0.89 / 0.92 | +2.04 | +1.19 | inverted |
| chb09 | 4 | +10.48 | 10.58 | 0.981 | 0.98 / 0.98 / 0.98 | +3.76 | +0.10 | standard |
| chb10 | 7 | +2.71 | 3.54 | 0.585 | 0.10 / 0.61 / 0.73 | +0.67 | -0.87 | inverted |
| chb11 | 3 | +4.94 | 5.47 | 0.815 | 0.71 / 0.75 / 0.88 | +2.33 | +0.86 | standard |
| chb13 | 12 | +0.43 | 2.43 | 0.031 | 0.01 / 0.15 / 0.73 | +0.39 | -0.81 | standard |
| chb14 | 8 | -2.80 | 3.59 | 0.606 | 0.19 / 0.50 / 0.81 | -0.83 | +1.72 | standard |
| chb15 | 20 | -1.65 | 2.01 | 0.677 | 0.00 / 0.52 / 0.81 | -0.46 | -0.49 | standard |
| chb16 | 7 | +3.99 | 4.36 | 0.836 | 0.03 / 0.86 / 0.93 | +1.05 | -1.15 | standard |
| chb17 | 3 | +4.78 | 5.27 | 0.824 | 0.73 / 0.75 / 0.90 | +1.22 | -0.77 | standard |
| chb18 | 6 | +2.88 | 4.20 | 0.468 | 0.28 / 0.63 / 0.83 | +0.40 | -2.73 | standard |
| chb19 | 3 | +8.30 | 8.69 | 0.912 | 0.88 / 0.92 / 0.92 | +2.18 | -2.11 | standard |
| chb20 | 8 | +2.55 | 3.08 | 0.686 | 0.39 / 0.66 / 0.79 | +0.75 | -0.78 | inverted |
| chb21 | 4 | -0.75 | 2.61 | 0.084 | 0.03 / 0.07 / 0.16 | +0.26 | +0.34 | standard |
| chb22 | 3 | +6.17 | 6.70 | 0.849 | 0.80 / 0.85 / 0.88 | +1.95 | -0.97 | inverted |
| chb23 | 7 | +4.87 | 5.23 | 0.866 | 0.80 / 0.86 / 0.88 | +1.58 | -0.67 | standard |

**Median f_scale = 0.83** (range 0.001-0.981). For 14 of 22 patients more than 80 % of the ictal displacement is a uniform increase in power (18 of 22 above 50 %); for 19 of 22 the scale component is positive. The per-seizure spread is narrow for the high-f_scale patients (the seizures agree on *how much* is scale, not just on direction - this is 037's coherence result seen from another angle). The exceptions are the many-seizure / short-seizure patients (chb06, chb13, chb21, and to a lesser degree chb14, chb15, chb18) whose shifts are small and mostly shape.

### Polarity vs scale, polarity vs shape (`q1_polarity_crosstab.json`)

| test | agree with t3_polarity | Fisher p |
|---|---|---|
| sign(alpha) | 13/22 | 0.53 |
| sign of projection on cohort shape axis (leading PC of s_perp, 38 % variance, sign anchored on chb03) | 7/22 (15/22 after global flip) | 0.33 |
| exploratory: leave-one-out cosine of each shape vector with the others' mean | 14/22 | - |

Neither the scale sign nor the leading shape axis reproduces `t3_polarity`. Two facts explain most of this. First, 19 of 22 scale components are positive, so a scale sign cannot encode a two-class label that splits 16/6. Second, the cross-patient shape vectors are nearly orthogonal (leave-one-out cosines all within +/-0.2 in 36 dimensions), so there is no shared shape direction for a cohort sign to live on. The t3 polarity label itself is a leave-one-subject-out LDA direction, which whitens by within-class covariance and therefore need not align with either the scale direction or the mean-difference shape direction; and 9 of the 22 labels come from raw AUC within 0.05 of 0.5 (chb04, 07, 09, 11, 13, 14, 15, 18, 20), i.e. they are essentially coin flips. Any agreement count over all 22 has a floor of noise built in.

### Shape-only Bures-Wasserstein (`q1_shape_only_bw.csv`)

Per-seizure BW distance recomputed on trace-normalised covariances $\rho = \Sigma/\operatorname{tr}\Sigma$:

| | Spearman | p |
|---|---|---|
| $\bar D_{\rho}$ vs 037 D_bar | -0.12 | 0.59 |
| $\bar D_{\rho}$ vs mean_seizure_duration | -0.18 | 0.41 |
| $\bar D_{\rho}$ vs n_seizures | -0.09 | 0.69 |

**037's surviving duration correlation (P3, rho = +0.66 on D_bar) does not survive trace normalisation.** It was a power effect: patients with longer seizures have a larger ictal power increase. The shape-only distance carries no duration or count signal at all.

### Q1 verdict: **Trace-carried** (pre-declared branch: median f_scale > 0.5)

The geometric displacement that 035-037 measured is, for most patients, a uniform increase in total power. The "magnitude phenotype" (P1) and the duration axis (P3) are largely statements about ictal power, and P3 vanishes on the trace-normalised state. The trace-normalised density matrix rho, which is what DSP-000 proposes to prepare, does not carry this displacement. It does not follow that rho carries *nothing* - the six low-f_scale patients have real shape shifts, and every patient has a non-zero s_perp - but the headline geometry of the last three prompts is not in the state.

Polarity is a separate matter: neither the scale sign nor the leading shape axis reproduces `t3_polarity`, so the pre-declared "sign(alpha) reproduces t3 in >= 18/22" clause is *not* met and the trace-carried call rests on the f_scale clause alone. Where the polarity label lives geometrically is not settled by Q1.

---

## Q2 - A-Gate balance observable vs manifold polarity

### Unit tests (`q2_unit_tests.json`) - all PASS

- Closed forms for <X>, <Z> on both uncoupled qubits vs statevector: max error 8e-16.
- numpy two-qubit state vs `create_single_channel_agate` (qiskit, q0 = E): max error 1e-15 after the little-endian swap.
- Balance exactness on the coupled state, $\langle U X_E U^{\dagger}\rangle/\sin b + \langle U X_I U^{\dagger}\rangle/\cos b = \sin 2a - \sin 2c$: max error 1e-13.
- Pauli expansion of $U X_E U^{\dagger}$: 0.7886 X_E + 0.3266 Y_E + 0.1464 X_E Y_I + 0.1353 X_E Z_I - 0.3536 Y_E Y_I - 0.3266 Y_E Z_I. Leading coefficient on X_E confirms qubit order.
- Ancilla shortcut vs the 17-qubit statevector: max error 2e-15. **The closed CNOT ring maps the E-parity onto channels {0, 2, 4, 6} only.** Paper 1's ancilla readout is `prod_{i in {0,2,4,6}} <Z_{E_i}>` - the product of the excitatory Z-expectations of the four even-indexed channels (FP1-F7, FP1-F3, FP2-F8, FP2-F4 in the CH8 order). The odd channels do not reach the readout.
- Window sets reproduce `results/prompt037/window_inventory.csv` for 22/22 (asserted in the EDF pass).

### Step 0 - encoding coverage (`q2_step0_encoding_coverage.csv`)

Pooled over 18,816 channel-windows:

| quantity | fraction |
|---|---|
| a > pi/4 | 0.216 |
| c > pi/4 | 0.023 |
| \|sin b\| < 0.3 (E term unreadable) | 0.194 |
| \|cos b\| < 0.3 (I term unreadable) | 0.198 |
| **fully readable** | **0.471** |

Per patient the readable fraction is 0.43-0.49, uniformly. **Below the pre-declared 0.5 threshold, so the Q2 verdict is provisional whatever it says.**

Two encoding facts fell out of this that matter beyond 038. The dead-zone fractions are 0.194 and 0.198, which is exactly what a uniformly distributed phase gives (2 arcsin(0.3)/pi = 0.194). Direct check: the pooled circular resultant length of b is 0.044 and the 8-bin histogram is flat to +/-2 %. **The shared phase parameter b is, to a good approximation, uniformly random per window.** The circular mean of a 4-13 Hz phase over 8 s does not carry structure. Consequently the sin b / cos b gains in the A-Gate are random draws, and the "E-I coupling through the shared phase" is a random per-window rotation of which amplitude reaches the readout. Second, c (PLV) sits at median 0.51-0.52 in both classes and almost never exceeds pi/4, while a is min-max normalised *within each window* (median 0.39 ictal, 0.45 interictal), so a cannot encode the absolute power change that Q1 shows is the dominant ictal effect.

### Per-patient signs (`q2_per_patient_signs.csv`)

delta = median(ictal) - median(interictal) per observable; "det" = bootstrap sign stability >= 0.70, otherwise "und" (undetermined, not counted).

| patient | t3 (raw AUC) | hw (Paper 2) | Pi_bal | Pi_naive (Z_E - Z_I) | Pi_anc |
|---|---|---|---|---|---|
| chb01 | + (0.86) | + | **+0.108** det | +0.017 det | +0.008 det |
| chb02 | + (0.66) | | +0.011 und | +0.014 und | -0.023 det |
| chb03 | - (0.30) | - | +0.017 und | -0.027 det | -0.025 det |
| chb04 | + (0.55) | | -0.060 und | +0.001 und | -0.020 det |
| chb05 | + (0.76) | + | +0.001 und | -0.034 und | -0.019 det |
| chb06 | - (0.34) | | **-0.037** det | -0.036 det | -0.063 det |
| chb07 | + (0.50) | + | -0.033 det | -0.081 det | -0.026 det |
| chb08 | - (0.13) | | **-0.194** det | -0.129 det | +0.006 det |
| chb09 | + (0.51) | | -0.028 und | -0.075 det | +0.010 und |
| chb10 | - (0.21) | | **-0.105** det | -0.171 det | -0.025 det |
| chb11 | + (0.52) | - | -0.035 det | -0.138 det | -0.043 det |
| chb13 | + (0.51) | | -0.045 und | +0.012 und | +0.022 det |
| chb14 | + (0.55) | + | -0.140 det | +0.303 det | +0.034 det |
| chb15 | + (0.55) | | -0.037 det | -0.076 det | -0.024 det |
| chb16 | + (0.65) | | **+0.180** det | +0.189 det | -0.010 und |
| chb17 | + (0.62) | | +0.012 und | -0.096 det | +0.034 det |
| chb18 | + (0.52) | | +0.091 det | -0.170 det | -0.040 det |
| chb19 | + (0.73) | | **+0.065** det | -0.214 det | -0.029 det |
| chb20 | - (0.46) | | +0.058 det | +0.006 und | -0.070 det |
| chb21 | + (0.98) | - | -0.055 det | +0.162 det | -0.029 det |
| chb22 | - (0.22) | | **-0.232** det | -0.030 det | +0.004 und |
| chb23 | + (0.68) | | **+0.040** det | +0.019 und | -0.046 det |

Bold = Pi_bal determined and agreeing with t3 among the 13 patients whose t3 label is itself determined ($|\mathrm{AUC} - 0.5|$ >= 0.1).

### Agreement (`q2_agreement.json`)

| observable | vs t3, all 22 (determined) | p | vs t3, \|AUC-0.5\| >= 0.1, n = 13 (post hoc) | p | vs Paper 2 hardware, 7 pts | p |
|---|---|---|---|---|---|---|
| **Pi_bal** | 9 / 15 | 0.61 | **8 / 9** | **0.039** | 3 / 5 | 1.0 |
| Pi_naive | 9 / 16 | 0.80 | 8 / 10 | 0.11 | 4 / 6 | 0.69 |
| Pi_anc | 8 / 19 | 0.65 | 5 / 11 | 1.0 | 5 / 7 | 0.45 |

Best-global-flip counts do not change any of these (the flip never helps).

Cross-check with Q1: Spearman of delta(Pi_bal) against Q1's alpha is +0.33 (p = 0.14), below the pre-declared 0.7 - the balance observable is not a power detector in disguise. Against 037's $L_{\rm corr}$ signed by t3: +0.43 (p = 0.046).

### Q2 verdict: **Inconclusive** (pre-declared: coverage < 0.5, and the all-22 count is at chance)

On the pre-declared primary comparison - Pi_bal against `t3_polarity` over all 22 - the derived observable is at chance (9/15 determined). So are the naive control and the ancilla. Coverage is 0.47, so the verdict is provisional by the prompt's own rule.

What is *not* null: restricted to the 13 patients whose t3 label is determined, Pi_bal agrees on 8 of the 9 patients where it is itself determined (p = 0.039), and the single miss is chb21, whose t3 and hardware polarities disagree with each other (t3 standard at AUC 0.98; hardware inverted) - Pi_bal sides with the hardware there. The naive control on the same subset is 8/10 (p = 0.11) and the ancilla 5/11. This subset restriction was **not** pre-declared; it is a post-hoc cut motivated by the observation that nine t3 labels are coin flips, and it is reported as a lead, not a result. The right next step is to pre-declare the determined-label subset and re-test on independent windows (W = 20 s cache, or the 1.95 s windows Paper 2 used) before believing it.

The ancilla observable - Paper 1's actual readout, simulated noiselessly - does not track t3 polarity or, more surprisingly, the Paper 2 hardware polarity assignments (5/7, p = 0.45). Those assignments came from a template-fidelity classifier on 1.95 s windows with shot noise, not from the ancilla expectation on 8 s windows, so this is not a contradiction of Paper 2; it does mean the polarity that hardware exhibited is not a simple sign of <Z_anc>.

---

## Cross-check between Q1 and Q2

Q1 landed trace-carried; Q2 did not land link-established, so the pre-declared cross-check condition does not arise. The correlation that would have flagged a power detector (delta Pi_bal vs alpha) is +0.33, well below 0.7. The balance observable and the scale component are measuring different things, which is expected: a is min-max normalised within each window and cannot see absolute power.

---

## Consequences for DSP-000 Phase 2

1. **The state alone does not carry the 035-037 geometry.** Median 83 % of the ictal displacement is uniform power gain, and the duration correlation (P3) vanishes on rho. Phase 2 must carry $\operatorname{tr}\Sigma$ as an explicit classical channel alongside the prepared state, and any observable built on rho should be expected to see a *different* structure from the one 035-037 characterised - the shape residual s_perp, which is patient-specific (cross-patient cosines within +/-0.2) and small for the high-power patients.
2. **Candidate A (first principal tangent direction) needs re-scoping.** With a median 83 % of each shift along the scale direction, the first principal direction of the cohort's full shift vectors will be dominated by scale (not computed here, but it follows from the f_scale column), and trace-normalisation removes that direction by construction. Candidate A on rho means the leading direction of s_perp, which explains only 38 % of the cohort's shape variance and does not reproduce t3 polarity. It is not a strong candidate as stated.
3. **Candidate B (PN-derived balance) survives as a lead, not as a result.** It is at chance over all 22 and 8/9 on the determined-label subset. Two encoding defects limit it independently of the geometry: b is uniform noise, so the gains that select which amplitude is read are random per window; and only 47 % of channel-windows are in the observable's readable range. A re-test on the determined subset with pre-declaration, plus a fix for b (e.g. instantaneous phase at window centre, or dropping b from the encoding), is the path.
4. **The t3 polarity label is not a clean target.** Nine of 22 labels are within 0.05 of chance. Any future polarity comparison should state which patients have a determined label and test on those, pre-declared. Paper 2's hardware assignments and t3 disagree on chb11 and chb21.
5. **Paper 1's ancilla reads four of eight channels.** The closed CNOT ring folds the E-parity onto channels 0, 2, 4, 6. This is a fact about the circuit that should be stated wherever the architecture is described.

---

## Deliverables

```
scripts/prompt038_state_vs_trace.py
scripts/prompt038_agate_balance.py
results/prompt038/
    q1_calibration.json
    q1_scale_shape.csv
    q1_per_seizure.csv
    q1_polarity_crosstab.json
    q1_shape_only_bw.csv
    q2_unit_tests.json
    q2_step0_encoding_coverage.csv
    q2_per_patient_signs.csv
    q2_per_window_chbNN.npz
    q2_agreement.json
    abc_cache/chbNN.npz
    scatters/q2_delta_{bal,anc}_vs_alpha.png
    SUMMARY.md
```

Run: `python scripts/prompt038_state_vs_trace.py`; `python scripts/prompt038_agate_balance.py --stage {test,abc,analyze,all} [--data-root ...]`.
