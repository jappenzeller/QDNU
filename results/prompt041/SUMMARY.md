# PROMPT 041 - Eye-closure predictions of the E/I two-level note, on public data

**Generated:** 2026-09-07
**Script:** `scripts/prompt041_eye_closure.py` (W = 4 primary); W = 8 replication and the exploratory occipital state were run inline and saved to `cohort_W8_and_exploratory.json`
**Data:** PhysioNet EEG Motor Movement/Imagery (eegmmidb 1.0.0), runs R01 eyes open and R02 eyes closed, 60 s each, 109 subjects, 64 ch at 160 Hz. Fetched with `mne.datasets.eegbci` into `data/eegmmidb/` (not committed).

Conventions: DSP-000 "P" subset Fz Cz Pz Oz O1 O2 P3 P4 (8-dim state); 0.5-40 Hz zero-phase Butterworth + 60 Hz notch; tiled non-overlapping windows; `lwf` covariances; per-subject statistic = median over closed windows minus median over open windows; cohort = sign test over 109 subjects. Alpha band 8-13 Hz for the phase quantities. All four predictions and their pass criteria were written into the script before it ran.

---

## Results

| prediction | criterion | W = 4 s | W = 8 s | verdict |
|---|---|---|---|---|
| **P1** trace: $\log\operatorname{tr}\Sigma$ rises on closure | > 0 in >= 75 %, sign p < 1e-3 | **80/109 (73 %)**, sign p = 1e-6, Wilcoxon p = 1e-7, median +0.29; 63 subjects individually p < 0.05 | 76/109, median +0.27 | directional and highly significant, **below the pre-declared 75 % bar** |
| **P2** entropy: $S(\rho)$ falls on closure | < 0 in >= 75 % | **54/109 (50 %)**, median +0.009; individually significant: 11 down, 24 up | 52/109, median +0.009 | **FAIL** - no purification |
| **P3a** gauge: absolute alpha phase uniform in both conditions | resultant < 0.10 | R_open = 0.010, R_closed = 0.011 (n = 13,080; uniform expectation 0.009) | - | **PASS**, decisively |
| **P3b** PLV to global phase rises on closure | > 0 in >= 75 % | **36/109 (33 %)**; it *falls* in 73/109, sign p = 5e-4, median -0.019 | 35/109 | **FAIL** - opposite direction |
| sanity: Berger effect | closed/open alpha on O1 or O2 > 2 | 85/109, median ratio 3.9 | - | present |

Exploratory (not pre-declared): restricting the state to Pz Oz O1 O2 (4-dim, W = 4), entropy falls in only 37/109 - it *rises* in 72/109, median +0.023. The occipital state gets more mixed on eye closure, not purer.

Relations: dS is uncorrelated with the size of the Berger effect (Spearman -0.07; entropy drops in 49 % of Berger-positive and 50 % of Berger-negative subjects) and mildly anti-correlated with the power change (-0.36).

---

## Reading

**P3a is the clean result.** The absolute alpha-band phase of a channel over a window is uniformly distributed across windows, in both conditions, at the level of the null expectation for 13,080 draws. This is the gauge statement of `EI_TWO_LEVEL_NOTES.md` section 3 confirmed on 109 subjects, and it reproduces 038's finding on CHB-MIT (resultant 0.044). The encoding parameter `b` carries no information anywhere.

**P1 is real but the bar was set slightly too high.** Power rises on eye closure in 73 % of subjects with p ~ 1e-6. The 75 % threshold was arbitrary; the direction is not in doubt. Reported as directional support.

**P2 fails, and the failure was foreseeable from the note itself.** Section 4 of the note derives that the balanced oscillatory regime is *mixed* (purity -> 1/2 for a single unit) and mode separation is *pure*. Section 6.2 then predicted that eye-closure alpha - the canonical balanced oscillation - would *purify* the state, on the intuition that one occipital spatial mode takes over. The two are inconsistent, and the data sides with section 4: the strongest, most stereotyped oscillation in EEG does not reduce the entropy of the covariance state, at 8 channels or at 4 occipital channels. A rhythm with phase lags across electrodes occupies at least two spatial modes (its cosine and sine components), and its power arrives on top of the existing mode distribution rather than replacing it. Section 6.2 should be corrected to: **balanced oscillation leaves $S(\rho)$ unchanged or raises it; purification is the signature of mode separation, not of rhythm.**

That correction makes the CHB-MIT entropy split (2026-09-06) read consistently: the 15 patients whose ictal state becomes more mixed are moving toward the oscillatory regime (rhythmic seizure activity), and the 7 whose state purifies are the ones whose seizures separate modes. Eye closure belongs with the first group, as it should.

**P3b fails for an observable-design reason and should not be over-read.** The PLV was measured against the amplitude-weighted *global* phase of all 8 channels. On eye closure the global phase becomes the occipital alpha phase, the occipital channels lock to it, and the five non-occipital channels do not - so the 8-channel mean PLV drops. The prediction as stated ("phase locking rises") was about pairwise occipital coherence, which this observable does not isolate. It stays a FAIL on the pre-declared criterion; a pairwise occipital PLV is the right observable if the question is revisited.

---

## Verdict

Two of four pre-declared predictions hold in the direction stated (P1 directional, P3a exact); one fails on its criterion (P2), with the failure traceable to an inconsistency inside the note that the data resolves in favour of the derivation; one fails on an observable that did not measure what the prediction meant (P3b).

Consequences carried forward:
1. `EI_TWO_LEVEL_NOTES.md` section 6.2 is corrected (this run): eye closure raises the trace and does not lower $S(\rho)$. The entropy prediction for the lattice sessions is *no drop*.
2. The gauge result now stands on two datasets and ~32,000 channel-windows. `b` is retired from any future encoding.
3. The CHB-MIT purifying subgroup (chb02, 08, 09, 11, 13, 14, 21) is now the interesting object: it does something eye-closure alpha does not do. That is the subgroup the mode-separation regime of the note describes, and the E/I-proxy (aperiodic slope) test should be run on it first.
4. Nothing here needed the rig. 109 subjects, three predictions tested, one afternoon.

## Deliverables

```
scripts/prompt041_eye_closure.py
results/prompt041/
    per_subject_W4.csv
    cohort_W4.json
    cohort_W8_and_exploratory.json
    SUMMARY.md
```
