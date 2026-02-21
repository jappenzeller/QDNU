# Classical Baseline Implementation Plan
## QPNN Quantum Practicality — EEG Seizure Detection

**Purpose:** Establish a rigorous multi-tier classical baseline against which QPNN/A-Gate quantum results
can be honestly compared. The quantum practicality argument (that quantum state space is a natural geometric
fit for the brain state manifold) is only defensible if the classical comparators use equivalent or better
preprocessing and include Riemannian geometry methods that approximate the same manifold.

**Dataset:** CHB-MIT scalp EEG (23 patients, ~1000 hours)  
**Reference implementations:** MichaelHills/seizure-detection (Kaggle winner, UPenn/Mayo), Barachant (2016 seizure prediction winner)  
**Target tiers:**
- Tier 1 — Classical Euclidean: FFT power bands (current weak baseline)
- Tier 2 — Classical Correlation Eigenvalue: MichaelHills method (strong classical)
- Tier 3 — Classical Riemannian: Barachant SPD/pyRiemann method (best classical, maps directly to quantum geometry argument)

---

## Prompt 1 — Audit and Fix Preprocessing Pipeline

**Context:** The current preprocessing pipeline has known critical gaps: missing bandpass filtering,
potentially problematic normalization, and unvalidated window sizing. Before adding new feature extractors,
the foundation must be correct. All downstream tiers depend on this.

**Background for executor:**  
The CHB-MIT dataset is 256 Hz scalp EEG. The winning competition solutions universally apply bandpass
filtering before any feature extraction. MichaelHills uses 1–47 Hz FFT magnitude. Barachant uses
0.5–128 Hz bandpass with 60 Hz notch then downsample to 256 Hz. For scalp EEG (not intracranial),
the meaningful physiological signal is 0.5–40 Hz. DC drift and line noise (60 Hz US) are artifacts.

**Prompt to Claude Code:**

```
I have an EEG seizure detection pipeline running on the CHB-MIT dataset. Audit the current 
preprocessing module and fix the following known gaps:

1. BANDPASS FILTER: Add a Butterworth bandpass filter, 5th order, 0.5–40 Hz cutoff for scalp EEG.
   Apply zero-phase filtering (scipy.signal.filtfilt, not lfilter — lfilter introduces phase shift
   that corrupts temporal features). Apply before any windowing or feature extraction.
   Add a 60 Hz notch filter (Q=30) for US power line noise rejection.

2. NORMALIZATION AUDIT: Check the current normalization approach. Channel-wise z-score normalization
   (zero mean, unit variance per channel per segment) is correct. Global normalization across channels
   is wrong — it destroys inter-channel amplitude relationships needed for the correlation features
   in later steps. If global normalization is present, replace with per-channel.

3. WINDOW SIZING VALIDATION: Confirm windows are non-overlapping for cross-validation integrity.
   The window size should be 1 second (256 samples at 256 Hz) for the MichaelHills feature pipeline,
   or 30 seconds for the Riemannian covariance pipeline. If overlapping windows are used for training
   data augmentation, ensure the CV split happens at the seizure level (whole seizures in/out),
   not at the window level — otherwise data leakage inflates validation scores.

4. OUTPUT: Write a preprocessing validation script that loads one CHB-MIT patient file, applies
   the corrected pipeline, and produces diagnostic plots: raw vs filtered signal overlay,
   power spectral density before/after filtering, and a summary of window counts per class.
   Save to preprocessing_validation.py.

The preprocessing function signature should be:
preprocess_eeg(raw_signal, fs=256, bandpass=(0.5, 40), notch=60, window_sec=1, 
               normalize='channel') -> np.ndarray of shape (n_windows, n_channels, n_samples)
```

**Acceptance criteria:**
- Bandpass and notch filters confirmed applied pre-windowing
- Per-channel normalization confirmed
- CV split operates at seizure level, not window level
- Diagnostic plots visually confirm filter is removing DC drift and line noise
- No data leakage between train and validation sets

---

## Prompt 2 — Tier 1 Baseline: FFT Power Bands (Document Current State)

**Context:** This is the existing weak baseline. The goal here is not to build it (it exists) but to
formalize it into a reproducible benchmark with honest metrics so Tier 2 and Tier 3 improvements
are measured against a fixed, documented reference.

**Background for executor:**  
The current pipeline extracts FFT features, but it has not been benchmarked with the corrected
preprocessing from Prompt 1, and results haven't been stored in a format that supports direct
comparison across tiers. This prompt standardizes the output format.

**Prompt to Claude Code:**

```
Implement a standardized benchmark runner for the Tier 1 (FFT Power Band) classical baseline
using the corrected preprocessing pipeline from the previous step.

Feature extraction:
- Apply FFT to each 1-second window per channel
- Extract mean log power in the standard EEG bands:
    delta: 0.5–4 Hz
    theta: 4–8 Hz  
    alpha: 8–13 Hz
    beta: 13–30 Hz
    gamma: 30–40 Hz
- Feature vector per window: 5 bands × n_channels = (5 * n_channels,) dimensional
- Apply log10 after averaging power within each band (log power is more Gaussian-distributed)

Classifier:
- Random Forest, 3000 trees, patient-specific models (train/test on same patient)
- Cross-validation: split on whole seizures (leave-one-seizure-out). For a patient with N seizures,
  train on N-1, validate on 1, rotate. Report mean ± std AUC across folds.

Output format (save to results/tier1_fft_power/{patient_id}.json):
{
  "patient_id": "chb01",
  "feature_dim": 80,
  "n_seizures": 7,
  "cv_auc_mean": 0.XX,
  "cv_auc_std": 0.XX,
  "cv_auc_per_fold": [...],
  "preprocessing": {
    "bandpass": [0.5, 40],
    "notch": 60,
    "window_sec": 1,
    "normalize": "channel"
  }
}

Also produce a summary CSV: results/tier1_summary.csv with one row per patient.
Run on all 23 CHB-MIT patients. This is the fixed reference point — do not modify it
after Tier 2 and Tier 3 are run.
```

**Acceptance criteria:**
- Results stored in structured JSON per patient
- AUC reported with standard deviation across folds (not just mean)
- Preprocessing parameters logged in results (reproducibility)
- Summary CSV produced for all 23 patients

---

## Prompt 3 — Tier 2 Baseline: MichaelHills Correlation Eigenvalue Features

**Context:** This is the strong classical baseline. MichaelHills won the Kaggle UPenn/Mayo seizure
detection competition using this exact feature set. The key insight is that the eigenvalues of the
inter-channel correlation matrix in both time and frequency domains capture synchronization structure
that raw power band features miss. This feature set is expected to substantially outperform Tier 1.

**Background for executor:**  
The MichaelHills pipeline computes FFT magnitude in 1–47 Hz (intracranial data), then builds correlation
matrices in both time and frequency domains and appends their eigenvalues. For scalp EEG we adjust the
frequency range to 1–40 Hz. The eigenvalues of the correlation matrix are the classical approximation
to what quantum density matrix eigenvalues represent — this connection is central to the quantum
practicality argument.

**Prompt to Claude Code:**

```
Implement the Tier 2 MichaelHills-style correlation eigenvalue feature extractor for the CHB-MIT
EEG seizure detection benchmark.

Feature extraction (per 1-second window):
Step 1 — FFT magnitude:
    fft_mag = np.abs(np.fft.rfft(window, axis=1))  # shape: (n_channels, n_freqs)
    # Keep 1–40 Hz bins only (for 256 Hz sampling: bins 1 through 40)
    fft_mag = fft_mag[:, 1:41]  # shape: (n_channels, 40)
    fft_mag = np.log10(fft_mag + 1e-6)  # log magnitude

Step 2 — Correlation in frequency domain:
    C_freq = np.corrcoef(fft_mag)  # shape: (n_channels, n_channels)
    eigenvalues_freq = np.linalg.eigvalsh(C_freq)  # shape: (n_channels,), ascending order

Step 3 — Correlation in time domain:
    C_time = np.corrcoef(window)  # shape: (n_channels, n_channels)
    eigenvalues_time = np.linalg.eigvalsh(C_time)  # shape: (n_channels,)

Step 4 — Concatenate feature vector:
    features = np.concatenate([
        fft_mag.flatten(),           # n_channels * 40
        eigenvalues_freq,            # n_channels
        eigenvalues_time             # n_channels
    ])

Classifier and CV: same as Tier 1 (RF 3000 trees, leave-one-seizure-out, patient-specific).

Output format: save to results/tier2_correlation_eigenvalue/{patient_id}.json with same schema as Tier 1.
Produce results/tier2_summary.csv.

After running all patients, produce results/tier1_vs_tier2_comparison.csv with columns:
patient_id, tier1_auc, tier2_auc, delta_auc, tier2_wins (bool)

This comparison is a key validation checkpoint. Tier 2 should outperform Tier 1 on most patients.
If it does not, there is likely a preprocessing or feature extraction bug.
```

**Acceptance criteria:**
- Feature vector correctly concatenates FFT log magnitude + frequency eigenvalues + time eigenvalues
- `np.linalg.eigvalsh` used (not `eig`) — eigvalsh is for symmetric matrices, numerically stable
- Results show Tier 2 outperforming Tier 1 on majority of patients (expected based on Kaggle results)
- Comparison CSV produced

---

## Prompt 4 — Tier 3 Baseline: Riemannian Covariance (pyRiemann)

**Context:** This is the best-known classical method for EEG classification and the direct geometric
analog of what quantum circuits do naturally. Barachant's approach won the 2016 Melbourne seizure
prediction competition using this method. Covariance matrices of EEG are Symmetric Positive Definite
(SPD) matrices. The correct distance metric on SPD matrices is the affine-invariant Riemannian metric,
not Euclidean distance. Quantum density matrices are also SPD (with trace=1). This is the geometric
bridge that supports the quantum practicality claim.

**Background for executor:**  
Install: `pip install pyriemann`. The MDM (Minimum Distance to Mean) classifier computes the Riemannian
mean of each class's covariance matrices, then classifies by nearest mean under the geodesic distance.
This is the baseline to beat with quantum. If QPNN cannot outperform Riemannian MDM on any patients,
the quantum practicality claim requires rethinking.

**Prompt to Claude Code:**

```
Implement the Tier 3 Riemannian covariance baseline for the CHB-MIT EEG seizure detection benchmark.
This is the strongest classical comparator and maps directly to the quantum geometry argument.

Dependencies: pip install pyriemann mne scipy numpy

Preprocessing for this tier uses 30-second windows (not 1-second):
- Same bandpass/notch as previous tiers
- Window: 30 seconds non-overlapping (7680 samples at 256 Hz)
- No per-channel normalization before covariance (normalization is implicit in the 
  Riemannian metric — applying z-score beforehand changes the covariance structure)

Feature extraction (per 30-second window):
    from pyriemann.estimation import Covariances
    # Compute regularized covariance matrix per window
    # shape input: (n_windows, n_channels, n_samples)
    cov_matrices = Covariances(estimator='lwf').transform(windows)
    # lwf = Ledoit-Wolf-Schäfer regularization — critical for rank-deficient EEG
    # Output shape: (n_windows, n_channels, n_channels) — each is an SPD matrix

Classifier:
    from pyriemann.classification import MDM
    from pyriemann.utils.mean import mean_riemann
    clf = MDM(metric='riemann')  # Minimum Distance to Mean under Riemannian metric
    # Patient-specific, leave-one-seizure-out CV (same scheme as Tiers 1 and 2)
    # Note: MDM needs at least 2 seizures per class for a mean — skip patients with <2 seizures

Also implement a second variant using Riemannian tangent space + LDA:
    from pyriemann.tangentspace import TangentSpace
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    pipe = Pipeline([
        ('cov', Covariances(estimator='lwf')),
        ('ts', TangentSpace(metric='riemann')),
        ('lda', LinearDiscriminantAnalysis())
    ])
    # Tangent space projection maps SPD manifold to flat vector space at reference point
    # LDA then operates in this locally-Euclidean space

Output: 
- results/tier3_riemannian_mdm/{patient_id}.json
- results/tier3_riemannian_ts_lda/{patient_id}.json
- results/tier3_summary.csv
- results/full_comparison.csv: patient_id, tier1_auc, tier2_auc, tier3_mdm_auc, tier3_ts_lda_auc

Add a column in full_comparison.csv: best_classical_tier (which tier won per patient).
```

**Acceptance criteria:**
- Ledoit-Wolf regularization used (`lwf`), not raw sample covariance (ill-conditioned for EEG)
- Both MDM and tangent space + LDA variants implemented
- full_comparison.csv generated across all tiers
- Patients with insufficient seizures for CV flagged, not silently skipped

---

## Prompt 5 — MAX Aggregation and Sub-Window Analysis

**Context:** Barachant's analysis and prior QPNN work both show that seizure precursors appear as
transient, high-confidence signals within a window — not sustained patterns. MAX aggregation across
sub-windows consistently outperforms MEAN aggregation. This prompt adds sub-window aggregation variants
to all tiers and validates the MAX vs MEAN finding on CHB-MIT.

**Background for executor:**  
For each 30-second segment, compute predictions on 1-second sub-windows, then aggregate to get the
segment-level prediction. Barachant's post-competition analysis showed: mean probability AUC 0.674,
std of probability AUC 0.805. The MAX finding validates the hypothesis that seizure-predictive
patterns are transient and spatially local in time. This also directly informs how quantum
sub-window predictions should be aggregated in the QPNN pipeline.

**Prompt to Claude Code:**

```
Add sub-window MAX aggregation analysis to the Tier 2 and Tier 3 classifiers.

The goal is to compare three aggregation strategies applied to classifier predictions
on 1-second sub-windows within each 30-second segment:
  - MEAN: mean probability across sub-windows
  - MAX: maximum probability across sub-windows  
  - STD: standard deviation of probability across sub-windows (Barachant's finding)

Implementation:

For Tier 2 (correlation eigenvalue):
1. Train classifier on 1-second windows (same as Prompt 3)
2. At prediction time, for each 30-second segment:
   - Split into 30 non-overlapping 1-second sub-windows
   - Get predicted seizure probability for each sub-window
   - Compute mean, max, and std of those 30 probabilities
   - Each gives a segment-level seizure score
3. Compute AUC for each aggregation strategy

For Tier 3 (Riemannian):
1. Train MDM on 30-second covariance matrices (as in Prompt 4)
2. Additionally train on 1-second covariance matrices, then apply same MAX/MEAN/STD aggregation

Output per patient per tier (append to existing result JSONs):
{
  "aggregation_comparison": {
    "mean_auc": 0.XX,
    "max_auc": 0.XX,
    "std_auc": 0.XX
  }
}

Produce: results/aggregation_analysis.csv with columns:
patient_id, tier, mean_auc, max_auc, std_auc, max_wins (bool: max_auc > mean_auc)

Expected outcome: MAX outperforms MEAN on majority of patients. If this is not the case on
CHB-MIT, it is a meaningful negative finding — log it explicitly and note that CHB-MIT
seizure detection (ictal vs interictal) may differ from seizure prediction (preictal vs interictal)
in this regard, since the signal is active during detection rather than precursory.
```

**Acceptance criteria:**
- All three aggregation strategies implemented and compared
- Results logged per patient, per tier
- Aggregation comparison CSV produced
- Negative findings (if MAX does not win) explicitly noted, not suppressed

---

## Prompt 6 — Patient-Specific Calibration Analysis

**Context:** Prior QPNN results showed that seizure state direction varies by patient — some patients
show seizure states inside the Mandelbrot set boundary, others outside. This is a patient-specific
calibration issue. The classical baselines should be examined for the same phenomenon: do some patients
systematically show inverted class separability (i.e., the classifier trained in standard orientation
gets AUC < 0.5, indicating it is consistently wrong in the correct direction)?

**Background for executor:**  
AUC < 0.5 on a binary classifier is informative — it means the model is systematically placing the
wrong class on the wrong side of the decision boundary. For a well-calibrated model, AUC = 1 - actual_AUC
when you flip the class labels. This is the classical analog of the patient-specific Mandelbrot
boundary direction in the quantum model. Documenting this in the classical baseline strengthens the
parallel and provides grounding for the personalized quantum model argument.

**Prompt to Claude Code:**

```
Add a patient-specific calibration analysis pass over all tier results.

For each patient and each tier:
1. Check if mean CV AUC < 0.5
2. If so, flag this patient as "inverted" — the classifier is systematically more confident
   in the wrong direction. Compute corrected_auc = 1 - raw_auc.
3. Also compute the "calibrated AUC" by flipping predicted probabilities (1 - p) for
   inverted patients, then recomputing AUC. This should give corrected_auc >= 0.5.

Produce: results/calibration_analysis.csv with columns:
patient_id, tier, raw_auc, is_inverted, calibrated_auc, 
calibration_direction (str: "standard" or "inverted")

After all tiers:
Produce: results/calibration_summary.json
{
  "n_patients_total": 23,
  "n_inverted_any_tier": X,
  "patients_inverted": {
    "chb04": ["tier1", "tier3_mdm"],
    ...
  },
  "cross_tier_consistency": "X of Y inverted patients are inverted consistently across all tiers"
}

Cross-tier consistency is important: if a patient is inverted in Tier 1 but not Tier 3, it
suggests the inversion is feature-dependent, not a fundamental property of that patient's seizure
signature. If a patient is inverted across all tiers, it is likely a fundamental calibration property
of that patient — exactly the analog of the Mandelbrot boundary direction in QPNN.
```

**Acceptance criteria:**
- All patients checked for AUC < 0.5 across all tiers
- Calibration direction documented per patient per tier
- Cross-tier consistency analysis produced
- calibration_summary.json created

---

## Prompt 7 — Final Comparison Report and Quantum Readiness Assessment

**Context:** With all three classical tiers benchmarked, the quantum readiness assessment determines
which patients are good quantum experiment candidates, which tier to use as the primary comparator
for publication, and what the quantum practicality claim must beat to be defensible.

**Prompt to Claude Code:**

```
Generate the final classical baseline comparison report.

Inputs: all results/ CSVs and JSONs from Prompts 2–6.

Produce: results/classical_baseline_report.md

Structure:
1. Executive Summary: best classical method overall (by mean AUC across patients), 
   number of patients where each tier wins, overall performance distribution.

2. Per-Tier Summary Table:
   | Patient | Tier1 AUC | Tier2 AUC | Tier3 MDM | Tier3 TS-LDA | Best Classical | Inverted? |
   One row per patient, sorted by Best Classical AUC descending.

3. Aggregation Analysis Summary: 
   Across all patients and tiers — does MAX aggregation outperform MEAN? Report as: 
   "MAX outperformed MEAN in X/Y patient-tier pairs (Z%)"

4. Patient-Specific Calibration Summary:
   List inverted patients, note cross-tier consistency.

5. Quantum Candidate Assessment:
   For each patient, assess quantum experiment priority:
   - HIGH: Best classical AUC is 0.65–0.85. Too easy (>0.90) wastes quantum runs. 
     Too hard (<0.60) may be noise-dominated and uninformative.
   - MEDIUM: Best classical AUC 0.55–0.65 (harder problem, more interesting if quantum improves it)
   - LOW: AUC < 0.55 after calibration (possibly noise-dominated, quantum won't help)
   Output as: results/quantum_candidate_list.csv with columns:
   patient_id, best_classical_auc, quantum_priority, calibration_direction, notes

6. Minimum bar for quantum practicality claim:
   State explicitly what AUC the QPNN must achieve per patient to claim:
   (a) parity with best classical
   (b) statistically significant improvement (note that with small seizure counts, 
       this requires careful treatment — report effect sizes, not just p-values)

Save quantum_candidate_list.csv separately for direct use in experimental planning.
```

**Acceptance criteria:**
- Full comparison table in markdown report
- Quantum candidate list CSV with priority rankings
- Explicit statement of AUC threshold required for quantum practicality claim
- Report is self-contained — a reader unfamiliar with this project can understand the baseline state

---

## Execution Notes

**Order:** Prompts must be run sequentially — each builds on prior outputs.

**Environment assumptions:** Python 3.9+, CHB-MIT dataset downloaded and accessible at a configured
data root path, pyriemann, scipy, scikit-learn, numpy, mne installed.

**CHB-MIT data path:** Set `DATA_ROOT` environment variable or edit config. 
Patient directories named `chb01/` through `chb24/` (chb04 is missing in standard download).

**IBM Quantum / QPNN integration:** The quantum_candidate_list.csv from Prompt 7 feeds directly
into the four-phase experimental plan. Only HIGH and MEDIUM priority patients should be run on
IBM hardware — LOW priority patients consume quantum compute budget without generating informative
comparisons.

**Repository structure expected after completion:**
```
results/
├── tier1_fft_power/          # per-patient JSONs
├── tier2_correlation_eigenvalue/
├── tier3_riemannian_mdm/
├── tier3_riemannian_ts_lda/
├── tier1_summary.csv
├── tier2_summary.csv
├── tier3_summary.csv
├── tier1_vs_tier2_comparison.csv
├── full_comparison.csv
├── aggregation_analysis.csv
├── calibration_analysis.csv
├── calibration_summary.json
├── classical_baseline_report.md
└── quantum_candidate_list.csv
```
