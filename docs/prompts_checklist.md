# QDNU Prompts Checklist

Last updated: 2026-02-22

## Classical Baseline Phase (PROMPT 001–006)

### PROMPT 001–005 — Classical Baselines

- **Status**: DONE
- **Output**: Tier 1/2/3 baselines, aggregation analysis, calibration analysis
- **Key Result**: Tier 2 MAX correlation eigenvalues = 0.7234 AUC (classical ceiling)

### PROMPT 006 — CH8 Hardware Validation (PLV)

- **Status**: DONE
- **Script**: `scripts/quantum_heron_validation.py`
- **Output**: `results/hardware_validation/ch8_heron_loso_results.json`
- **Backend**: IBM Torino (Heron r2)
- **Encoding**: PLV_theta_alpha
- **Result**: AUC 0.5110 (raw), 0.6365 (calibrated)
- **Cost**: ~63s (9s/subject × 7 subjects)

---

## Encoding & Analysis Phase

### PROMPT 007 — Encoding Audit

- **Status**: DONE
- **Script**: `scripts/encoding_audit.py`
- **Output**: `results/hardware_validation/encoding_audit.json`
- **Key Finding**: 8 result files audited, V2_band_power best in simulation (0.534), PLV best on hardware

### PROMPT 008 — Simulation Scaling Curve

- **Status**: DONE
- **Script**: `scripts/simulation_scaling.py`
- **Output**: `results/projection/simulation_scaling.json`
- **Results**:
  - CH4: AUC 0.5222 (9 qubits)
  - CH8: AUC 0.5344 (17 qubits)
  - CH12: AUC 0.6436 (25 qubits)
  - CH16: INFEASIBLE (33 qubits, requires 128GB RAM)

### PROMPT 009 — CH4 Hardware (V2 band_power)

- **Status**: DONE
- **Script**: `scripts/hardware_scaling_ch4.py`
- **Output**: `results/projection/hardware_scaling.json` (ch4 section)
- **Backend**: IBM Torino
- **Encoding**: V2_band_power
- **Result**: AUC 0.5000 (degenerate — all subjects at chance)
- **Cost**: ~63s
- **Finding**: 4 channels below representational floor for this encoding

### PROMPT 009B — CH8 V2 band_power (sub-windows)

- **Status**: ABANDONED
- **Script**: `scripts/hardware_scaling_v2bp_ch8.py`
- **Partial Result**: 2/7 subjects complete, both AUC 0.5000 (degenerate)
- **Note**: Used 30 sub-windows = ~228s/subject, exceeded 600s credit limit
- **Finding**: V2_band_power dead on hardware — best simulation encoding produces degenerate output on real QPU. Second instance of simulation-hardware inversion.

---

## Polarity Analysis Phase

### PROMPT 011 — Per-Patient Quantum Performance with Polarity Detection

- **Status**: DONE
- **Script**: `scripts/quantum_patient_profiles.py`
- **Output**:
  - `results/patient_analysis/quantum_patient_profiles.json`
  - `results/patient_analysis/quantum_patient_profiles.md`
- **Key Results**:
  - Raw overall HW AUC: 0.5241
  - Calibrated overall AUC: 0.6365
  - Inverted patients: 3/7 (chb03, chb11, chb21)
  - Calibration lift: +0.1124
  - chb11: raw 0.283 → calibrated 0.717 (STRONG-INVERTED)
  - Mandelbrot concordance: 50% (inconclusive — dataset mismatch, 4 overlap patients)

### PROMPT 012 — Calibrated Gap Inputs for Projection

- **Status**: DONE
- **Script**: `scripts/calibrated_gap_inputs.py`
- **Output**: `results/calibration/calibrated_gap_inputs.json`
- **Key Results**:
  - Raw gap to classical: 0.1993
  - Calibrated gap to classical: 0.0869
  - Gap reduction from polarity calibration: 56.4%

---

## Projection & Visualization Phase

### PROMPT 010 — Hardware Readiness Projection

- **Status**: DONE
- **Script**: `scripts/hardware_projection.py`
- **Output**:
  - `results/projection/hardware_projection.json`
  - `results/projection/hardware_projection.md`
- **Key Finding**: Calibrated HW AUC 0.6365, gap to clinical threshold (0.70) is 0.0635
- **Note**: Simulation-hardware inversion documented — hardware beats simulation for PLV encoding

### VIZ-PREP — Visualization Preparation

- **Status**: DONE
- **Script**: `scripts/viz_hardware_scaling.py`
- **Output**: `figures/hardware_scaling/`
  - `fig1_scaling_curve.png/.pdf`
  - `fig2_gap_analysis.png/.pdf`
  - `fig3_polarity_calibration.png/.pdf`
  - `fig4_classical_quantum.png/.pdf`

### VIZ — 4D Circuit State Evolution (Three.js)

- **Status**: TODO
- **Depends on**: VIZ-PREP trajectory data
- **Prompt**: `visualization_prompt.md` (in project knowledge)
- **Purpose**: Interactive gate-by-gate animation showing polarity inversion on SPD manifold

---

## Window Size Analysis Phase

### PROMPT 015 — Window Size Impact on Classical Baselines

- **Status**: DONE
- **Script**: `scripts/window_size_experiment.py`
- **Output**:
  - `results/window_analysis/window_size_impact.json`
  - `results/window_analysis/window_size_report.md`
- **Key Results**:
  - Best Tier 2 MAX: 0.8726 AUC at 20s windows (vs 0.80 at 1s)
  - Condition number: drops from 67.3 (1s) to 47.4 (30s)
  - Tier 3 Riemannian: modest improvement 0.44 → 0.49 with longer windows
  - **Projected classical ceiling at 20s: 0.87 AUC** (vs current 0.72 at 1.95s)
- **Finding**: Window size is a major AUC bottleneck. +0.15 AUC available from 20s windows alone.
- **Implication**: Quantum encoding may benefit similarly from longer windows.

### PROMPT 016 — Polarity Calibration on Tier 3 Riemannian Results

- **Status**: DONE
- **Script**: `scripts/tier3_polarity_calibration.py`
- **Output**:
  - `results/window_analysis/tier3_polarity_calibration.json`
  - `results/window_analysis/tier3_polarity_report.md`
- **Key Results**:
  - Tier 3 raw: 0.44–0.57 across window sizes
  - Tier 3 calibrated: 0.69–0.72 across window sizes (+0.12–0.19 lift)
  - **Best calibrated: 0.724 AUC at 30s** (matches classical Tier 2 ceiling)
  - Tier 2 shows near-zero calibration lift (eigenvalue sorting is polarity-invariant)
  - Tier 3/Quantum concordance: 5/7 (71%)
  - Polarity consistency across windows: 7/7 (100%)
- **Finding**: Polarity inversion is THE explanation for Riemannian underperformance in cross-patient evaluation.
- **Implication**: The polarity discovered via quantum hardware is not quantum-specific — it's a geometric property of the SPD manifold visible in any direction-preserving method.

### PROMPT 017-SIM — Quantum PLV CH8, 20s Windows, Statevector Simulation

- **Status**: DONE
- **Script**: `scripts/quantum_20s_simulation.py`
- **Output**: `results/window_analysis/quantum_20s_simulation.json`
- **Key Results**:
  - Simulation raw AUC: 0.386 (DECREASED from 0.460 at 1.95s)
  - Simulation calibrated AUC: 0.630
  - Inverted subjects: 6/7 (vs 3/7 on hardware at 1.95s)
  - Window effect on simulation: -0.073 (longer windows make it WORSE)
- **Finding**: Simulation-hardware inversion is FUNDAMENTAL to encoding geometry, NOT window-dependent
- **Prediction for March 22**: Hardware at 20s should still beat simulation (~0.63 calibrated)
- **Polarity concordance**:
  - Simulation 20s vs Hardware 1.95s: 4/7 (57%)
  - Simulation 20s vs Tier 3 20s: 4/7 (57%)

---

## Scheduled for March 22, 2026+ (Credits Reset)

### Execution Priority

| Priority | Prompt | Description | Qubits | Est. Cost | Running Total |
|----------|--------|-------------|--------|-----------|---------------|
| 1 | **017** | Quantum PLV CH8, 20s windows | 17 | ~45s | 45s |
| 2 | **013** | CH12 PLV hardware | 25 | ~63s | 108s |
| 3 | **014** | CH16 PLV hardware | 33 | ~63s | 171s |
| — | Buffer | — | — | — | 429s remaining |

Total: ~171s out of 600s budget. Leaves 429s for reruns/debugging.

### PROMPT 017 — Quantum PLV CH8 with 20-Second Windows (HIGHEST PRIORITY)

- **Status**: READY (waiting for credits)
- **Script**: `scripts/quantum_20s_hardware.py`
- **Output**: `results/window_analysis/quantum_20s_hardware.json`
- **Backend**: IBM Torino
- **Encoding**: PLV_theta_alpha (SAME as PROMPT 006)
- **Window**: 20 seconds (was 1.95s)
- **Qubits**: 17
- **Cost estimate**: ~45s (faster — 1 circuit/segment instead of ~15)
- **Rationale**: PROMPT 015 showed 20s windows improve Tier 2 from 0.72→0.87. This tests whether quantum encoding benefits similarly.
- **Key question**: Does calibrated quantum at 20s exceed 0.72 (competitive with classical)?

### PROMPT 013 — CH12 Hardware (PLV segment-level)

- **Status**: READY (waiting for credits)
- **Script**: `scripts/hardware_scaling_ch12_plv.py`
- **Backend**: IBM Torino
- **Encoding**: PLV_theta_alpha (segment-level, NOT sub-windows)
- **Qubits**: 25
- **Cost estimate**: ~63s (9s/subject × 7 subjects)

### PROMPT 014 — CH16 Hardware (PLV segment-level)

- **Status**: READY (waiting for credits)
- **Script**: `scripts/hardware_scaling_ch16_plv.py`
- **Backend**: IBM Torino
- **Encoding**: PLV_theta_alpha (segment-level, NOT sub-windows)
- **Qubits**: 33
- **Cost estimate**: ~63s (9s/subject × 7 subjects)
- **Note**: CH16 simulation infeasible (128GB RAM), so hardware is the ONLY way to get this data point

### PROMPT 010-R — Rerun Projection with 4-Point Curve

- **Status**: PENDING (after PROMPT 013 + 014 + 017 complete)
- **Purpose**: Generate 4-point hardware scaling curve + window size comparison
- **Cost**: Free (local computation)

---

## IBM Quantum Budget

| Window           | Limit | Used | Remaining            |
|------------------|-------|------|----------------------|
| Feb 22 – Mar 22  | 600s  | 603s | 0s (over by 3s)      |
| Mar 22 – Apr 22  | 600s  | 0s   | 600s                 |

### Cost per run (segment-level PLV)

- Per subject: ~9s
- 7 subjects: ~63s
- CH12 + CH16 in one window: ~126s total (fits easily in 600s)

### Cost per run (sub-window — DO NOT USE)

- Per subject: ~228s
- 7 subjects: ~1596s (exceeds single window)
- Lesson learned from PROMPT 009B

---

## Key Results Summary

| Metric                         | Value                        | Source                       |
|--------------------------------|------------------------------|------------------------------|
| Classical ceiling (CH8)        | 0.7234                       | Tier 2 MAX XGBoost           |
| Clinical threshold             | ≥ 0.70                       | Target                       |
| Simulation best (CH12)         | 0.6436                       | PROMPT 008                   |
| Hardware raw (CH8 PLV)         | 0.5241                       | PROMPT 006                   |
| Hardware calibrated (CH8 PLV)  | 0.6365                       | PROMPT 011 polarity calibration |
| Gap to classical (raw)         | 0.1993                       | PROMPT 012                   |
| Gap to classical (calibrated)  | 0.0869                       | PROMPT 012                   |
| Gap to clinical threshold      | 0.0635                       | PROMPT 010                   |
| Polarity calibration lift      | +0.1124 (56.4% gap reduction)| PROMPT 011                   |

---

## Key Findings

### 1. Simulation-Hardware Inversion

- PLV encoding: simulation AUC 0.460, hardware calibrated 0.637
- V2 encoding: simulation AUC 0.534, hardware AUC 0.500 (degenerate)
- Hardware outperforms simulation for PLV; simulation ranks V2 above PLV but hardware ranks opposite
- Implication: encoding selection MUST be validated on hardware, not simulation

### 2. Patient-Specific Polarity

- 3/7 patients require polarity inversion (sign bit per patient)
- chb11: raw 0.283 → calibrated 0.717 (strongest discrimination after calibration)
- Quantum circuit makes manifold direction an observable; classical classifiers hide it

### 3. V2 Band Power Hardware Degeneracy

- Best simulation encoding produces exactly 0.500 AUC on hardware at both CH4 and CH8
- Circuit output is degenerate (uniform/symmetric) regardless of input
- Do not spend credits on V2 hardware runs

### 4. Polarity Explains Tier 3 Riemannian Underperformance

- Tier 3 raw AUC ~0.44–0.57 across all window sizes
- Tier 3 calibrated AUC ~0.69–0.72 (+0.12–0.19 lift)
- Best calibrated Tier 3: 0.724 at 30s windows (matches Tier 2 ceiling)
- Tier 2 shows zero calibration lift — eigenvalue sorting is polarity-invariant
- 100% of subjects have consistent polarity across ALL window sizes
- 71% concordance between Tier 3 polarity and quantum polarity
- Polarity is a geometric property of the SPD manifold, not a quantum-specific phenomenon

---

## Encoding Reference

| Encoding        | Description                      | Hardware Status                      |
|-----------------|----------------------------------|--------------------------------------|
| PLV_theta_alpha | Phase-locking value, 4-13 Hz     | Works on hardware (primary encoding) |
| V2_band_power   | Band power ratios across 6 bands | Degenerate on hardware               |
| V1_PN_dynamics  | Raw PN neuron (a,b,c)            | Untested on hardware                 |
| V3_PLV variants | PLV at different frequency bands | Untested on hardware                 |

---

## Priority Subjects

Used across all LOSO CV runs:

```text
chb01, chb03, chb05, chb07, chb11, chb14, chb21
```

| Subject | Classical AUC | HW Raw | HW Calibrated | Polarity |
|---------|---------------|--------|---------------|----------|
| chb01   | 0.686         | 0.686  | 0.686         | standard |
| chb03   | —             | 0.436  | 0.564         | inverted |
| chb05   | —             | 0.610  | 0.610         | standard |
| chb07   | —             | 0.667  | 0.667         | standard |
| chb11   | —             | 0.283  | 0.717         | inverted |
| chb14   | —             | 0.600  | 0.600         | standard |
| chb21   | —             | 0.387  | 0.612         | inverted |

---

## Paper Drafts (in project knowledge)

- `abstract_draft.md` — Simulation-hardware inversion paper abstract
- `visualization_prompt.md` — 4D circuit state evolution visualization spec
