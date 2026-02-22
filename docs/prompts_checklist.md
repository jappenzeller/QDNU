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

## Scheduled for March 22, 2026+ (Credits Reset)

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
- **Status**: PENDING (after PROMPT 013 + 014 complete)
- **Purpose**: Generate 4-point hardware scaling curve (CH4/CH8/CH12/CH16)
- **Cost**: Free (local computation)

---

## IBM Quantum Budget

| Window | Limit | Used | Remaining |
|--------|-------|------|-----------|
| Feb 22 – Mar 22 | 600s | 603s | 0s (over by 3s) |
| Mar 22 – Apr 22 | 600s | 0s | 600s |

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

| Metric | Value | Source |
|--------|-------|--------|
| Classical ceiling (CH8) | 0.7234 | Tier 2 MAX XGBoost |
| Clinical threshold | ≥ 0.70 | Target |
| Simulation best (CH12) | 0.6436 | PROMPT 008 |
| Hardware raw (CH8 PLV) | 0.5241 | PROMPT 006 |
| Hardware calibrated (CH8 PLV) | 0.6365 | PROMPT 011 polarity calibration |
| Gap to classical (raw) | 0.1993 | PROMPT 012 |
| Gap to classical (calibrated) | 0.0869 | PROMPT 012 |
| Gap to clinical threshold | 0.0635 | PROMPT 010 |
| Polarity calibration lift | +0.1124 (56.4% gap reduction) | PROMPT 011 |

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

---

## Encoding Reference

| Encoding | Description | Hardware Status |
|----------|-------------|-----------------|
| PLV_theta_alpha | Phase-locking value, 4-13 Hz | Works on hardware (primary encoding) |
| V2_band_power | Band power ratios across 6 bands | Degenerate on hardware |
| V1_PN_dynamics | Raw PN neuron (a,b,c) | Untested on hardware |
| V3_PLV variants | PLV at different frequency bands | Untested on hardware |

---

## Priority Subjects

Used across all LOSO CV runs:
```
chb01, chb03, chb05, chb07, chb11, chb14, chb21
```

| Subject | Classical AUC | HW Raw | HW Calibrated | Polarity |
|---------|--------------|--------|---------------|----------|
| chb01 | 0.686 | 0.686 | 0.686 | standard |
| chb03 | — | 0.436 | 0.564 | inverted |
| chb05 | — | 0.610 | 0.610 | standard |
| chb07 | — | 0.667 | 0.667 | standard |
| chb11 | — | 0.283 | 0.717 | inverted |
| chb14 | — | 0.600 | 0.600 | standard |
| chb21 | — | 0.387 | 0.612 | inverted |

---

## Paper Drafts (in project knowledge)

- `abstract_draft.md` — Simulation-hardware inversion paper abstract
- `visualization_prompt.md` — 4D circuit state evolution visualization spec
