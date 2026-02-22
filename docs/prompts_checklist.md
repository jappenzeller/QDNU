# QDNU Prompts Checklist

Last updated: 2026-02-22

## Hardware Validation Phase

### PROMPT 006 - CH8 Hardware Validation (PLV)
- **Status**: DONE
- **Script**: `scripts/quantum_heron_validation.py`
- **Output**: `results/hardware_validation/ch8_heron_loso_results.json`
- **Backend**: IBM Torino (Heron r2)
- **Encoding**: PLV_theta_alpha
- **Result**: AUC 0.5110 (raw), 0.6365 (calibrated)
- **Cost**: ~63s (9s/subject x 7 subjects)

### PROMPT 007 - Encoding Audit
- **Status**: DONE
- **Output**: `results/hardware_validation/encoding_audit.json`
- **Purpose**: Verify encoding consistency across hardware runs

### PROMPT 009 - CH4 Hardware (V2 band_power)
- **Status**: DONE
- **Script**: `scripts/hardware_scaling_ch4.py`
- **Output**: `results/projection/hardware_scaling.json` (ch4 section)
- **Backend**: IBM Torino
- **Encoding**: V2_band_power
- **Result**: AUC 0.5000
- **Cost**: ~63s

### PROMPT 009B - CH8 V2 band_power (sub-windows)
- **Status**: CANCELLED (credits exhausted)
- **Script**: `scripts/hardware_scaling_v2bp_ch8.py`
- **Note**: Used 30 sub-windows = ~228s/subject, exceeded 600s limit
- **Lesson**: Use segment-level encoding for hardware runs

---

## Simulation Phase

### PROMPT 008 - Simulation Scaling Curve
- **Status**: DONE
- **Script**: `scripts/simulation_scaling.py`
- **Output**: `results/projection/simulation_scaling.json`
- **Results**:
  - CH4: AUC 0.5222 (9 qubits)
  - CH8: AUC 0.5344 (17 qubits)
  - CH12: AUC 0.6436 (25 qubits)
  - CH16: INFEASIBLE (33 qubits, requires 128GB RAM)

---

## Projection & Visualization Phase

### PROMPT 010 - Hardware Readiness Projection
- **Status**: DONE
- **Script**: `scripts/hardware_projection.py`
- **Output**:
  - `results/projection/hardware_projection.json`
  - `results/projection/hardware_projection.md`
- **Key Finding**: Calibrated HW AUC 0.6365, gap to clinical threshold (0.70) is 0.0635

### VIZ-PREP - Visualization Preparation
- **Status**: DONE
- **Script**: `scripts/viz_hardware_scaling.py`
- **Output**: `figures/hardware_scaling/`
  - `fig1_scaling_curve.png/.pdf`
  - `fig2_gap_analysis.png/.pdf`
  - `fig3_polarity_calibration.png/.pdf`
  - `fig4_classical_quantum.png/.pdf`

---

## Scheduled for March 22, 2026+ (Credits Reset)

### PROMPT 011 - CH12 Hardware (PLV segment-level)
- **Status**: READY (waiting for credits)
- **Script**: `scripts/hardware_scaling_ch12_plv.py`
- **Backend**: IBM Torino
- **Encoding**: PLV_theta_alpha (segment-level)
- **Qubits**: 25
- **Cost estimate**: ~63s (9s/subject x 7 subjects)

### PROMPT 012 - CH16 Hardware (PLV segment-level)
- **Status**: READY (waiting for credits)
- **Script**: `scripts/hardware_scaling_ch16_plv.py`
- **Backend**: IBM Torino
- **Encoding**: PLV_theta_alpha (segment-level)
- **Qubits**: 33
- **Cost estimate**: ~63s (9s/subject x 7 subjects)

### PROMPT 010 (rerun)
- **Status**: PENDING (after CH12/CH16 complete)
- **Purpose**: Generate 4-point hardware scaling curve

---

## IBM Quantum Budget

| Window | Limit | Used | Remaining |
|--------|-------|------|-----------|
| Feb 22 - Mar 22 | 600s | 603s | 0s (over by 3s) |
| Mar 22 - Apr 22 | 600s | 0s | 600s |

### Cost per run (segment-level PLV)
- Per subject: ~9s
- 7 subjects: ~63s
- CH12 + CH16: ~126s total

---

## Key Results Summary

| Metric | Value | Source |
|--------|-------|--------|
| Clinical threshold | AUC >= 0.70 | Target |
| Classical ceiling (CH8) | 0.7234 | XGBoost baseline |
| Simulation best (CH12) | 0.6436 | PROMPT 008 |
| Hardware raw (CH8) | 0.5110 | PROMPT 006 |
| Hardware calibrated (CH8) | 0.6365 | Polarity calibration |
| Gap to threshold | 0.0635 | PROMPT 010 |
| Hardware-simulation gap | ~4% | PROMPT 010 |

---

## Encoding Reference

| Encoding | Parameters | Use Case |
|----------|------------|----------|
| PLV_theta_alpha | Hilbert phase/amplitude, 4-13Hz band | Primary hardware encoding |
| V2_band_power | Band power ratios (delta/theta/alpha/beta/gamma) | Simulation comparison |

---

## Priority Subjects

Used across all LOSO CV runs:
```
chb01, chb03, chb05, chb07, chb11, chb14, chb21
```

Selected for moderate classical AUC (0.65-0.85) - good quantum candidates.
