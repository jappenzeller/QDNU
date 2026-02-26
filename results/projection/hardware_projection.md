# Hardware Readiness Projection — Updated

Generated: 2026-02-22T12:01:35.012747

## Current State (pre-March 22)

| Data Point | Channels | Window | Backend | Calibrated AUC | Status |
|------------|----------|--------|---------|----------------|--------|
| ch4_1.95s_hw | 4 | 1.95s | hardware | 0.5000 | degenerate |
| ch4_1.95s_sim | 4 | 1.95s | simulation | 0.5222 | degenerate |
| ch8_1.95s_hw | 8 | 1.95s | hardware | 0.6365 | available |
| ch8_1.95s_sim | 8 | 1.95s | simulation | 0.5344 | available |
| ch8_20s_sim | 8 | 20.0s | simulation | 0.6302 | available |
| ch8_20s_hw | 8 | 20.0s | hardware | — | pending |
| ch12_1.95s_sim | 12 | 1.95s | simulation | 0.6436 | available |
| ch12_1.95s_hw | 12 | 1.95s | hardware | — | pending |
| ch16_1.95s_sim | 16 | 1.95s | simulation | — | infeasible |
| ch16_1.95s_hw | 16 | 1.95s | hardware | — | pending |

## Gap Analysis (against three comparators)

**Quantum Best**: ch8_1.95s_hw — 0.6365 calibrated AUC

| Comparator | AUC | Quantum Best | Gap | Status |
|------------|-----|--------------|-----|--------|
| Classical Ceiling (336 features) | 0.952 | 0.637 | -0.316 | - below |
| Riemannian Calibrated (36 features) | 0.659 | 0.637 | -0.022 | ~ nearly_there |
| Eigenvalues Only (8 features) | 0.628 | 0.637 | +0.009 | + EXCEEDED |

## Channel Scaling Curve

Fixed window: 1.95s

| Channels | Qubits | Raw AUC | Calibrated AUC | Status |
|----------|--------|---------|----------------|--------|
| 4 | 9 | 0.5000 | 0.5000 | degenerate |
| 8 | 17 | 0.5110 | 0.6365 | available |
| 12 | 25 | — | — | pending |
| 16 | 33 | — | — | pending |

## Window Size Impact

Fixed channels: 8

| Window | Sim Raw | Sim Cal | HW Raw | HW Cal | HW Status |
|--------|---------|---------|--------|--------|-----------|
| 1.95s | 0.534 | 0.534 | 0.511 | 0.637 | available |
| 20.0s | 0.386 | 0.630 | — | — | pending |

## Simulation Reliability Assessment

- **Simulation underestimates hardware (PLV 1.95s)**
  - Gap: -0.1021
  - Hardware with calibration beats simulation

- **Simulation worsens with longer windows**
  - Raw AUC drops, but calibration recovers

- **Encoding mismatch between sim and hardware**
  - Direct gap comparison is not reliable

**Recommendation**: Do not use simulation for encoding selection or performance projection. Hardware with calibration is the ground truth.

## Polarity Tracking

**Quantum Hardware Points:**

| Point | Channels | Window | N Inverted | Calibration Lift |
|-------|----------|--------|------------|------------------|
| ch4_1.95s_hw | 4 | 1.95s | 0 | — |
| ch8_1.95s_hw | 8 | 1.95s | 3/7 | 0.112 |

**Classical Reference (PROMPT 018, 22 patients):**
- Tier 2A: 10/22 inverted, lift 0.107
- Tier 3: 6/22 inverted, lift 0.123

## Projections (after March 22 data)

Pending data points: 3
- ch8_20s_hw, ch12_1.95s_hw, ch16_1.95s_hw
- Expected: March 22, 2026+

After March 22 hardware data arrives:
1. Run `python scripts/hardware_projection.py` (PROMPT 010-R)
2. 4-point channel scaling curve will be available
3. 2-point window scaling (1.95s vs 20s) will be available
4. Combined CH16 @ 20s projection can be computed

## Key Conclusions

1. Quantum EXCEEDS matched-dimensionality classical (Tier 2A) by +0.009
2. Gap to Tier 3 calibrated is small (-0.022) — may close with 20s windows
3. Gap to Tier 2B (classical ceiling) is large (-0.316) — not closeable on near-term hardware
4. Simulation is unreliable proxy — hardware with calibration is ground truth
