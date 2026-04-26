# Hardware Projection Report (PROMPT 031)
Generated: 2026-03-28T16:17:48.778343

## 1. Channel Scaling Curve (Hardware)

| Channels | Qubits | Window | Raw AUC | Oracle-cal. AUC | N_inv | Status |
|----------|--------|--------|---------|-----------------|-------|--------|
| 4 | 9 | 1.95s | 0.5000 | 0.5000 | — | degenerate |
| 8 | 17 | 1.95s | 0.5241 | 0.6365 | 3 | available |
| 8 | 17 | 20.0s | 0.4035 | 0.6046 | 6 | available |
| 12 | 25 | 30.0s | 0.5220 | 0.5652 | 4 | available |
| 16 | 33 | 30.0s | 0.5119 | 0.5119 | 0 | noise_floor |

## 2. Window Size Scaling (CH8)

| Window | Raw AUC | Oracle-cal. AUC | N_inverted |
|--------|---------|-----------------|------------|
| 1.95s | 0.5241 | 0.6365 | 3 |
| 20.0s | 0.4035 | 0.6046 | 6 |

## 3. Simulation vs Hardware

| Config | Sim AUC | HW Raw AUC | HW Oracle-cal. | Sim-HW Gap |
|--------|---------|------------|----------------|------------|
| CH4 | 0.5222 | 0.5000 | 0.5000 | +0.0222 |
| CH8 | 0.5344 | 0.5241 | 0.6365 | +0.0103 |
| CH12 | 0.6436 | 0.5220 | 0.5652 | +0.1216 |

## 4. Gap Analysis vs Classical Comparators

| Comparator | AUC | Best HW Oracle-cal. | Gap | Note |
|------------|-----|---------------------|-----|------|
| Tier 2A Eigenvalues (clean-validated) | 0.6831 | 0.6365 | +0.0466 | Only comparator with validated label-free calibration (PROMPT 022) |
| Tier 2A Eigenvalues (oracle) | 0.6892 | 0.6365 | +0.0527 | Oracle calibration for reference |
| Tier 3 Riemannian (oracle) | 0.7011 | 0.6365 | +0.0646 | Oracle only - clean calibration fails (PROMPT 022) |
| Classical Ceiling (336 features) | 0.9520 | 0.6365 | +0.3155 | Not a fair quantum comparator |

Best hardware result (oracle-calibrated): **0.6365** (CH8 1.95s)

## 5. Key Findings

- **CH16 (33q, 246 CZ gates):** Noise floor exceeded. All subjects at chance (0.500).
- **Polarity instability:** 4/7 subjects show inconsistent polarity across configs.
  Only chb21 is consistently inverted; chb03 consistently inverted on A-Gate.
- **Channel scaling:** No improvement from CH8 to CH12 on hardware.
  CH12 oracle-cal (0.565) < CH8 1.95s oracle-cal (0.637).
- **Window scaling:** 20s windows did NOT improve over 1.95s on hardware.
  CH8 20s oracle-cal (0.605) < CH8 1.95s oracle-cal (0.637).
- **Clean calibration gap:** A-Gate best (0.637) still below Tier 2A clean (0.683).

## 6. Polarity Tracking

| Subject | CH8-1.95s | CH8-20s | CH12 | CH16 | Consistent? |
|---------|-----------|---------|------|------|-------------|
| chb01 | standard | standard | inverted | chance | NO |
| chb03 | inverted | inverted | inverted | chance | YES |
| chb05 | standard | inverted | inverted | chance | NO |
| chb07 | standard | inverted | standard | chance | NO |
| chb11 | inverted | inverted | chance | standard | YES |
| chb14 | standard | inverted | standard | chance | NO |
| chb21 | inverted | inverted | inverted | chance | YES |