# State of the work — what we have, and the arc

**Date:** 2026-09-07
**Purpose:** one page that says what exists, what it adds up to, and what is open. Written to be read cold.

---

## The arc, in five moves

**1. A circuit motivated by a neuron model.** Paper 1 built the A-Gate: a quantum circuit whose rotations are set by EEG-derived features, motivated by the excitatory/inhibitory (PN) neuron. It ran on IBM hardware. It underperformed a classical baseline (0.637 vs 0.742 AUC). It reported that honestly and named the encoding as the bottleneck. *Published, Discover Quantum Science.*

**2. The circuit exposed a sign.** Some patients came out with AUC far below chance on a circuit with no trained parameters. Paper 2 showed why: the circuit measures in a fixed basis, and a fixed basis cannot absorb a sign flip. The ictal shift on the covariance manifold points a patient-specific way. Flexible classifiers hide this; direction-preserving ones expose it. Validated classically across 22 patients. *Submitted.*

**3. The circuit was found to be classically replaceable, and the sign was found to be simpler than described.** This week. Every gate after the encoding is a fixed unitary, so anything it measures is a transformed observable on three classical numbers per channel (`AGATE_OBSERVABLE_NOTES.md`). One of the three numbers, the absolute phase, carries nothing — it is a gauge, confirmed on 32,000 windows across two datasets (038, 041). The robust part of polarity is the sign of the ictal power change; the Tier-3 inverted set was axis-dependent, exactly as Paper 2's mechanism predicts, and moves with the reference (039). Within a patient, seizures shift the covariance coherently (037).

**4. The object was replaced.** The trace-normalised covariance is a density matrix — by definition, not analogy. Most of the ictal displacement is total power, which the state cannot hold (038). So the framework becomes: the trace travels as a classical scalar; the state carries the shape; measure the state. That is DSP-000, and it now has a reason. The state has at least one real within-patient observable: its entropy changes during seizures with a patient-specific sign — 4 patients purify, 10 mix, 8 undetermined (042).

**5. The neuron model was rebuilt as a two-level system.** The PN equations as written have no coupling and saturate at EEG amplitude. The minimal coupled E/I unit is a 2×2 non-Hermitian generator with two knobs: net gain (sets the trace) and coupling-vs-asymmetry (sets the purity). Its balanced oscillatory state *is* a complex amplitude — envelope and phase — which is what the Hilbert encoding read all along, and why the absolute phase is a gauge. Derived, numerically verified, tested on eye closure in 109 subjects: trace up, entropy not down, phase uniform — all as the derivation says once an extra intuition of mine was removed (`EI_TWO_LEVEL_NOTES.md`, 041).

---

## What is established (and where)

| claim | evidence |
|---|---|
| The A-Gate is a fixed basis change on classical features; it cannot answer "why quantum" | `AGATE_OBSERVABLE_NOTES.md` §3–4 |
| The absolute phase parameter is a gauge and carries no information | 038 (CHB-MIT, resultant 0.044); 041 (PhysioNet, 0.010, n = 13,080) |
| Per-patient ictal shift directions are coherent across seizures (c ≈ 1) | 037 |
| Most of the ictal displacement is uniform power gain (median 83 %) | 038 Q1 |
| The robust, reference-free polarity is the power sign; the pooled-class inverted set is axis-dependent | 039 |
| Alignment, not non-commutativity, explains axis-dependent polarity | 040 (negative result, mechanism corrected) |
| The state's entropy changes during seizures with a stable patient-specific sign | 042 Part A |
| E/I unit = two-level open system; balanced oscillation = analytic signal; oscillation is mixed, mode separation is pure | `EI_TWO_LEVEL_NOTES.md` §2–4; 041 |
| Eye closure raises the trace and does not purify the state | 041 (109 subjects) |

## What is open

- Whether the E/I generator's regime sets a patient's entropy sign. Underpowered null on the aperiodic-slope proxy (042). The right test is within-seizure and time-resolved; designed, not run.
- Whether a state-based observable derived from theory (DSP-000 Phase 2) says anything a covariance measure doesn't already name. Purity is the first candidate.
- Whether cortex is the minimal E/I unit. Adopted from the standard two-population picture, not derived.

## What "why quantum" and "why EEG" now say

*Why quantum:* the formalism is rigid where classical ML is flexible — fixed measurement basis, trace one, global phase unobservable, known unitaries — and each of those rigidities produced a discovery here (the sign, the trace/shape split, the dead parameter, the replaceable circuit). Not speed. Formulation.

*Why EEG:* a balanced E/I unit is a complex amplitude, a field of them is a mixed state, and EEG is the measurement of that field. The formalism fits because the system has the structure.

## What the dissertation is, in one sentence

A quantum-information geometry for multichannel EEG covariance states — the density-matrix identification, the E/I unit as its generator, diagnostics derived from both, and their verification on public data and by direct state preparation — with the hardware feature map as the history that led there.

## What is next

Not another test. The DSP-000 Phase 1 reading, and a `theory_notes.md` in your own words. Tests resume when a claim in that draft needs one.
