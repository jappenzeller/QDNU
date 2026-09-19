# The E/I unit as a two-level system — derivation notes

**Date:** 2026-09-06, §6 corrected 2026-09-07 (PROMPT 041)
**Status:** theory note; every numerical claim verified (statevector / Lyapunov / PN simulation); §6.2–6.3 tested on 109 public subjects
**Feeds:** dissertation theory chapter ("why quantum" as formulation); DSP-000 Phase 2 (what the state carries vs the trace); the entropy split found 2026-09-06 (results not yet a PROMPT)
**Related:** `AGATE_OBSERVABLE_NOTES.md` (the circuit), `PROMPT_038_RESULTS.md` (scale vs shape), `PROMPT_039_RESULTS.md` (robust polarity = power sign)

---

## 0. Summary

1. The PN model as implemented (`qdnu/pn_dynamics.py`) has no E–I coupling and saturates at EEG amplitude: for a 20 µV, 10 Hz input over 8 s it returns $(a,b,c) = (0.992, 1.000, 1.000)$. It cannot be the source of a two-level derivation, and it is the reason the hardware pipeline used the Hilbert/PLV encoding instead.
2. The minimal *coupled* E/I unit is a real $2\times2$ linear system whose generator splits into a **net-gain** scalar $s$ and a **traceless** part with two parameters, rate-asymmetry $g$ and coupling $\kappa$. Eigenvalues $s \pm \sqrt{g^2-\kappa^2}$; exceptional point at $\kappa = g$.
3. In the balanced oscillatory regime ($\kappa > g$, $s<0$) the unit's state is a complex amplitude rotating at $\sqrt{\kappa^2-g^2}$ with envelope $e^{st}$. **The Hilbert-transform encoding (envelope, phase) reads exactly this state.** PLV between channels is phase coherence between two such units; the absolute phase $b$ is a gauge and is unobservable — which 038 found empirically.
4. The unit's windowed covariance has trace controlled by $s$ and purity controlled by $\kappa/g$ (monotone: oscillatory → mixed, mode-separated → pure). These map onto 038's scale/shape split and onto the ictal entropy split (15 patients more mixed, 7 purer).

---

## 1. The PN model, as written, and what it does

`qdnu/pn_dynamics.py`, mode `clamp` (default):

$$
\frac{da}{dt} = -\lambda_a a + f(t)(1-a), \qquad
\frac{db}{dt} = f(t)(1-b), \qquad
\frac{dc}{dt} = +\lambda_c c + f(t)(1-c)
$$

with $f(t) = |x(t)|$ the rectified EEG sample and $a,c \in [0,1]$, $b \in [0,2\pi]$ clipped.

Observations:

- No term couples $a$ and $c$. The three equations are independent first-order filters of the same input. There is no E→I or I→E interaction in the dynamics; the "bidirectional coupling" of Paper 1 exists only in the circuit ($CR_y$, $CR_z$).
- For $f \ge 0$ and $c \in [0,1]$, $dc/dt = \lambda_c c + f(1-c) \ge 0$. So $c$ is monotone non-decreasing and saturates at 1. Likewise $b \to 1$ (the clip at $2\pi$ never binds).
- $a$ relaxes to the fixed point $a^* = \bar f/(\lambda_a + \bar f)$ with time constant $1/(\lambda_a+\bar f)$. With $\lambda_a = 0.1$ and $\bar f$ in the tens of µV, $a^* \approx 1$ and the relaxation takes milliseconds.

Verified (`PNDynamics(lambda_a=0.1, lambda_c=0.1, dt=0.001)`, 8 s windows):

| input | $(a, b, c)$ | predicted $a^*$ |
|---|---|---|
| 10 Hz sine, 1 µV | (0.673, 0.729, 0.791) | 0.864 |
| 10 Hz sine, 20 µV | (0.992, 1.000, 1.000) | 0.992 |
| 10 Hz sine, 100 µV | (0.998, 1.000, 1.000) | 0.998 |

At physiological amplitude the PN parameters are $(\approx 1, 1, 1)$ regardless of the signal. **The model carries no information at EEG scale**, and it has no two-level structure to linearize. The derivation below therefore starts from the minimal coupled unit the circuit *assumed*, not from these equations.

---

## 2. The minimal coupled E/I unit

Two population levels, excitatory $e$ and inhibitory $i$. E has net self-rate $\gamma_E$ (recurrent excitation minus leak; may be positive), I decays at $\gamma_I$. E drives I and I suppresses E with the same strength $\kappa$ (unequal strengths are handled in §5). Linearized about the operating point:

$$
\frac{d}{dt}\begin{pmatrix} e \\ i \end{pmatrix}
= J \begin{pmatrix} e \\ i \end{pmatrix} + \xi(t), \qquad
J = \begin{pmatrix} \gamma_E & -\kappa \\ \kappa & -\gamma_I \end{pmatrix}
$$

The opposite signs on the off-diagonal are the biology: excitation one way, inhibition the other.

### 2.1 Trace / traceless split

$$
J = s\,I + M, \qquad
M = \begin{pmatrix} g & -\kappa \\ \kappa & -g \end{pmatrix}, \qquad
s = \frac{\gamma_E-\gamma_I}{2}, \quad g = \frac{\gamma_E+\gamma_I}{2}
$$

- $s$: **net gain** (excitation minus inhibition). Sign of $s$ decides growth vs decay of everything.
- $g$: **rate asymmetry** between the two populations.
- $\kappa$: **coupling**.

$M$ is traceless and, in Pauli notation, $M = g\,\sigma_z + i\kappa\,\sigma_y$ — a real non-Hermitian generator: a Hermitian part ($g\sigma_z$) and an anti-Hermitian part ($i\kappa\sigma_y$). This is the structure of a gain–loss dimer with the roles of Hermitian and anti-Hermitian exchanged relative to the usual PT-symmetric Hamiltonian; the eigenvalue algebra is the same.

### 2.2 Eigenvalues and the exceptional point

$$
\lambda_\pm = s \pm \sqrt{g^2 - \kappa^2}
$$

| regime | condition | eigenvalues | behaviour |
|---|---|---|---|
| oscillatory (balanced) | $\kappa > g$ | $s \pm i\sqrt{\kappa^2-g^2}$ | rotation at $\omega=\sqrt{\kappa^2-g^2}$, envelope $e^{st}$ |
| exceptional point | $\kappa = g$ | $s, s$ (defective) | eigenvectors coalesce |
| mode-separated | $\kappa < g$ | $s \pm \sqrt{g^2-\kappa^2}$, real | two rates; one mode outlives the other |

$s$ never enters the square root. **Stability ($s<0$) and regime ($\kappa$ vs $g$) are independent knobs.**

---

## 3. The balanced oscillatory regime is the analytic signal

Take $g = 0$ (equal rates) for clarity. Then $M = \kappa\begin{pmatrix}0&-1\\1&0\end{pmatrix}$ is the generator of rotations in the $(e,i)$ plane and

$$
e^{Jt} = e^{st}\, R(\kappa t), \qquad R(\theta) = \begin{pmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta\end{pmatrix}
$$

(verified numerically to $10^{-12}$). Write the unit's state as one complex number $z = e + \mathrm{i}\,i$. Then

$$
z(t) = z(0)\, e^{(s + \mathrm{i}\kappa)t}
$$

an envelope $|z(0)|e^{st}$ and a phase advancing at $\kappa$. That is the analytic signal of a narrow-band oscillation. So:

- The **Hilbert envelope** of a band-limited channel is the unit's $|z|$; the **instantaneous phase** is its $\arg z$. The PROMPT 006 encoding (`extract_plv_params`) reads the natural coordinates of the oscillatory E/I unit — which is why it worked where the PN parameters could not.
- **PLV between two channels** is $|\langle e^{\mathrm{i}(\arg z_p - \arg z_q)}\rangle|$: phase coherence between two E/I oscillators. This is the encoding's $c$.
- The **absolute phase** $\arg z$ of one unit is a gauge: nothing observable depends on it, only on phase differences. 038 found the encoding's $b$ (circular-mean phase of one channel) to be uniformly random per window (resultant length 0.044). That is this gauge freedom showing up in data.
- With $g \ne 0$ (still $\kappa > g$) the rotation is elliptical: the I population lags E by $90^\circ$ at $g=0$ and the lag moves with $g$ (verified: $g/\kappa = 0.3 \to 107.5^\circ$, $0.6 \to 126.9^\circ$; amplitude ratio stays 1 for symmetric $\kappa$). The E–I phase lag is the unit's azimuthal Bloch coordinate. Unequal couplings $\kappa_{EI}\ne\kappa_{IE}$ set the amplitude ratio $|i|/|e|$, the polar coordinate. Neither is directly observable from scalp EEG, which sees mainly $e$; only the field mixture over units is.

---

## 4. From the unit to the state: trace and purity

Drive the stable unit with white noise of covariance $I$. Its stationary covariance $P$ solves the Lyapunov equation $JP + PJ^\top + I = 0$. Normalise $\rho = P/\operatorname{tr}P$.

- **Trace.** $\operatorname{tr}P \propto 1/|s|$ as $s \to 0^-$: total variance diverges as net gain approaches instability. The trace is the $s$ knob.
- **Purity.** $\operatorname{tr}\rho^2$ depends on $\kappa/g$ only (for fixed $s$). Verified at $s=-1$, $g=0.8$:

| $\kappa$ | regime | purity $\operatorname{tr}\rho^2$ |
|---|---|---|
| 0.0 | mode-separated | 0.820 |
| 0.4 | mode-separated | 0.776 |
| 0.8 | exceptional point | 0.695 |
| 1.0 | oscillatory | 0.660 |
| 2.0 | oscillatory | 0.564 |
| 5.0 | oscillatory | 0.512 |
| $g=0$, any $\kappa$ | balanced | 0.500 exactly (isotropic) |

**Oscillatory means mixed; mode-separated means pure.** The purity is monotone in $\kappa/g$; the exceptional point is where the eigenvectors coalesce, not a discontinuity in purity.

For a field of many units with inter-unit coupling, the 8-channel $\rho$ of the montage is the mixture over the units' spatial modes; its von Neumann entropy $S(\rho)$ is the field-level counterpart of the single-unit purity.

---

## 5. Correspondence with the data (038 / 039 / entropy check)

| generator knob | state quantity | what the data showed |
|---|---|---|
| net gain $s$ | $\operatorname{tr}\Sigma$ (power) | 038: ictal shift is 83 % scale for the median patient; 039: the robust, reference-free polarity is the sign of the power change (19 up, chb14/21/15/06 down or flat) |
| coupling vs asymmetry $\kappa/g$ | purity / $S(\rho)$ (shape) | $S(\rho)$ rises ictally in 15/22 (toward oscillatory/mixed) and falls in 7/22 (chb02, 08, 09, 11, 13, 14, 21; toward mode separation); within-patient p < 0.01 for 13 |

The two knobs are independent in the model and the two data axes are independent in the cohort: the purifying set is not the power-down set (chb02, 08, 09 purify with large power increases). The naive prediction "seizure = purification" is false; the model's actual prediction — seizures move the field along two independent axes, with both purity directions allowed — is what the data shows.

---

## 6. Predictions and tests

1. **E/I proxy vs purity direction.** The aperiodic (1/f) slope of the EEG spectrum is an established classical proxy for cortical E/I balance. Prediction: the ictal change in slope should separate the purifying 7 from the mixing 15. Classical, on the 037 windows.
2. **Alpha blocking.** *Corrected 2026-09-07 after PROMPT 041.* The original version of this item predicted that $S(\rho)$ drops on eye closure because "one occipital mode takes over". That contradicted §4, which says the balanced oscillatory regime is *mixed*. PROMPT 041 (109 PhysioNet subjects) settled it in favour of §4: $\operatorname{tr}\Sigma$ rises in 73 % of subjects ($p \sim 10^{-6}$) and $S(\rho)$ does not fall (54/109; occipital 4-channel state rises in 72/109). Corrected prediction: **balanced oscillation raises the trace and leaves $S(\rho)$ unchanged or higher; purification is the signature of mode separation, not of rhythm.** For the lattice sessions the expected entropy change on eye closure is therefore *none or up*.
3. **Gauge check.** The absolute phase $b$ is uniform in eyes-open and eyes-closed alike — confirmed on PhysioNet (resultant 0.010 / 0.011 over 13,080 channel-windows, PROMPT 041) and on CHB-MIT (0.044, PROMPT 038). The PLV-to-global-phase observable *falls* on closure (73/109) because non-occipital channels do not lock to the occipital alpha that comes to define the global phase; a pairwise occipital PLV is the right observable if phase locking is revisited.
4. **Purity on hardware.** $\operatorname{tr}\rho^2$ is directly estimable on a prepared state (SWAP test / randomized measurement). It is the first DSP-000 observable that is (a) in the state, (b) within-patient, (c) predicted by the E/I model. Candidate for Phase 2 alongside the trace as a classical channel.

---

## 7. What is and is not established

Established here: the structure of the minimal coupled unit; the trace/traceless split and its two knobs; the identification of the oscillatory regime with the analytic signal and hence with the encoding actually used; the monotone purity–$\kappa/g$ relation; the qualitative match to 038/039 and the entropy split.

Not established: that cortex is described by this unit (it is the standard linearised two-population picture, adopted, not derived); which patients sit in which regime; any quantitative parameter estimate; the field-level (multi-unit) version of §4 beyond the qualitative mixture argument. The PT/exceptional-point language is a description of the $2\times2$ algebra and should not be presented as more than that until the E/I proxy test (§6.1) has been run.
