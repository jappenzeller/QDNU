# A-Gate observable notes — deriving an E/I balance observable from the PN model

**Date:** 2026-09-04
**Status:** theory note, verified numerically (statevector, max error < 1e-15)
**Feeds:** DSP-000 Phase 2 (Candidate B, "reference observable from PN motivation"); the "why this circuit" reviewer question; PROMPT 038
**Companion code:** `/tmp/agate.py`-style 2-qubit statevector check (trivial to reproduce; single-channel gate from `qdnu/quantum_agate.py`)

---

## 0. What the circuit measures today

`qdnu/multichannel_circuit.py` default readout measures only qubit 0, the ancilla. The ancilla is prepared in H superposition, receives a CZ from every excitatory qubit, and is read out. So Paper 1's "fixed observable" is the ancilla's $Z$ — a function of the interference between all excitatory-qubit phases. Consequences:

- Inhibitory qubits reach the readout only indirectly, through the $CR_z$ coupling each applies to its excitatory partner.
- The readout is a synchronization detector, not a signed quantity. Its E/I content is incidental.
- The PN model motivated the *encoding*, not the *measurement*. That is the substance of the reviewer's "why this circuit."

## 1. The single-channel A-Gate

Two qubits per channel. From `quantum_agate.py` / Paper 1 Algorithm 1:

- Excitatory qubit E: $H \cdot P(b) \cdot R_x(2a) \cdot P(b) \cdot H$
- Inhibitory qubit I: $H \cdot P(b) \cdot R_y(2c) \cdot P(b) \cdot H$
- Coupling: $CR_y(\pi/4)$ control E → target I, then $CR_z(\pi/4)$ control I → target E

with $a, c \in [0,1]$ (radians) and $b \in [0, 2\pi]$ from the PN extraction.

The PN model's one signed asymmetry (Paper 1 eqs. 1–2):

$$
\frac{da}{dt} = -\lambda_a a + f(t)(1-a), \qquad
\frac{dc}{dt} = +\lambda_c c + f(t)(1-c)
$$

The natural signed quantity is the balance $a - c$. The task: find an observable $\Pi$ on the encoded state whose sign is the sign of $a - c$, derived from the model rather than fitted.

## 2. Uncoupled single-qubit expectations (closed form)

Drop the coupling gates; each qubit is then a product factor. For the E qubit after the sandwich:

$$
\langle Z_E\rangle = \sin^2 a + \cos^2 a\,\cos 2b
$$

$$
\langle X_E\rangle = \sin 2a\,\sin b
$$

$$
\langle Y_E\rangle = -\cos^2 a\,\sin 2b
$$

For the I qubit (only difference is $R_y$ in place of $R_x$):

$$
\langle Z_I\rangle = -\sin^2 c + \cos^2 c\,\cos 2b
$$

$$
\langle X_I\rangle = -\sin 2c\,\cos b
$$

$$
\langle Y_I\rangle = -\cos^2 c\,\sin 2b
$$

Checks: $b=0$ gives $\langle Z_E\rangle = 1$ for all $a$ and $\langle Z_I\rangle = \cos 2c$; $a=0$ gives $\langle Z_E\rangle = \cos 2b$ ($H P(2b) H$).

### 2.1 Why $Z_E - Z_I$ fails

$$
\langle Z_E - Z_I\rangle = (\cos^2 a - \cos^2 c)\cos 2b + \sin^2 a + \sin^2 c
$$

At $b = 0$: $= 2\sin^2 c$ (no $a$, never negative). At $b = \pi/2$: $= 2\sin^2 a$ (no $c$, never negative). The phase does not modulate a balance readout; it selects *which* amplitude is read, and neither limit is signed. $Z_E - Z_I$ is not the quantum image of $a - c$.

### 2.2 The $X$ readouts factor into amplitude × phase gain

$\langle X_E\rangle = \sin 2a \cdot \sin b$ and $\langle X_I\rangle = -\sin 2c \cdot \cos b$. The phase factors are gains, 90° apart. The offset is the $R_x$-vs-$R_y$ asymmetry — the one place the PN model's E/I distinction became circuit structure, and it appears as a measurable signature.

Interpretation of "phase decides what you're reading": $b$ is a dial that turns the E amplitude up in the $X$ readout while turning the I amplitude down, and vice versa.

### 2.3 The balance observable (uncoupled)

$b$ is known classically per window (it is a computed feature), so it may be used in interpreting the measurement. Divide each $X$ readout by its gain and add:

$$
\langle \Pi_b\rangle \equiv \frac{\langle X_E\rangle}{\sin b} + \frac{\langle X_I\rangle}{\cos b} = \sin 2a - \sin 2c
$$

Signed E/I balance. $\operatorname{sign}\langle\Pi_b\rangle = \operatorname{sign}(a - c)$ provided $a, c \le \pi/4 \approx 0.785$ (where $\sin 2a$ is still increasing).

Limits:
- Gains vanish at $b \in \{0, \pi\}$ (E term unreadable) and $b \in \{\pi/2, 3\pi/2\}$ (I term unreadable); dividing by a small gain amplifies shot noise. Windows near those phases are unreliable.
- PN extraction allows $a, c$ up to 1 > $\pi/4$; above $\pi/4$ the sign relation can break. **Check the empirical distribution of $(a,b,c)$ on CHB-MIT before relying on this.**

## 3. With the coupling gates

Let $U = CR_z(\pi/4)_{I\to E}\; CR_y(\pi/4)_{E\to I}$ (applied in that order after the sandwiches). $U$ entangles the pair and has no free parameters.

Numerical result: applying the uncoupled $\Pi_b$ (measure $X_E$, $X_I$ on the coupled state, divide by gains, subtract) gives $\operatorname{sign}$ agreement with $a - c$ of only **69%** over random $(a,b,c)$ with $a,c \in [0,\pi/4]$ and $|\sin b|, |\cos b| > 0.3$. The coupling breaks the simple subtraction.

But $U$ is a fixed known unitary, so it relabels rather than destroys information. Measuring $O$ after $U$ equals measuring $U^{\dagger} O U$ before it; equivalently

$$
\langle \Pi_b\rangle_{\text{uncoupled}} = \big\langle U\,\Pi_b\,U^{\dagger}\big\rangle_{\text{coupled}}
$$

Numerical result: the transformed observable reproduces $\sin 2a - \sin 2c$ **exactly** (100%, machine precision) on the coupled state.

Pauli expansion of the transformed single-qubit $X$ operators (coefficients rounded; E is the first tensor factor):

$$
U X_E U^{\dagger} = 0.789\,X_E + 0.327\,Y_E + 0.146\,X_E Y_I + 0.135\,X_E Z_I - 0.354\,Y_E Y_I - 0.327\,Y_E Z_I
$$

$$
U X_I U^{\dagger} = 0.789\,X_I - 0.056\,Y_I - 0.354\,Z_I + 0.135\,Z_E X_I - 0.327\,Z_E Y_I + 0.354\,Z_E Z_I
$$

So the model-derived balance observable on the actual A-Gate is

$$
\Pi_b^{\text{coupled}} = \frac{U X_E U^{\dagger}}{\sin b} + \frac{U X_I U^{\dagger}}{\cos b}
$$

Six Pauli settings per qubit on hardware instead of one. Entirely measurable.

## 4. What this says about "why this circuit"

Every gate after the encoding — the two couplings, and in the multichannel circuit the CNOT rings and the ancilla CZs — is a fixed unitary. Anything measurable after them is a transformed observable on a product state whose only information content is the classical triples $(a_k, b_k, c_k)$. The circuit is a fixed change of measurement basis; every expectation value is a trigonometric polynomial in the inputs that a classical computer evaluates directly.

This is DSP-000's opening concession ("the circuit could be replaced by a classical function computing the same scalar observables", Schuld–Killoran sense 2) made explicit and constructive: the classical function is the Pauli expansion above. The $O(M)$-vs-$O(M^2)$ gate-count claim describes the cost of a fixed function, not an advantage.

The way out is not a cleverer fixed circuit; it is making the state carry information a classical scalar cannot — the covariance itself as a density matrix (DSP-000).

## 5. What to do with it

1. **PROMPT 038 test (classical, no hardware):** compute $\langle\Pi_b^{\text{coupled}}\rangle$ per window from the existing CHB-MIT $(a,b,c)$ features, aggregate ictal and interictal per patient, take $\operatorname{sign}(\langle\Pi\rangle_{\text{ictal}} - \langle\Pi\rangle_{\text{interictal}})$, and compare with manifold polarity (Paper 2 / 037 per-patient direction). Match → the E/I ↔ polarity link has an equation. Mismatch → E/I motivated the encoding and nothing more; "why EEG" rests on the practical argument.
2. Before (1): histogram $(a, b, c)$ on CHB-MIT. Fraction of $a, c > \pi/4$ and of $b$ within the dead zones bounds how much of the data the observable can read.
3. For DSP-000 Phase 2, this is Candidate B's derivation. Its geometric interpretation in tangent space is *not* established — that is the comparison in (1).
4. The ancilla observable Paper 1 used is a different fixed transformation of the same product state; it reads excitatory-phase synchronization. Worth writing its Pauli expansion too, for completeness, so the two observables can be stated side by side.

## 6. Related fact from PROMPT 037 (same day)

Per-patient ictal shift vectors in tangent space are coherent across seizures ($c_{\text{corr}} \approx 0.84$–$1.00$ for 19/22 patients). This is what makes a per-patient polarity *sign* well defined, and therefore what makes the comparison in §5(1) meaningful. Separately, whether polarity is a scale change ($\operatorname{tr}\Sigma$) or a shape change ($\rho = \Sigma/\operatorname{tr}\Sigma$) is unresolved and decides whether the trace-normalized quantum state of DSP-000 can see it at all; it is the other PROMPT 038 question.
