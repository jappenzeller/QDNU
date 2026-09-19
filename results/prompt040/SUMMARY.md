# PROMPT 040 - Does non-commutativity between patients' shift generators predict reference-dependent polarity?

**Generated:** 2026-09-06
**Script:** `scripts/prompt040_commutator.py`
**Depends on:** `results/prompt037/cache/W8`, `results/prompt039/per_patient_W8*.csv`, `results/prompt038/q1_scale_shape.csv`

Conventions: 22 patients, W = 8, `lwf` covariances, ictal and interictal windows only (1358 windows). Common frame: Riemannian mean of all windows. Per patient the shift generator is $G_p = \overline{\log C}_{\rm ictal} - \overline{\log C}_{\rm interictal}$ in that frame (symmetric 8x8); $G_p^{\perp}$ removes the identity component. Pairwise: cosine (alignment), $\kappa = \|[G_p, G_q]\|_F \,/\, (\sqrt{2}\,\|G_p\|\,\|G_q\|)$ in [0,1] (incompatibility), and `$\mathrm{AUC}_{p|q}$` = patient p's windows scored by the linear functional $\langle \log C, G_q\rangle$ (q's direction read on p). 462 ordered pairs.

**Claim tested:** polarity is axis-dependent because patients' generators do not share an eigenbasis, so the commutator norm should predict how badly one patient's direction reads another beyond what alignment explains, and mean incompatibility should predict the 039 reference stability.

---

## Results

### Pairwise (T1)

| generator | kappa median (min-max) | cos median | cos < 0 | sign(AUC-0.5) = sign(cos) | dev ~ \|cos\| | dev ~ kappa | partial(kappa \| \|cos\|) |
|---|---|---|---|---|---|---|---|
| full G | 0.043 (0.01-0.19) | +0.79 | 5 % | 97.6 % | +0.59 | -0.57 | -0.19 |
| shape G^perp | 0.255 (0.13-0.42) | +0.08 | 42 % | 96.8 % | **+0.87** | -0.15 | **+0.06** |

`dev` = $|\mathrm{AUC}_{p|q} - 0.5|$. Pairwise |cos| and kappa are themselves correlated at -0.80 on the full generators.

### Per patient (T2, T3; Spearman, n = 22)

| target | mean kappa | mean \|cos\| | mean cos (signed) | f_scale (038) |
|---|---|---|---|---|
| 039 strict-LDA stability | -0.48 (p = 0.024) | **+0.62 (p = 0.002)** | - | +0.48 (p = 0.024) |
| 039 pooled-LDA stability | -0.31 (n.s.) | +0.42 (p = 0.053) | - | +0.25 (n.s.) |
| fraction of other patients that read me inverted (full) | +0.45 (p = 0.036) | -0.21 (n.s.) | -0.39 (p = 0.07) | -0.37 (p = 0.09) |
| same, shape generators | -0.17 (n.s.) | -0.11 (n.s.) | **-0.84 (p = 1e-6)** | +0.18 (n.s.) |

And the one that explains the rest: **mean kappa vs f_scale, rho = -0.95 (p = 5e-12).** The commutator norm of the full generators is, to within noise, one minus the scale fraction. The identity commutes with everything, so patients whose shift is mostly scale have small commutators with everybody. Every kappa correlation in the full-generator rows is f_scale in disguise. On the shape generators kappa is flat across patients (0.23-0.32) and predicts nothing.

Per-patient table (full / shape): chb06, chb13, chb21 have the largest full-kappa (0.11-0.13) and f_scale 0.00-0.08; chb14, chb15, chb21, chb06 are read inverted by 86 % of other patients (they are the power-down patients from 039); in shape space the patients read inverted most often are chb09 (0.71), chb14 (0.71), chb03 (0.67), chb08 (0.67) - and that is predicted by their signed mean cosine, not by kappa.

---

## Verdict: negative for the non-commutativity mechanism, as posed

The commutator does not predict linear polarity beyond alignment. Pairwise, the sign of `$\mathrm{AUC}_{p|q}$` is the sign of the cosine 97-98 % of the time, and the deviation from chance is explained by |cos| (rho 0.87 on shape generators) with no residual role for kappa (partial +0.06). Per patient, reference stability tracks alignment and scale fraction; the apparent kappa effect on the full generators is the scale fraction wearing a different name.

This is what the formalism itself says, and it was misapplied in the 039 discussion. A commutator governs what cannot be sharp *simultaneously* - variances, uncertainty relations, sequential and basis-projective measurements. It does not govern expectation values, and a linear projection $\langle \log C, G_q\rangle$ is an expectation value. Polarity as measured in Papers 2-3 and in 039 is an expectation-value readout, so its axis-dependence is a Euclidean fact: the shape generators are nearly orthogonal across patients (cos median 0.08) and anti-aligned for 42 % of pairs. No eigenbasis argument is needed to explain it, and none is supported.

What survives: the shape generators genuinely do not commute (kappa 0.26 median, well above the full-generator floor), so a *basis-projective* readout - measuring patient p in patient q's eigenbasis, or the variance of $\langle \log C, G_q\rangle$ over p's windows against a bound set by |<[G_p, G_q]>| - is where non-commutativity would show if it shows anywhere. That is a different observable from anything run so far, and it is the correct follow-up if the "patients' states are incompatible" thread is to be kept. It was not run here because it was not the claim tested.

Of the four correspondences proposed for the theory chapter (analytic signal / gauge; states / non-commutativity; covariance / mixed state; E-I unit / two-level system), the second is now unsupported in its expectation-value form and reduces to "shape shifts are nearly orthogonal across patients", which is classical geometry. The first (global phase of `b` is unobservable) still stands on 038's evidence.

---

## Deliverables

```
scripts/prompt040_commutator.py
results/prompt040/
    generators.npz     G, G_perp per patient, common reference
    pairwise.csv       462 ordered pairs: cos, kappa, AUC (full and shape)
    per_patient.csv
    tests.json
    SUMMARY.md
```
