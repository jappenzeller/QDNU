# P2REV-03 Job 2 (revised) — How Paper 1 describes the classification rule

Source: `submission/dqs_v1/main.tex` (the published Discover Quantum Science submission,
doi:10.1007/s44464-026-00032-w). Quotes verbatim, with line numbers from that file.

## The quotes

- **Abstract (l. 26):** "the quantum **fidelity classifier** achieves 53.4\% AUC compared to
  62.5\% for an 8-channel classical baseline. Notably, per-subject AUC ranges from 0.183 to
  1.000, with sub-chance performance in some cases suggesting what we term \emph{manifold
  polarity}---patient-specific geometric orientations that may cause cross-subject **template
  mismatch**."
- **Complexity section (l. 137):** "**Template matching** via the SWAP test \cite{buhrman2001}
  requires $O(M)$ additional gates rather than $O(M^2)$ classical operations."
- **Measurement statistics (l. 171):** "The SWAP test yields fidelity
  $F = |\langle\psi|\phi\rangle|^2$ through repeated measurements."
- **Hardware validation (l. 343):** "Discrimination quality was computed as
  $1 - \bar{F}_{\text{off-diag}}$ where $\bar{F}_{\text{off-diag}}$ is the mean off-diagonal
  **Hellinger fidelity** \cite{fuchs1999}."
- **Limitations (l. 507):** "\textbf{Template-based classification}: The quantum **fidelity**
  approach uses **fixed templates** rather than learned decision boundaries, potentially
  limiting adaptability."
- **Discussion (l. 509):** "ictal signatures project to opposing orientations on the SPD
  manifold, causing **templates** optimized for one group to anti-correlate with another."

Searched and absent: "ancilla" appears only in the gate-count accounting (l. 112, "The ancilla
qubit requires one Hadamard gate for initialization..."); `N_0`, `N_1`, and any
$(N_0 - N_1)/N$ readout formula appear nowhere; no sentence describes measuring the ancilla's
$Z$ expectation as the classification score.

## Verdict

**Paper 1 describes the template classifier.** The classification rule it names — a fidelity
classifier scoring test windows against fixed class templates, with Hellinger fidelity defined
for hardware count distributions — is the rule the executed code implements
(`run_inventory.md`). Paper 1 does **not** describe the ancilla observable; the
$(N_0-N_1)/N$ ancilla-expectation description is unique to Paper 2's §3.5 and matches neither
Paper 1 nor the code. Two precision caveats for the response letter: (i) Paper 1's complexity
section attributes template matching to "the SWAP test", but no SWAP-test circuit was executed
— simulation computed fidelity classically from statevectors and hardware compared measured
count distributions by Hellinger fidelity; (ii) the templates are "fixed" only within a fold —
they are rebuilt per LOSO fold from the training patients' averaged parameters.

## Reproducibility statement (R2 §1.9)

For a template classifier the classical implementation **is** the statevector pipeline, and it
is in the repository: `scripts/quantum_20s_simulation.py` — templates built by
`build_plv_template`-style parameter averaging over the training patients (the per-fold
average of `extract_plv_params` outputs, circular mean for $b$), scoring by
`statevector_fidelity` (Bhattacharyya coefficient of basis-state probability distributions)
as `score = fid_ictal − fid_inter`. P2REV-03 Job 1 re-executed this pipeline end-to-end
(`scripts/p2rev03_quantum_gap.py`, stage job1) and reproduced the submitted 20 s simulation
row **exactly** (per-patient maximum deviation 0.0; `job1_reproduction.json`). No IBM access
is needed to reproduce every simulated number in the revised quantum section.
`qdnu/agate_observable.py` (written in the interim fixed-observable analysis, archived under
`z0_analysis/`) is not a deliverable of this prompt.
