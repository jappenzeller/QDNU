# QNFM — Quantum Neural Field Mapping

**A quantum-information geometry for multichannel EEG covariance states — and the hardware circuit that led there.**

*Repository name `QDNU` is historical (Quantum Dynamic Neural Unit, the Paper 1 era); the framework is QNFM.*

[![Site](https://img.shields.io/badge/qdnu.ai-site-00d4ff)](https://qdnu.ai)
[![Notebook](https://img.shields.io/badge/qdnu.ai-notebook-f59e0b)](https://qdnu.ai/notebook/)
[![Paper 1](https://img.shields.io/badge/Discover_Quantum_Science-10.1007%2Fs44464--026--00032--w-1a73e8)](https://doi.org/10.1007/s44464-026-00032-w)

**Author:** James Appenzeller · PhD student, National University · independent researcher

---

## What this repository is

An EEG window of *M* channels gives an *M*×*M* covariance matrix Σ. Divided by its trace, Σ/tr Σ is symmetric, positive semidefinite and has trace one — which is the definition of a density matrix. This repository treats that identification literally: the trace-normalised covariance is a quantum state, the total power is a classical scalar that travels alongside it, and the questions asked of the state are the ones quantum information asks — its spectrum, its purity, its entropy, how it moves under a generator.

Three things live here:

1. **The current line of work (2026-09 →).** The covariance-as-state formulation ("DSP-000"), prepared on a simulated register by purification at 2, 4 and 8 channels and read back exactly; a two-level excitatory/inhibitory (E/I) generator derived as the state's dynamics; and the tests of both on public data. Written up as readable pages at [qdnu.ai/notebook](https://qdnu.ai/notebook/), with sources in `docs/web/`.
2. **The polarity work (Paper 2, 2026-04 → 09).** Within a patient, seizures shift the covariance in a coherent direction (coherence ≈ 1 across seizures), and the sign of that shift is patient-specific. Results in `results/prompt035…042/`, each with a `SUMMARY.md`.
3. **The hardware feature map (Paper 1, published).** The A-Gate: a fixed two-qubit circuit per channel, driven by EEG phase-locking features, run on IBM Heron. Code in `qdnu/`, results in `analysis_results/` and `results/hardware_validation/`. This is the history — see *What changed since Paper 1* below before reading it as the current claim.

## What changed since Paper 1

Paper 1 (*Discover Quantum Science* 2:30, 2026) reported that an 8-channel A-Gate circuit on IBM Heron reached 0.637 AUC on CHB-MIT after per-patient polarity calibration, against 0.742 for the strongest classical baseline, and named the encoding as the bottleneck. Those numbers stand. Two claims made around them do not, and this README is the place to say so plainly:

- **The A-Gate is a fixed basis change on classical features.** Every gate after the encoding is a fixed unitary, so anything the circuit measures is a transformed observable on three classical numbers per channel. One of the three, the absolute phase `b`, carries no information at all — it is a gauge, confirmed on ~32,000 windows across CHB-MIT and PhysioNet (`docs/AGATE_OBSERVABLE_NOTES.md`; `results/prompt038`, `results/prompt041`).
- **There is no O(M) versus O(M²) advantage.** The earlier README claimed one. The circuit does not compute inter-channel correlation; the classical preprocessing does. The claim is withdrawn.

What the circuit did do was expose a sign that flexible classifiers absorb: a fixed measurement basis cannot flip with the patient, so patients whose ictal shift points the other way come out below chance. That observation is real, it is Paper 2, and it is what led to the state formulation. The honest summary of "why quantum" here is *formulation, not speed*: the rigidities of the formalism — fixed basis, trace one, unobservable global phase — each produced a finding.

`docs/STATE_OF_THE_WORK.md` is the one-page version of this arc.

## Layout

```text
qdnu/               A-Gate circuit, PN dynamics, multichannel builder (Paper 1 era)
scripts/            analysis scripts; promptNNN_*.py pair with results/promptNNN/
  phase5_*.py       purification worked examples, 2 / 4 / 8 channels
results/promptNNN/  outputs with a SUMMARY.md each (035–042 = polarity and state work)
results/paper1/     Paper 1 tables
analysis_results/   Paper 1 era LOSO and hardware outputs
docs/               theory notes (AGATE_OBSERVABLE_NOTES, EI_TWO_LEVEL_NOTES, STATE_OF_THE_WORK)
docs/web/           story pages, light-theme sources of qdnu.ai/notebook
visualization/      Three.js SPD-manifold viewer (qdnu.ai/viz)
aws/  infra/        Braket job package and data-layer bootstrap
arxiv_preflight/    Paper 1 LaTeX source
```

The qdnu.ai site itself lives in a separate repository. Prompt files that drive the work (`docs/PROMPT_*.md`) are local and not tracked.

## Reproduce the notebook pages

The phase 5 examples run on public data with no credentials.

```bash
git clone https://github.com/jappenzeller/QDNU.git && cd QDNU
python -m venv .venv && source .venv/bin/activate      # .venv\Scripts\activate on Windows
pip install -r requirements.txt

# fetch PhysioNet eegmmidb subject 1, runs 1–2 (eyes open / closed), ~2 MB
python -c "import mne; mne.datasets.eegbci.load_data(1, [1, 2], path='data/eegmmidb', update_path=False)"

python scripts/phase5_two_channel.py     # 1 system + 1 ancilla qubit
python scripts/phase5_four_channel.py    # 2 + 2
python scripts/phase5_eight_channel.py   # 3 + 3, 26 CNOTs transpiled
```

Each prints Σ, ρ, the spectrum, the reduced state after purification (equal to ρ to ~1e-15), shot estimates of the Z-string observables against tr(ρO), and the eigenvalues read back from the ancilla. The 041 eye-closure test (`scripts/prompt041_eye_closure.py`) runs on the same dataset for all 109 subjects.

The CHB-MIT scripts (`prompt035…040`, `042`) need the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) downloaded locally; pass its root with `--data-root`. Data is not distributed here.

## Paper 1 hardware results, for the record

IBM Heron (ibm_torino), CHB-MIT, 7 patients, LOSO, 17 qubits, 97 CZ gates, 1024 shots, no error mitigation.

| Patient | Raw AUC | Calibrated AUC | Polarity |
|---------|---------|----------------|----------|
| chb01 | 0.686 | 0.686 | standard |
| chb03 | 0.436 | 0.564 | inverted |
| chb05 | 0.610 | 0.610 | standard |
| chb07 | 0.667 | 0.667 | standard |
| chb11 | 0.283 | 0.717 | inverted |
| chb14 | 0.600 | 0.600 | standard |
| chb21 | 0.388 | 0.613 | inverted |

Cohort: raw 0.531, calibrated 0.637, strongest classical baseline (log-FFT + correlation eigenvalues, XGBoost) 0.742. Full tables in `results/paper1/` and `analysis_results/`.

## Citation

```bibtex
@article{appenzeller2026qpnn,
  title   = {Hardware-Validated Quantum Positive-Negative Neuron Architecture for Multi-Channel EEG Analysis},
  author  = {Appenzeller, James},
  journal = {Discover Quantum Science},
  volume  = {2},
  number  = {30},
  year    = {2026},
  doi     = {10.1007/s44464-026-00032-w}
}
```

## License

Research use. Contact the author for collaboration or reuse.
