# P2REV-03 Job 0 - the submitted quantum rows, pinned

Every cell below was read out of the producing script, not inferred.

| row | result file | script that produced it | window | montage | labelling | segments per patient | shots | scoring function |
|---|---|---|---|---|---|---|---|---|
| hardware 1.95 s | `results/hardware_validation/ch8_heron_loso_results.json` | `scripts/quantum_heron_validation.py` (PROMPT 006) | **first 2.0 s sub-window** of each 30 s `train_chbmit` segment (`SUBWINDOW_SEC = 2.0`; the manuscript's "1.95 s" appears nowhere in the code) | fronto-central CH8: FP1-F7, F7-T7, FP1-F3, F3-C3, FP2-F8, F8-T8, FP2-F4, F4-C4 (from `classical_channel_scaling.json`) | original (`label_map = {interictal:0, preictal:1, ictal:1}`) | chb01 24, chb03 24, chb05 20, chb07 16, chb11 16, chb14 26, chb21 18 (= 144; one score per segment) | 1024, `optimization_level=3`, no mitigation | **Hellinger fidelity of the measured 17-bit count distribution against two template circuits** (ictal / interictal, parameters averaged over the training patients' sub-window params, templates run on hardware); score = fid_ictal - fid_inter |
| hardware 20 s | `results/window_analysis/quantum_20s_hardware.json` | `scripts/quantum_20s_hardware.py` (PROMPT 017) | first 20 s of each 30 s segment, one circuit per segment | same CH8 (the "bipolar" list in its config IS the fronto-central eight - all CHB-MIT channels are bipolar pairs; same montage as every other run) | original | same segment sets as above (24/24/20/16/16/26/18) | 1024, `optimization_level=3`, no mitigation | same dual-template Hellinger scoring, templates re-run on hardware per fold (template segments capped at first 50 per class) |
| simulation 1.95 s | `results/scaling/quantum_channel_scaling.json` `ch8` | `scripts/quantum_channel_scaling.py` | **all** 2.0 s sub-windows of each 30 s segment (15 per full segment), **MAX over sub-windows** | same CH8 | original | 22 subjects, 504 segments total (`n_samples` per subject in the file) | statevector (exact) | **amplitude fidelity** abs(<psi_test|psi_template>)^2 against statevector templates; score = fid_ictal - fid_inter, MAX over sub-windows |
| (simulation 20 s) | `results/window_analysis/quantum_20s_simulation.json` | `scripts/quantum_20s_simulation.py` | first 20 s, one circuit per segment | same CH8 | original | 7 subjects only (templates from the other 6 of the 7, first 50 per class) | statevector | **Bhattacharyya coefficient of basis-state probability distributions** against template distributions; score = fid_ictal - fid_inter |

Facts the table forces into the open (each verified in the scripts):

1. **None of the four rows measures the ancilla <Z_0>.** Every submitted quantum
   number is dual-template fidelity classification: the test state's distance to an
   ictal and an interictal *template circuit* whose parameters are averages over the
   **training patients'** (a, b, c) features. The templates are rebuilt per LOSO fold
   from training data, so the method has a trained component; the manuscript's
   section-3.5 scoring definition ((N0 - N1)/N on the output qubit) and the
   "no trained parameters" framing of section 5.2 do not describe the submitted
   numbers.
2. The hardware 1.95 s and simulation 1.95 s rows differ in more than backend:
   hardware scores the **first** 2.0 s sub-window per segment by **count-distribution
   Hellinger** fidelity; the simulation scores **all 15** sub-windows by **amplitude**
   fidelity and takes the MAX. The submitted "simulation-hardware gap"
   (Delta = 0.0515) therefore compares two different estimators as well as two
   backends. The 20 s rows are closer (same one-window-per-segment design;
   count-Hellinger vs statevector-probability Bhattacharyya).
3. The window the manuscript calls "1.95 s" is 2.0 s in every producing script.
4. No quantum run applies the classical preprocessing chain (0.5-40 Hz bandpass,
   60 Hz notch, z-score); raw EDF goes straight into the PLV extraction's own
   4-13 Hz bandpass. Section 3.1's "preprocessing applied to all experiments" is
   wrong for the quantum rows.
5. All four rows use `train_chbmit` 30 s segments with the original labelling
   (preictal 60 s labelled 1) and the first-minutes interictal selection; the
   circuit builder is `QA1/multichannel_circuit.py`, whose inter-channel layer is an
   **open chain** (M-1 CNOTs); `qdnu/multichannel_circuit.py` is a *closed ring* and
   was NOT what the runs used.
