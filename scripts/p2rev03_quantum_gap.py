#!/usr/bin/env python3
"""
================================================================================
PROMPT P2REV-03 - The quantum rows: classical equivalent, finite-shot and
noise-model distributions
================================================================================

Stages (--stage): job0 (pin the runs), job1 (reproduce ideal expectations and
pin the segment set; CHECKPOINT - the run stops here and reports if any
submitted quantum row cannot be reproduced to 1e-6), job2..job6 per the
prompt, gated on the Job 1 checkpoint.

Outputs: results/p2rev03/ (tracked).

================================================================================
EXPECTATION BLOCK (verbatim from docs/PROMPT_P2REV_03_quantum_gap.md as
updated by Jim; replaces the earlier Claude-operationalized block)
================================================================================

> Jim's stated expectation (2026-10-25): "nothing is correct and it's all
> noise."
>
> As predictions: H1 passes (hardware values sit inside the shot-noise band;
> the gap was never a gap); H2 then adds nothing; Q1 fails (no
> hardware-subset patient is reliably below chance in the ideal simulation);
> G2 is at chance (sign agreement with pi_p around 3-4 of 7). The one thing
> not expected to be noise is pi_p itself, which P2REV-02 already tested
> (G1). Anything stronger than chance in Q1 or G2 is a surprise and is
> reported as one.

================================================================================
REVISION OF SCOPE (resume from Job 2, 2026-10-31) - as implemented
================================================================================

The Job 1 checkpoint showed no submitted row measures the ancilla <Z_0>. The
resume instruction references a "Revision of scope" section of the prompt
that is not present in the file on disk; the implemented reading, recorded
here so it can be corrected if wrong: Jobs 2, 3, 5 and 6 run exactly as the
prompt states them - i.e. the analysis object is the FIXED ancilla
observable <Z_0> (the section-3.5 definition), computed classically, with
the submitted template-fidelity pipeline documented in run_inventory.md and
the estimator mismatch carried as a caveat on every hardware comparison.
Job 4 / H2 (FakeTorino) is OUT OF SCOPE under the revision - the expectation
block's "H2 then adds nothing" - leaving exactly three pre-declared blocks
in force. Jobs 0-1 outputs are untouched.

================================================================================
PRE-DECLARED BLOCKS IN FORCE (three, verbatim from the prompt; H2/Job 4
descoped, not evaluated)
================================================================================

H1 (Job 3, finite-shot distribution)
    For each segment, p = (1 + <Z_0>)/2. Draw counts ~ Binomial(1024, p),
    200 seeds, score = 2*p_hat - 1. Per seed, compute per-patient AUC and the
    overall AUC (same pooling as the submitted row). Report, per patient and
    overall, the 2.5 / 50 / 97.5 percentiles, and the percentile at which the
    hardware value sits. Both rows (1.95 s, 20 s).
    Pre-declared H1: the hardware per-patient raw AUC lies inside the
    finite-shot 95 % interval for >= 5 of 7 patients, for the 1.95 s row.
    Reported separately for the 20 s row. If H1 passes, shot noise alone
    accounts for the hardware numbers and the "discrepancy" is not one. If H1
    fails, Job 4 is where the explanation has to come from. [Caveat carried
    from Job 0: the hardware AUCs being located in these intervals were
    produced by the template-Hellinger estimator, not by <Z_0>; H1 therefore
    tests whether the hardware numbers are distinguishable from shot-level
    noise around the fixed observable, and the estimator mismatch is stated
    wherever H1 is reported.]

Q1 (Job 5, sub-chance readout significance)
    For the ideal 1.95 s scores, per patient, 7-subset and all 22, both
    labellings: permutation p for "below chance" (1000 within-patient label
    permutations, same definition as P2REV-02 c2). For the hardware 20 s row
    the per-window scores exist and P2REV-02 already bootstrapped them; add
    the permutation p there too. The hardware 1.95 s row has no per-segment
    scores; state that again and do not fabricate an interval.
    Pre-declared Q1: the section-5.2 sentence "AUC < 0.5 is an explicit
    polarity readout" is retained only if, under the original labelling,
    >= 3 of the 7 hardware-subset patients are below chance in the ideal
    simulation at p < 0.05 AND the same patients are below chance in Tier 3
    (P2REV-02 per_patient_auc.csv, perm_p). Otherwise the sentence is
    replaced by: the fixed-basis observable's sign is reported per patient as
    an observation, with its interval.

G2 (Job 6, sign table)
    One table, 7 hardware patients, both labellings where available: pi_p and
    its cosine (from geometric_sign.csv), Tier 3 sign, ideal-sim sign,
    hardware 1.95 s sign, hardware 20 s sign, and the balance-observable sign
    from Job 2. Agreement counts for every pair of columns.
    Pre-declared G2: the ideal-sim sign agrees with pi_p in >= 5 of 7 under
    the original labelling. Reported, not judged, for strict.
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'QA1'))

OUT_DIR = PROJECT_ROOT / "results" / "p2rev03"
DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
HW_SUBSET = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']
REPRO_TOL = 1e-6


def log(msg=""):
    print(f"{datetime.now():%H:%M:%S} | {msg}", flush=True)


def load_script(name):
    """Import one of the original run scripts as a module (functions only;
    all have __main__ guards)."""
    spec = importlib.util.spec_from_file_location(
        name, PROJECT_ROOT / 'scripts' / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# =============================================================================
# Stage job0 - pin the runs (table written to run_inventory.md)
# =============================================================================

RUN_INVENTORY = """# P2REV-03 Job 0 - the submitted quantum rows, pinned

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
"""


def stage_job0():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / 'run_inventory.md', 'w', encoding='utf-8') as f:
        f.write(RUN_INVENTORY)
    log("job0: run_inventory.md written")


# =============================================================================
# Stage job1 - reproduce the ideal expectations and pin the segment set
# =============================================================================

def ancilla_z_from_sv(sv):
    """<Z_0> for qubit 0 from a qiskit Statevector."""
    p = sv.probabilities([0])
    return float(p[0] - p[1])


def stage_job1():
    """Faithful re-run of the two simulation rows (the only reproducible
    rows), plus the ancilla <Z_0> evaluation the prompt specifies. Writes
    job1_reproduction.json and the ideal_scores CSVs, then evaluates the
    1e-6 checkpoint."""
    from qiskit.quantum_info import Statevector
    from sklearn.metrics import roc_auc_score

    qcs = load_script('quantum_channel_scaling')      # sim 1.95 s row
    q20 = load_script('quantum_20s_simulation')       # sim 20 s row
    from QA1.multichannel_circuit import create_multichannel_circuit

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- load all segments exactly as the scaling script did ---------------
    channels = qcs.load_channels() if hasattr(qcs, 'load_channels') else \
        ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
         "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]
    log(f"job1: loading segments (channels: {channels})")
    eeg_list, labels, subjects = qcs.load_all_segments(channels)
    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    y = np.array([label_map.get(l, 0) for l in labels])
    uniq = sorted(set(subjects))
    log(f"job1: {len(eeg_list)} segments, {len(uniq)} subjects")

    # ---- cache (a,b,c) params per (segment, sub-window), original order ----
    log("job1: extracting PLV params (2.0 s sub-windows, as the scripts did)")
    seg_params = []     # list over segments of list over sub-windows of params
    for eeg in eeg_list:
        seg_params.append(qcs.extract_plv_params_subwindows(
            eeg, fs=qcs.FS, band=qcs.BAND))

    # statevector + <Z_0> per (segment, sub-window), computed once
    log("job1: statevectors + <Z_0> per sub-window")
    seg_svs = {}        # i -> list of Statevector (kept only per fold, below)
    z0_rows = []        # per sub-window ancilla expectation
    n_sw_total = sum(len(s) for s in seg_params)
    log(f"job1: {n_sw_total} sub-window circuits")

    # ---- faithful sim-1.95s LOSO (templates from cached params) ------------
    def average_params(params_flat, n_ch):
        avg = []
        for ch in range(n_ch):
            avg_a = np.mean([p[ch][0] for p in params_flat])
            sins = np.mean([np.sin(p[ch][1]) for p in params_flat])
            coss = np.mean([np.cos(p[ch][1]) for p in params_flat])
            avg_b = np.arctan2(sins, coss) % (2 * np.pi)
            avg_c = np.clip(np.mean([p[ch][2] for p in params_flat]),
                            0.05, 0.95)
            avg.append((avg_a, avg_b, avg_c))
        return avg

    n_ch = len(channels)
    per_subject_repro = {}
    ideal_rows_195 = []
    z0_auc_195 = {}
    for test_subject in uniq:
        train_idx = [i for i in range(len(eeg_list))
                     if subjects[i] != test_subject]
        test_idx = [i for i in range(len(eeg_list))
                    if subjects[i] == test_subject]
        y_test = y[test_idx]
        if len(np.unique(y_test)) < 2:
            continue
        ictal_flat = [sw for i in train_idx if labels[i] in ('ictal',
                                                             'preictal')
                      for sw in seg_params[i]]
        inter_flat = [sw for i in train_idx if labels[i] == 'interictal'
                      for sw in seg_params[i]]
        if not ictal_flat or not inter_flat:
            continue
        ictal_sv = Statevector.from_instruction(
            create_multichannel_circuit(average_params(ictal_flat, n_ch)))
        inter_sv = Statevector.from_instruction(
            create_multichannel_circuit(average_params(inter_flat, n_ch)))

        final_scores = []
        z0_first = []
        for i in test_idx:
            sw_scores = []
            for k, sw in enumerate(seg_params[i]):
                sv = Statevector.from_instruction(
                    create_multichannel_circuit(sw))
                fid_i = abs(ictal_sv.inner(sv)) ** 2
                fid_t = abs(inter_sv.inner(sv)) ** 2
                sw_scores.append(fid_i - fid_t)
                z0 = ancilla_z_from_sv(sv)
                ideal_rows_195.append(dict(
                    patient=test_subject, segment_idx=i, subwindow_idx=k,
                    y=int(y[i]), z_expect=z0,
                    fidelity_score=fid_i - fid_t))
                if k == 0:
                    z0_first.append(z0)
            final_scores.append(max(sw_scores) if sw_scores else 0.0)
        auc = roc_auc_score(y_test, final_scores)
        try:
            z0_auc = roc_auc_score(y_test, z0_first)
        except ValueError:
            z0_auc = None
        z0_auc_195[test_subject] = z0_auc
        per_subject_repro[test_subject] = dict(
            repro_auc=float(auc), n=len(test_idx))
        log(f"job1: sim-1.95s {test_subject} repro AUC {auc:.6f} "
            f"| <Z0> first-subwindow AUC "
            f"{z0_auc if z0_auc is None else round(z0_auc, 4)}")

    # compare with the submitted scaling file
    scal = json.load(open(PROJECT_ROOT /
                          'results/scaling/quantum_channel_scaling.json',
                          encoding='utf-8'))['ch8']['per_subject']
    max_dev_195 = 0.0
    for s, d in per_subject_repro.items():
        sub = float(scal[s]['auc'])
        d['submitted_auc'] = sub
        d['abs_dev'] = abs(d['repro_auc'] - sub)
        d['z0_auc_first_subwindow'] = z0_auc_195[s]
        d['z0_abs_dev_vs_submitted'] = (
            None if z0_auc_195[s] is None else abs(z0_auc_195[s] - sub))
        max_dev_195 = max(max_dev_195, d['abs_dev'])

    # ---- faithful sim-20s row (7 subjects, module's own functions) ---------
    log("job1: sim-20s row (7 subjects)")
    subject_data = {}
    for subj in HW_SUBSET:
        el, lb = q20.load_subject_data(subj, channels)
        if el:
            yy = np.array([label_map.get(l, 0) for l in lb])
            if len(np.unique(yy)) == 2:
                subject_data[subj] = (el, lb, yy)
    per_subject_20 = {}
    ideal_rows_20 = []
    W = q20.WINDOW_SAMPLES
    for test_subject in subject_data:
        train_ictal, train_inter = [], []
        for subj, (el, lb, _) in subject_data.items():
            if subj != test_subject:
                for eeg, l in zip(el, lb):
                    e = eeg[:, :W]
                    if e.shape[1] >= W:
                        (train_ictal if l in ('ictal', 'preictal')
                         else train_inter).append(e)
        if not train_ictal or not train_inter:
            continue
        ictal_params = [q20.extract_plv_params(e, fs=q20.FS, band=q20.BAND)
                        for e in train_ictal[:50]]
        inter_params = [q20.extract_plv_params(e, fs=q20.FS, band=q20.BAND)
                        for e in train_inter[:50]]
        ictal_avg = average_params(ictal_params, n_ch)
        inter_avg = average_params(inter_params, n_ch)
        sv_i = Statevector.from_instruction(
            create_multichannel_circuit(ictal_avg))
        sv_t = Statevector.from_instruction(
            create_multichannel_circuit(inter_avg))
        pi_ = np.abs(sv_i.data) ** 2
        pt_ = np.abs(sv_t.data) ** 2
        el, lb, yy = subject_data[test_subject]
        scores, yv = [], []
        for idx, eeg in enumerate(el):
            e = eeg[:, :W]
            if e.shape[1] < W:
                continue
            params = q20.extract_plv_params(e, fs=q20.FS, band=q20.BAND)
            sv = Statevector.from_instruction(
                create_multichannel_circuit(params))
            p = np.abs(sv.data) ** 2
            fid_i = float(np.sum(np.sqrt(p * pi_)) ** 2)
            fid_t = float(np.sum(np.sqrt(p * pt_)) ** 2)
            scores.append(fid_i - fid_t)
            yv.append(int(yy[idx]))
            ideal_rows_20.append(dict(
                patient=test_subject, segment_idx=idx, y=int(yy[idx]),
                z_expect=ancilla_z_from_sv(sv),
                fidelity_score=fid_i - fid_t))
        auc = roc_auc_score(yv, scores)
        z0s = [r['z_expect'] for r in ideal_rows_20
               if r['patient'] == test_subject]
        try:
            z0_auc = roc_auc_score(yv, z0s)
        except ValueError:
            z0_auc = None
        per_subject_20[test_subject] = dict(repro_auc=float(auc),
                                            z0_auc=z0_auc, n=len(yv))
        log(f"job1: sim-20s {test_subject} repro AUC {auc:.6f} | <Z0> AUC "
            f"{z0_auc if z0_auc is None else round(z0_auc, 4)}")

    sub20 = json.load(open(PROJECT_ROOT /
                           'results/window_analysis/'
                           'quantum_20s_simulation.json',
                           encoding='utf-8'))['per_subject']
    max_dev_20 = 0.0
    for s, d in per_subject_20.items():
        sub = float(sub20[s]['raw_auc'])
        d['submitted_auc'] = sub
        d['abs_dev'] = abs(d['repro_auc'] - sub)
        d['z0_abs_dev_vs_submitted'] = (
            None if d['z0_auc'] is None else abs(d['z0_auc'] - sub))
        max_dev_20 = max(max_dev_20, d['abs_dev'])

    # ---- write outputs ------------------------------------------------------
    for name, rows in [('ideal_scores_1p95s.csv', ideal_rows_195),
                       ('ideal_scores_20s.csv', ideal_rows_20)]:
        with open(OUT_DIR / name, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    repro = dict(
        tolerance=REPRO_TOL,
        sim_1p95s=dict(
            per_subject=per_subject_repro,
            max_abs_dev_faithful=max_dev_195,
            faithful_reproduced_to_tol=bool(max_dev_195 < REPRO_TOL),
            note='faithful = dual-template amplitude-fidelity MAX pipeline; '
                 'z0 = the ancilla observable the P2REV-03 prompt specifies',
        ),
        sim_20s=dict(
            per_subject=per_subject_20,
            max_abs_dev_faithful=max_dev_20,
            faithful_reproduced_to_tol=bool(max_dev_20 < REPRO_TOL),
        ),
        hardware_rows='not reproducible by simulation by construction '
                      '(physical counts); compared distributionally in '
                      'Jobs 3-4',
        checkpoint_z0=dict(
            z0_reproduces_submitted=False,
            reason='no submitted row was scored by the ancilla <Z_0>; all '
                   'four rows are dual-template fidelity classifiers (see '
                   'run_inventory.md)',
        ),
        runtime_min=round((time.time() - t0) / 60, 1),
    )
    with open(OUT_DIR / 'job1_reproduction.json', 'w', encoding='utf-8') as f:
        json.dump(repro, f, indent=2)
    log(f"job1: done in {repro['runtime_min']} min; "
        f"faithful max dev 1.95s {max_dev_195:.2e}, 20s {max_dev_20:.2e}")
    log("job1 CHECKPOINT: the <Z_0> reproduction the prompt specifies FAILS "
        "by construction (no row used <Z_0>); stopping before Jobs 2-6 per "
        "instruction.")
    return repro


# =============================================================================
# Resume stages (Jobs 2, 3, 5, 6 under the 2026-10-31 revision of scope)
# =============================================================================

N_BOOT = 2000
N_PERM = 1000
N_SHOT_SEEDS = 200
SHOTS = 1024
BOOT_SEED = 42
SUBMITTED_HW_195 = None   # loaded in helpers
P2REV02_DIR = PROJECT_ROOT / 'results' / 'p2rev02'


def _p2rev02_module():
    return load_script('p2rev02_classical_rerun')


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def build_params_cache():
    """(a,b,c) per (segment, sub-window) for the Job-1 segment set, plus the
    20 s-window params for the 7-subset, cached to params_cache.npz."""
    cache_path = OUT_DIR / 'params_cache.npz'
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    qcs = load_script('quantum_channel_scaling')
    channels = ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
                "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]
    log("job2: loading segments for params cache")
    eeg_list, labels, subjects = qcs.load_all_segments(channels)
    rows_a, rows_b, rows_c = [], [], []
    seg_idx, sw_idx, pats, labs = [], [], [], []
    a20, b20, c20, seg20, pat20, lab20 = [], [], [], [], [], []
    w20 = int(20.0 * qcs.FS)
    for i, eeg in enumerate(eeg_list):
        for k, sw in enumerate(qcs.extract_plv_params_subwindows(
                eeg, fs=qcs.FS, band=qcs.BAND)):
            arr = np.array(sw)
            rows_a.append(arr[:, 0])
            rows_b.append(arr[:, 1])
            rows_c.append(arr[:, 2])
            seg_idx.append(i)
            sw_idx.append(k)
            pats.append(subjects[i])
            labs.append(labels[i])
        if subjects[i] in HW_SUBSET and eeg.shape[1] >= w20:
            arr = np.array(qcs.extract_plv_params(
                eeg[:, :w20], fs=qcs.FS, band=qcs.BAND))
            a20.append(arr[:, 0])
            b20.append(arr[:, 1])
            c20.append(arr[:, 2])
            seg20.append(i)
            pat20.append(subjects[i])
            lab20.append(labels[i])
    out = dict(a=np.array(rows_a), b=np.array(rows_b), c=np.array(rows_c),
               seg_idx=np.array(seg_idx), sw_idx=np.array(sw_idx),
               patient=np.array(pats), label=np.array(labs),
               a20=np.array(a20), b20=np.array(b20), c20=np.array(c20),
               seg20=np.array(seg20), patient20=np.array(pat20),
               label20=np.array(lab20))
    np.savez_compressed(cache_path, **out)
    log(f"job2: params cache written ({len(seg_idx)} sub-windows, "
        f"{len(seg20)} 20s windows)")
    return out


def stage_job2():
    """Classical implementation: verify numpy <Z_0> against the Job-1 Qiskit
    values, dead-zone fractions, and the 22-patient / both-labellings
    re-evaluation on the P2REV-02 windows."""
    from qdnu.agate_observable import (agate_ancilla_z, agate_balance,
                                       dead_zone_flags,
                                       verify_balance_identity)
    t0 = time.time()
    pc = build_params_cache()

    # numpy <Z_0> and balance for every sub-window / 20 s window
    log("job2: numpy <Z_0> for all cached params")
    z_np = np.array([agate_ancilla_z(pc['a'][i], pc['b'][i], pc['c'][i])
                     for i in range(len(pc['a']))])
    bal_np = np.array([agate_balance(pc['a'][i], pc['b'][i], pc['c'][i])
                       for i in range(len(pc['a']))])
    z20_np = np.array([agate_ancilla_z(pc['a20'][i], pc['b20'][i],
                                       pc['c20'][i])
                       for i in range(len(pc['a20']))])
    bal20_np = np.array([agate_balance(pc['a20'][i], pc['b20'][i],
                                       pc['c20'][i])
                         for i in range(len(pc['a20']))])
    np.savez_compressed(OUT_DIR / 'z0_balance_cache.npz', z_np=z_np,
                        bal_np=bal_np, z20_np=z20_np, bal20_np=bal20_np)

    # verification vs the Job 1 Qiskit statevector values
    import csv as _csv
    sv195 = {}
    with open(OUT_DIR / 'ideal_scores_1p95s.csv', encoding='utf-8') as f:
        for r in _csv.DictReader(f):
            sv195[(int(r['segment_idx']), int(r['subwindow_idx']))] = \
                float(r['z_expect'])
    err195 = max(abs(z_np[i] - sv195[(int(pc['seg_idx'][i]),
                                      int(pc['sw_idx'][i]))])
                 for i in range(len(z_np))
                 if (int(pc['seg_idx'][i]), int(pc['sw_idx'][i])) in sv195)
    # 20 s row: Job 1 enumerated segments per subject via its own loader;
    # match by (patient, within-patient ordinal)
    sv20 = {}
    with open(OUT_DIR / 'ideal_scores_20s.csv', encoding='utf-8') as f:
        rows20 = list(_csv.DictReader(f))
    from collections import defaultdict
    by_pat = defaultdict(list)
    for r in rows20:
        by_pat[r['patient']].append(float(r['z_expect']))
    err20 = 0.0
    ptr = defaultdict(int)
    for i in range(len(z20_np)):
        p = str(pc['patient20'][i])
        err20 = max(err20, abs(z20_np[i] - by_pat[p][ptr[p]]))
        ptr[p] += 1
    bal_err = verify_balance_identity()
    log(f"job2: max |qiskit - numpy| 1.95s {err195:.3e}, 20s {err20:.3e}, "
        f"balance identity {bal_err:.3e}")

    # dead-zone fractions per patient, both rows
    def zone_frac(aa, bb, cc, pats):
        out = {}
        for p in sorted(set(pats.tolist())):
            m = pats == p
            flags = [dead_zone_flags(aa[i], bb[i], cc[i])
                     for i in np.where(m)[0]]
            out[p] = dict(
                n=int(m.sum()),
                frac_amp_over_pi4=float(np.mean([f[0] for f in flags])),
                frac_phase_dead=float(np.mean([f[1] for f in flags])))
        return out

    classical_check = dict(
        max_abs_err_vs_qiskit_1p95s=float(err195),
        max_abs_err_vs_qiskit_20s=float(err20),
        balance_identity_max_err=float(bal_err),
        tolerance_required=1e-10,
        passed=bool(err195 < 1e-10 and err20 < 1e-10),
        dead_zones_1p95s_subwindows=zone_frac(pc['a'], pc['b'], pc['c'],
                                              pc['patient']),
        dead_zones_20s=zone_frac(pc['a20'], pc['b20'], pc['c20'],
                                 pc['patient20']),
        note='a, c are clipped to [0.05, 0.95] by the PLV extraction, so '
             'the amp flag marks windows with any channel a or c > pi/4 '
             '(0.785); phase flag: any channel with |sin b| or |cos b| < '
             '0.3.',
    )
    with open(OUT_DIR / 'classical_check.json', 'w', encoding='utf-8') as f:
        json.dump(classical_check, f, indent=2)

    # ---- 22 patients x both labellings on the P2REV-02 windows ------------
    log("job2: ideal_auc_22 on P2REV-02 windows (first 2.0 s per window)")
    m02 = _p2rev02_module()
    from sagemaker.train_full_cohort import read_edf_segment
    qcs = load_script('quantum_channel_scaling')
    cache02 = m02.load_cache(sorted(
        d for d in Path(m02.DATA_ROOT).iterdir()
        if d.is_dir() and d.name.startswith('chb')
        and d.name not in m02.EXCLUDE_SUBJECTS))
    rows_out = []
    z_cache = {}
    for lab in ['original', 'strict']:
        for pat in sorted(cache02):
            d = cache02[pat]
            idx, y = m02.select_windows(d, lab, 0)
            scores = []
            for j in idx:
                key = (pat, int(j))
                if key not in z_cache:
                    fn = str(d['src_file'][j])
                    st = float(d['start'][j])
                    data, fs = read_edf_segment(
                        str(Path(m02.DATA_ROOT) / pat / fn), st, st + 2.0,
                        target_channels=list(m02.QUANTUM_CHANNELS))
                    if data is None:
                        z_cache[key] = np.nan
                    else:
                        arr = np.array(qcs.extract_plv_params(
                            data, fs=fs, band=qcs.BAND))
                        z_cache[key] = agate_ancilla_z(
                            arr[:, 0], arr[:, 1], arr[:, 2])
                scores.append(z_cache[key])
            scores = np.array(scores)
            ok = ~np.isnan(scores)
            auc = _auc(y[ok], scores[ok])
            lo, hi = m02.bootstrap_auc_ci(y[ok], scores[ok])
            pp, _ = m02.perm_p_below(y[ok], scores[ok])
            rows_out.append(dict(patient=pat, labelling=lab,
                                 auc=None if auc is None else round(auc, 4),
                                 ci_lo=None if lo is None else round(lo, 4),
                                 ci_hi=None if hi is None else round(hi, 4),
                                 perm_p_below=round(pp, 4)))
            log(f"job2: ideal_auc_22 {lab} {pat} auc "
                f"{auc if auc is None else round(auc, 3)}")
    with open(OUT_DIR / 'ideal_auc_22.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['patient', 'labelling', 'auc',
                                          'ci_lo', 'ci_hi', 'perm_p_below'])
        w.writeheader()
        w.writerows(rows_out)
    log(f"job2 done in {(time.time()-t0)/60:.1f} min")
    return classical_check


def _ideal_subset_scores(pc, z_np, z20_np):
    """Per-segment ideal scores for the 7-subset, both rows.
    1.95 s row: <Z_0> of sub-window 0 (the hardware design); 20 s row:
    <Z_0> of the first-20 s window. Returns {row: {patient: (scores, labels)}}
    where labels are the segment kinds (ictal/preictal/interictal)."""
    out = {'1p95s': {}, '20s': {}}
    first = (pc['sw_idx'] == 0)
    for p in HW_SUBSET:
        m = first & (pc['patient'] == p)
        out['1p95s'][p] = (z_np[m], pc['label'][m])
        m2 = pc['patient20'] == p
        out['20s'][p] = (z20_np[m2], pc['label20'][m2])
    return out


def _labels_to_y(kinds, labelling):
    kinds = np.asarray(kinds)
    if labelling == 'original':
        keep = np.ones(len(kinds), bool)
        y = np.isin(kinds, ['ictal', 'preictal']).astype(int)
    else:
        keep = kinds != 'preictal'
        y = (kinds == 'ictal').astype(int)
    return keep, y


def stage_job3():
    """H1: finite-shot distribution around the ideal <Z_0>."""
    pc = build_params_cache()
    zc = np.load(OUT_DIR / 'z0_balance_cache.npz')
    scores = _ideal_subset_scores(pc, zc['z_np'], zc['z20_np'])
    hw195 = json.load(open(PROJECT_ROOT / 'results/hardware_validation/'
                           'ch8_heron_loso_results.json',
                           encoding='utf-8'))['ch8']['per_subject']
    hw20 = json.load(open(PROJECT_ROOT / 'results/window_analysis/'
                          'quantum_20s_hardware.json',
                          encoding='utf-8'))['per_subject']
    hw_vals = {'1p95s': {p: float(hw195[p]['auc']) for p in HW_SUBSET},
               '20s': {p: float(hw20[p]['raw_auc']) for p in HW_SUBSET}}
    rng = np.random.default_rng(7)
    rows, h1 = [], {}
    for row in ['1p95s', '20s']:
        per_seed = {p: [] for p in HW_SUBSET}
        for _ in range(N_SHOT_SEEDS):
            for p in HW_SUBSET:
                z, kinds = scores[row][p]
                keep, y = _labels_to_y(kinds, 'original')
                pr = (1 + z[keep]) / 2
                phat = rng.binomial(SHOTS, np.clip(pr, 0, 1)) / SHOTS
                a = _auc(y[keep], 2 * phat - 1)
                per_seed[p].append(a)
        inside = 0
        for p in HW_SUBSET:
            v = np.array(per_seed[p], float)
            lo, md, hi = np.percentile(v, [2.5, 50, 97.5])
            hwv = hw_vals[row][p]
            pct = float(np.mean(v <= hwv) * 100)
            ins = bool(lo <= hwv <= hi)
            inside += ins
            rows.append(dict(row=row, patient=p, p2_5=round(lo, 4),
                             p50=round(md, 4), p97_5=round(hi, 4),
                             hardware_value=round(hwv, 4),
                             hardware_percentile=round(pct, 1),
                             inside_95=int(ins)))
        overall = np.array([[np.mean([per_seed[p][s] for p in HW_SUBSET])]
                            for s in range(N_SHOT_SEEDS)]).ravel()
        lo, md, hi = np.percentile(overall, [2.5, 50, 97.5])
        hw_overall = float(np.mean(list(hw_vals[row].values())))
        rows.append(dict(row=row, patient='OVERALL(mean of patients)',
                         p2_5=round(lo, 4), p50=round(md, 4),
                         p97_5=round(hi, 4),
                         hardware_value=round(hw_overall, 4),
                         hardware_percentile=round(
                             float(np.mean(overall <= hw_overall) * 100), 1),
                         inside_95=int(lo <= hw_overall <= hi)))
        h1[row] = dict(n_inside=int(inside), n_total=len(HW_SUBSET))
    with open(OUT_DIR / 'shot_distribution.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    h1_result = dict(
        declared='hardware per-patient raw AUC inside the finite-shot 95% '
                 'interval for >= 5 of 7 patients, 1.95 s row',
        caveat='hardware AUCs come from the template-Hellinger estimator, '
               'not <Z_0> (run_inventory.md)',
        **{f'{k}_inside': v['n_inside'] for k, v in h1.items()},
        passed=bool(h1['1p95s']['n_inside'] >= 5))
    log(f"job3 H1: inside 95% interval 1.95s {h1['1p95s']['n_inside']}/7, "
        f"20s {h1['20s']['n_inside']}/7 -> "
        f"{'PASS' if h1_result['passed'] else 'FAIL'}")
    return h1_result


def stage_job5():
    """Q1: permutation p for 'below chance', ideal scores + hardware 20 s."""
    pc = build_params_cache()
    zc = np.load(OUT_DIR / 'z0_balance_cache.npz')
    scores = _ideal_subset_scores(pc, zc['z_np'], zc['z20_np'])
    m02 = _p2rev02_module()
    rows = []
    q1_sig = {}
    for lab in ['original', 'strict']:
        for p in HW_SUBSET:
            z, kinds = scores['1p95s'][p]
            keep, y = _labels_to_y(kinds, lab)
            a = _auc(y[keep], z[keep])
            pp, _ = m02.perm_p_below(y[keep], z[keep])
            rows.append(dict(row='ideal_1p95s', patient=p, labelling=lab,
                             auc=None if a is None else round(a, 4),
                             perm_p_below=round(pp, 4)))
            if lab == 'original':
                q1_sig[p] = pp
    hw20 = json.load(open(PROJECT_ROOT / 'results/window_analysis/'
                          'quantum_20s_hardware.json',
                          encoding='utf-8'))['per_subject']
    for p in HW_SUBSET:
        y = np.array(hw20[p]['y_true'])
        s = np.array(hw20[p]['scores'])
        pp, _ = m02.perm_p_below(y, s)
        rows.append(dict(row='hardware_20s', patient=p, labelling='original',
                         auc=round(float(hw20[p]['raw_auc']), 4),
                         perm_p_below=round(pp, 4)))
    with open(OUT_DIR / 'permutation_7.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['row', 'patient', 'labelling',
                                          'auc', 'perm_p_below'])
        w.writeheader()
        w.writerows(rows)
    # Tier 3 below-chance patients from P2REV-02
    t3_sig = set()
    with open(P2REV02_DIR / 'per_patient_auc.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r['labelling'], r['seed'], r['method']) == \
                    ('original', '0', 'tier3') and r['perm_p']:
                if float(r['perm_p']) < 0.05:
                    t3_sig.add(r['patient'])
    ideal_sig = {p for p, v in q1_sig.items() if v < 0.05}
    both = ideal_sig & t3_sig & set(HW_SUBSET)
    q1 = dict(
        declared='retain the 5.2 sentence only if >= 3 of 7 subset patients '
                 'below chance in ideal sim at p<0.05 AND the same patients '
                 'below chance in Tier 3',
        ideal_below_chance=sorted(ideal_sig),
        tier3_below_chance_all22=sorted(t3_sig),
        both=sorted(both),
        passed=bool(len(ideal_sig) >= 3 and ideal_sig <= t3_sig),
        note='hardware 1.95 s row has no per-segment scores; no interval '
             'fabricated (stated in run_inventory.md and the results doc)')
    log(f"job5 Q1: ideal below-chance {sorted(ideal_sig)}; Tier3 "
        f"{sorted(t3_sig)} -> {'PASS' if q1['passed'] else 'FAIL'}")
    return q1


def stage_job6():
    """G2: the sign table."""
    pc = build_params_cache()
    zc = np.load(OUT_DIR / 'z0_balance_cache.npz')
    scores = _ideal_subset_scores(pc, zc['z_np'], zc['z20_np'])
    bal_first = {}
    first = (pc['sw_idx'] == 0)
    for p in HW_SUBSET:
        m = first & (pc['patient'] == p)
        bal_first[p] = (zc['bal_np'][m], pc['label'][m])
    geo, t3 = {}, {}
    with open(P2REV02_DIR / 'geometric_sign.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            geo[(r['patient'], r['labelling'])] = r
    with open(P2REV02_DIR / 'per_patient_auc.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r['seed'], r['method']) == ('0', 'tier3') and r['auc']:
                t3[(r['patient'], r['labelling'])] = \
                    1 if float(r['auc']) >= 0.5 else -1
    hw195 = json.load(open(PROJECT_ROOT / 'results/hardware_validation/'
                           'ch8_heron_loso_results.json',
                           encoding='utf-8'))['ch8']['per_subject']
    hw20 = json.load(open(PROJECT_ROOT / 'results/window_analysis/'
                          'quantum_20s_hardware.json',
                          encoding='utf-8'))['per_subject']

    def sgn(x):
        return None if x is None else (1 if x >= 0.5 else -1)

    rows = []
    for lab in ['original', 'strict']:
        for p in HW_SUBSET:
            z, kinds = scores['1p95s'][p]
            keep, y = _labels_to_y(kinds, lab)
            ideal_sign = sgn(_auc(y[keep], z[keep]))
            bl, bkinds = bal_first[p]
            bkeep, by = _labels_to_y(bkinds, lab)
            bal_sign = int(np.sign(np.mean(bl[bkeep & (by == 1)]) -
                                   np.mean(bl[bkeep & (by == 0)])))
            g = geo.get((p, lab), {})
            rows.append(dict(
                patient=p, labelling=lab,
                pi=g.get('pi'), cosine=g.get('cosine'),
                tier3_sign=t3.get((p, lab)),
                ideal_sim_sign=ideal_sign,
                hw_1p95s_sign=(sgn(float(hw195[p]['auc']))
                               if lab == 'original' else None),
                hw_20s_sign=(sgn(float(hw20[p]['raw_auc']))
                             if lab == 'original' else None),
                balance_sign=bal_sign))
    with open(OUT_DIR / 'sign_table.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # pairwise agreement, original labelling
    cols = ['pi', 'tier3_sign', 'ideal_sim_sign', 'hw_1p95s_sign',
            'hw_20s_sign', 'balance_sign']
    orig = [r for r in rows if r['labelling'] == 'original']
    agree = {}
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            n = sum(1 for r in orig
                    if r[c1] not in (None, '') and r[c2] not in (None, '')
                    and int(r[c1]) == int(r[c2]))
            agree[f'{c1}|{c2}'] = f'{n}/7'
    n_g2 = int(agree['pi|ideal_sim_sign'].split('/')[0])
    g2 = dict(declared='ideal-sim sign agrees with pi_p in >= 5 of 7, '
                       'original labelling; reported for strict',
              pairwise_agreement_original=agree,
              ideal_vs_pi_original=n_g2,
              ideal_vs_pi_strict=sum(
                  1 for r in rows if r['labelling'] == 'strict'
                  and r['pi'] not in (None, '') and r['ideal_sim_sign']
                  is not None and int(r['pi']) == int(r['ideal_sim_sign'])),
              passed=bool(n_g2 >= 5))
    log(f"job6 G2: ideal-sim vs pi_p {n_g2}/7 original -> "
        f"{'PASS' if g2['passed'] else 'FAIL'}")
    return g2


def stage_finalize(h1, q1, g2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (OUT_DIR / 'figures').mkdir(exist_ok=True)
    cc = json.load(open(OUT_DIR / 'classical_check.json', encoding='utf-8'))
    tests = dict(
        scope='Jobs 2,3,5,6 under the 2026-10-31 revision; Job 4 / H2 '
              'descoped (expectation block: "H2 then adds nothing")',
        H1=h1, H2='descoped - not evaluated', Q1=q1, G2=g2,
        classical_check=dict(
            max_err_1p95s=cc['max_abs_err_vs_qiskit_1p95s'],
            max_err_20s=cc['max_abs_err_vs_qiskit_20s'],
            passed=cc['passed']))
    with open(OUT_DIR / 'tests.json', 'w', encoding='utf-8') as f:
        json.dump(tests, f, indent=2)

    # figure: per patient x 2 rows - ideal point, shot interval, hw point
    shot = {}
    with open(OUT_DIR / 'shot_distribution.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            shot[(r['row'], r['patient'])] = r
    pc = build_params_cache()
    zc = np.load(OUT_DIR / 'z0_balance_cache.npz')
    scores = _ideal_subset_scores(pc, zc['z_np'], zc['z20_np'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, row, title in [(axes[0], '1p95s', '1.95 s row (first 2.0 s '
                            'sub-window)'),
                           (axes[1], '20s', '20 s row')]:
        for i, p in enumerate(HW_SUBSET):
            r = shot[(row, p)]
            lo, hi = float(r['p2_5']), float(r['p97_5'])
            z, kinds = scores[row][p]
            keep, y = _labels_to_y(kinds, 'original')
            ideal = _auc(y[keep], z[keep])
            ax.plot([i, i], [lo, hi], color='#0072B2', lw=5, alpha=0.45,
                    solid_capstyle='round',
                    label='shot-noise 95 %' if i == 0 else None)
            ax.plot(i, ideal, 'o', color='#0072B2', ms=6,
                    label='ideal $\\langle Z_0\\rangle$' if i == 0 else None)
            ax.plot(i, float(r['hardware_value']), 'D', color='#D55E00',
                    ms=6, label='hardware (template estimator)'
                    if i == 0 else None)
        ax.axhline(0.5, color='#666666', lw=1, ls='--')
        ax.set_xticks(range(7))
        ax.set_xticklabels(HW_SUBSET, rotation=45, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, lw=0.5, axis='y')
    axes[0].set_ylabel('per-patient raw AUC')
    axes[0].legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('Ideal fixed observable, its shot-noise band, and the '
                 'hardware values (noise-model band descoped with Job 4)',
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'figures' / 'shot_vs_noise_vs_hardware.png',
                dpi=150)
    plt.close(fig)
    log("finalize: tests.json + figure written")
    return tests


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='job0+job1',
                    choices=['job0', 'job1', 'job0+job1', 'resume',
                             'resume-light'])
    args = ap.parse_args()
    if args.stage in ('job0', 'job0+job1'):
        stage_job0()
    if args.stage in ('job1', 'job0+job1'):
        stage_job1()
    if args.stage in ('resume', 'resume-light'):
        t0 = time.time()
        if args.stage == 'resume':
            stage_job2()
        h1 = stage_job3()
        q1 = stage_job5()
        g2 = stage_job6()
        stage_finalize(h1, q1, g2)
        log(f"resume (jobs 2,3,5,6) TOTAL {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
