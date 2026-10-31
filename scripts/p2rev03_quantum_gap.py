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
REVISION OF SCOPE AFTER THE JOB 1 CHECKPOINT (2026-10-31, verbatim from
docs/PROMPT_P2REV_03_quantum_gap.md)
================================================================================

Jim's decision: anything that is just removed or fixed for this revision is
fine; this is about pulling back. Jobs 2-6 replace the originals. The two
"1.95 s" rows (hardware and simulation) are REMOVED from the revision - the
window is 2.0 s in the code, the two rows used different estimators, and the
hardware row has no per-segment scores - so no further computation touches
them. Everything below is on the 20 s rows only:
results/window_analysis/quantum_20s_hardware.json (per-subject scores,
y_true) and quantum_20s_simulation.json (reproduced to 0.0 in Job 1).
Jobs 3-6 as originally written (binomial on <Z_0>, FakeTorino, 1.95 s) are
cancelled. Do not install qiskit_ibm_runtime.

[Historical note: an interim resume on 2026-10-31 ran the original Jobs
2/3/5/6 on the fixed <Z_0> observable before this section reached the file;
its outputs are preserved under results/p2rev03/z0_analysis/ and analysed
the fixed observable, not the submitted classifier.]

================================================================================
REVISED PRE-DECLARED BLOCKS (verbatim from the prompt's revision of scope)
================================================================================

H1 (revised; Job 3 - shot-noise interval for the 20 s rows)
    For each of the 7 patients and each LOSO fold as the 20 s simulation
    script builds it: take the exact statevector probability distributions
    of the two template circuits and of each test circuit. Draw 1024-shot
    multinomial samples of BOTH templates and every test circuit, score each
    test window by Hellinger fidelity exactly as quantum_20s_hardware.py
    does (fid_ictal - fid_inter on counts), compute per-patient AUC and the
    7-patient overall AUC. 200 seeds. Report per patient and overall:
    2.5 / 50 / 97.5 percentiles, the exact-statevector value (the submitted
    simulation row, Bhattacharyya on probabilities), and the hardware value
    with the percentile it sits at.
    Pre-declared H1 (revised): the hardware 20 s per-patient AUC lies inside
    the shot-noise 95 % interval of the simulated template classifier for
    >= 5 of 7 patients. If it passes, the manuscript's simulation-hardware
    section is replaced by one sentence giving the percentile. If it fails,
    the manuscript reports the hardware values with their intervals and says
    the difference from simulation is not explained by shot noise; no
    noise-model run follows in this revision.

Q1 (revised; Job 4 - permutation and bootstrap on the 20 s rows)
    Per patient: 95 % stratified bootstrap (2000) and permutation p_below
    (1000 within-patient label permutations, P2REV-02 definition) on (i) the
    hardware 20 s per-window scores and (ii) the exact simulation 20 s
    scores. Both labellings are NOT available for these rows (the segments
    are fixed by the hardware run); say so.
    Pre-declared Q1 (revised): the section-5.2 sentence that sub-chance AUC
    is an explicit polarity readout is retained only if >= 3 of 7 patients
    are below chance at p < 0.05 on the hardware 20 s row AND the same
    patients are below chance in Tier 3
    (results/p2rev02/per_patient_auc.csv, original labelling, perm_p).
    Otherwise the sentence is removed.

G2 (revised; Job 5 - sign table)
    Seven patients, one table: pi_p and cosine (results/p2rev02/
    geometric_sign.csv, original labelling), Tier 3 sign and CI, hardware
    20 s sign and CI, simulation 20 s sign and CI, and one supplementary
    column: the sign of the ancilla <Z_0> AUC from Job 1's
    ideal_scores_20s.csv, labelled "fixed observable described in submitted
    section 3.5; not what produced any reported number". Agreement counts
    for every pair.
    Pre-declared G2 (revised): the simulation 20 s sign agrees with pi_p in
    >= 5 of 7.
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
# REVISED JOBS 2-6 (2026-10-31 revision of scope: 20 s rows only)
# =============================================================================

Z0_DIR = OUT_DIR / 'z0_analysis'    # archived fixed-observable run + caches


def _load_20s_params():
    """20 s-window (a,b,c) per segment for the 7 subjects, from the verified
    cache (identical to the Job 1 Qiskit inputs; reproduction dev 0.0)."""
    z = np.load(Z0_DIR / 'params_cache.npz', allow_pickle=True)
    return (z['a20'], z['b20'], z['c20'],
            z['patient20'].astype(str), z['label20'].astype(str))


def _avg_params(plist):
    """Average (a,b,c) parameter sets exactly as the run scripts did
    (circular mean for b, clip for c)."""
    arr = np.array(plist)            # (n, 8, 3)
    avg = []
    for ch in range(arr.shape[1]):
        avg_a = float(np.mean(arr[:, ch, 0]))
        sins = np.mean(np.sin(arr[:, ch, 1]))
        coss = np.mean(np.cos(arr[:, ch, 1]))
        avg_b = float(np.arctan2(sins, coss) % (2 * np.pi))
        avg_c = float(np.clip(np.mean(arr[:, ch, 2]), 0.05, 0.95))
        avg.append((avg_a, avg_b, avg_c))
    return avg


def stage_rev3():
    """H1 (revised): shot-noise interval of the simulated template
    classifier, 20 s row, 200 seeds."""
    from qiskit.quantum_info import Statevector
    from QA1.multichannel_circuit import create_multichannel_circuit
    t0 = time.time()
    a20, b20, c20, pat20, lab20 = _load_20s_params()
    n = len(pat20)
    log(f"rev3: probability vectors for {n} test circuits")
    probs = np.empty((n, 2 ** 17))
    for i in range(n):
        sv = Statevector.from_instruction(create_multichannel_circuit(
            list(zip(a20[i], b20[i], c20[i]))))
        probs[i] = np.abs(sv.data) ** 2

    hw = json.load(open(PROJECT_ROOT / 'results/window_analysis/'
                        'quantum_20s_hardware.json',
                        encoding='utf-8'))['per_subject']
    sim = json.load(open(PROJECT_ROOT / 'results/window_analysis/'
                         'quantum_20s_simulation.json',
                         encoding='utf-8'))['per_subject']

    folds = {}
    for test in HW_SUBSET:
        tr = [i for i in range(n) if pat20[i] != test]
        ict = [i for i in tr if lab20[i] in ('ictal', 'preictal')][:50]
        itr = [i for i in tr if lab20[i] == 'interictal'][:50]
        tmpl = {}
        for name, idxs in [('ictal', ict), ('inter', itr)]:
            avg = _avg_params([np.column_stack([a20[i], b20[i], c20[i]])
                               for i in idxs])
            sv = Statevector.from_instruction(
                create_multichannel_circuit(avg))
            tmpl[name] = np.abs(sv.data) ** 2
        te = [i for i in range(n) if pat20[i] == test]
        y = np.array([1 if lab20[i] in ('ictal', 'preictal') else 0
                      for i in te])
        folds[test] = dict(tmpl=tmpl, test_idx=te, y=y)
        # sanity: exact Bhattacharyya reproduces the submitted sim row
        ex = [float(np.sum(np.sqrt(probs[i] * tmpl['ictal'])) ** 2 -
                    np.sum(np.sqrt(probs[i] * tmpl['inter'])) ** 2)
              for i in te]
        dev = abs(_auc(y, ex) - float(sim[test]['raw_auc']))
        log(f"rev3: fold {test} exact-sim sanity dev {dev:.2e}")

    rng = np.random.default_rng(7)
    per_seed = {p: np.empty(200) for p in HW_SUBSET}
    pooled = np.empty(200)
    log("rev3: multinomial sampling, 200 seeds")
    for s in range(200):
        ys_all, sc_all = [], []
        for test in HW_SUBSET:
            f = folds[test]
            ct_i = rng.multinomial(SHOTS, f['tmpl']['ictal']).astype(float)
            ct_t = rng.multinomial(SHOTS, f['tmpl']['inter']).astype(float)
            sq_i, sq_t = np.sqrt(ct_i), np.sqrt(ct_t)
            scores = []
            for i in f['test_idx']:
                c = rng.multinomial(SHOTS, probs[i]).astype(float)
                sq = np.sqrt(c)
                fid_i = float(sq @ sq_i) ** 2 / SHOTS ** 2
                fid_t = float(sq @ sq_t) ** 2 / SHOTS ** 2
                scores.append(fid_i - fid_t)
            per_seed[test][s] = _auc(f['y'], scores)
            ys_all.extend(f['y'].tolist())
            sc_all.extend(scores)
        pooled[s] = _auc(ys_all, sc_all)
        if s % 50 == 49:
            log(f"rev3: seed {s + 1}/200")

    rows, inside = [], 0
    for p in HW_SUBSET:
        v = per_seed[p]
        lo, md, hi = np.percentile(v, [2.5, 50, 97.5])
        hwv = float(hw[p]['raw_auc'])
        ins = bool(lo <= hwv <= hi)
        inside += ins
        rows.append(dict(patient=p, p2_5=round(lo, 4), p50=round(md, 4),
                         p97_5=round(hi, 4),
                         sim_exact=round(float(sim[p]['raw_auc']), 4),
                         hardware_value=round(hwv, 4),
                         hardware_percentile=round(
                             float(np.mean(v <= hwv) * 100), 1),
                         inside_95=int(ins)))
    mean_seed = np.mean([per_seed[p] for p in HW_SUBSET], axis=0)
    for name, dist, hwv, simv in [
            ('OVERALL(mean of patients)', mean_seed,
             float(np.mean([hw[p]['raw_auc'] for p in HW_SUBSET])),
             float(np.mean([sim[p]['raw_auc'] for p in HW_SUBSET]))),
            ('OVERALL(pooled)', pooled,
             0.3891, 0.3863)]:
        lo, md, hi = np.percentile(dist, [2.5, 50, 97.5])
        rows.append(dict(patient=name, p2_5=round(lo, 4), p50=round(md, 4),
                         p97_5=round(hi, 4), sim_exact=round(simv, 4),
                         hardware_value=round(hwv, 4),
                         hardware_percentile=round(
                             float(np.mean(dist <= hwv) * 100), 1),
                         inside_95=int(lo <= hwv <= hi)))
    with open(OUT_DIR / 'shot_distribution_20s.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    h1 = dict(declared='hardware 20 s per-patient AUC inside the shot-noise '
                       '95% interval of the simulated template classifier '
                       'for >= 5 of 7',
              n_inside=int(inside), n_total=7, passed=bool(inside >= 5),
              runtime_min=round((time.time() - t0) / 60, 1))
    log(f"rev3 H1(revised): {inside}/7 inside -> "
        f"{'PASS' if h1['passed'] else 'FAIL'} "
        f"({h1['runtime_min']} min)")
    return h1


def stage_rev4():
    """Q1 (revised): bootstrap + permutation on the two 20 s rows."""
    m02 = _p2rev02_module()
    hw = json.load(open(PROJECT_ROOT / 'results/window_analysis/'
                        'quantum_20s_hardware.json',
                        encoding='utf-8'))['per_subject']
    sim_rows = {}
    with open(OUT_DIR / 'ideal_scores_20s.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            sim_rows.setdefault(r['patient'], []).append(
                (int(r['y']), float(r['fidelity_score'])))
    rows = []
    for name, data in [('hw', {p: (np.array(hw[p]['y_true']),
                                   np.array(hw[p]['scores']))
                               for p in HW_SUBSET}),
                       ('sim', {p: (np.array([t[0] for t in sim_rows[p]]),
                                    np.array([t[1] for t in sim_rows[p]]))
                                for p in HW_SUBSET})]:
        for p in HW_SUBSET:
            y, s = data[p]
            a = _auc(y, s)
            lo, hi = m02.bootstrap_auc_ci(y, s)
            pp, _ = m02.perm_p_below(y, s)
            rows.append(dict(row=name, patient=p, auc=round(a, 4),
                             ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                             perm_p_below=round(pp, 4)))
    with open(OUT_DIR / 'per_patient_20s.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['row', 'patient', 'auc', 'ci_lo',
                                          'ci_hi', 'perm_p_below'])
        w.writeheader()
        w.writerows(rows)
    hw_below = sorted(r['patient'] for r in rows
                      if r['row'] == 'hw' and r['perm_p_below'] < 0.05)
    t3_below = set()
    with open(P2REV02_DIR / 'per_patient_auc.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r['labelling'], r['seed'], r['method']) == \
                    ('original', '0', 'tier3') and r['perm_p'] \
                    and float(r['perm_p']) < 0.05:
                t3_below.add(r['patient'])
    q1 = dict(declared='retain 5.2 sentence only if >= 3/7 patients below '
                       'chance at p<0.05 on the hardware 20 s row AND the '
                       'same patients below chance in Tier 3; otherwise the '
                       'sentence is removed',
              hw_below_chance=hw_below,
              tier3_below_chance=sorted(t3_below),
              passed=bool(len(hw_below) >= 3 and
                          set(hw_below) <= t3_below),
              labelling_note='only the original labelling exists for these '
                             'rows; the segments are fixed by the hardware '
                             'run')
    log(f"rev4 Q1(revised): hw below-chance {hw_below} -> "
        f"{'PASS' if q1['passed'] else 'FAIL'}")
    return q1, rows


def stage_rev5(pp_rows):
    """G2 (revised): the sign table, 20 s rows."""
    geo = {}
    with open(P2REV02_DIR / 'geometric_sign.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['labelling'] == 'original':
                geo[r['patient']] = r
    t3 = {}
    with open(P2REV02_DIR / 'per_patient_auc.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if (r['labelling'], r['seed'], r['method']) == \
                    ('original', '0', 'tier3') and r['auc']:
                t3[r['patient']] = r
    z0 = {}
    with open(OUT_DIR / 'ideal_scores_20s.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            z0.setdefault(r['patient'], []).append(
                (int(r['y']), float(r['z_expect'])))
    pp = {(r['row'], r['patient']): r for r in pp_rows}

    def sgn(a):
        return 1 if a >= 0.5 else -1

    rows = []
    for p in HW_SUBSET:
        y = [t[0] for t in z0[p]]
        s = [t[1] for t in z0[p]]
        z0_sign = sgn(_auc(y, s))
        rows.append(dict(
            patient=p,
            pi=int(geo[p]['pi']), cosine=geo[p]['cosine'],
            tier3_sign=sgn(float(t3[p]['auc'])),
            tier3_ci=f"[{t3[p]['ci_lo']}, {t3[p]['ci_hi']}]",
            hw20_sign=sgn(pp[('hw', p)]['auc']),
            hw20_ci=f"[{pp[('hw', p)]['ci_lo']}, {pp[('hw', p)]['ci_hi']}]",
            sim20_sign=sgn(pp[('sim', p)]['auc']),
            sim20_ci=f"[{pp[('sim', p)]['ci_lo']}, "
                     f"{pp[('sim', p)]['ci_hi']}]",
            z0_sign_supplementary=z0_sign))
    with open(OUT_DIR / 'sign_table_20s.csv', 'w', newline='',
              encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
        f.write('# z0_sign_supplementary: fixed observable described in '
                'submitted section 3.5; not what produced any reported '
                'number\n')
    cols = ['pi', 'tier3_sign', 'hw20_sign', 'sim20_sign',
            'z0_sign_supplementary']
    agree = {}
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            agree[f'{c1}|{c2}'] = \
                f"{sum(1 for r in rows if r[c1] == r[c2])}/7"
    n_g2 = sum(1 for r in rows if r['sim20_sign'] == r['pi'])
    g2 = dict(declared='simulation 20 s sign agrees with pi_p in >= 5 of 7',
              sim20_vs_pi=n_g2, pairwise_agreement=agree,
              passed=bool(n_g2 >= 5))
    log(f"rev5 G2(revised): sim20 vs pi {n_g2}/7 -> "
        f"{'PASS' if g2['passed'] else 'FAIL'}")
    return g2


def stage_rev_finalize(h1, q1, g2, pp_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tests = dict(
        scope='revised Jobs 2-6 (2026-10-31): 20 s rows only; 1.95 s rows '
              'removed from the revision; earlier fixed-observable analysis '
              'archived under z0_analysis/',
        H1_revised=h1, Q1_revised=q1, G2_revised=g2)
    with open(OUT_DIR / 'tests.json', 'w', encoding='utf-8') as f:
        json.dump(tests, f, indent=2)
    shot = {}
    with open(OUT_DIR / 'shot_distribution_20s.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            shot[r['patient']] = r
    pp = {(r['row'], r['patient']): r for r in pp_rows}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, p in enumerate(HW_SUBSET):
        r = shot[p]
        ax.plot([i, i], [float(r['p2_5']), float(r['p97_5'])],
                color='#0072B2', lw=6, alpha=0.4, solid_capstyle='round',
                label='template classifier, 1024-shot 95 %'
                if i == 0 else None)
        ax.plot(i, float(r['sim_exact']), 'o', color='#0072B2', ms=6,
                label='simulation (exact)' if i == 0 else None)
        h = pp[('hw', p)]
        ax.errorbar(i + 0.12, float(h['auc']),
                    yerr=[[float(h['auc']) - float(h['ci_lo'])],
                          [float(h['ci_hi']) - float(h['auc'])]],
                    fmt='D', color='#D55E00', ms=6, lw=1.4, capsize=3,
                    label='hardware (bootstrap 95 %)' if i == 0 else None)
    ax.axhline(0.5, color='#666666', lw=1, ls='--')
    ax.set_xticks(range(7))
    ax.set_xticklabels(HW_SUBSET, rotation=45, fontsize=9)
    ax.set_ylabel('per-patient raw AUC (20 s row)')
    ax.set_title('Template classifier: exact simulation, its shot-noise '
                 'band, and hardware with bootstrap CI')
    ax.grid(alpha=0.25, lw=0.5, axis='y')
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    (OUT_DIR / 'figures').mkdir(exist_ok=True)
    fig.savefig(OUT_DIR / 'figures' / 'shot_interval_vs_hardware_20s.png',
                dpi=150)
    plt.close(fig)
    log("rev finalize: tests.json + figure written")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='job0+job1',
                    choices=['job0', 'job1', 'job0+job1', 'resume',
                             'resume-light', 'revised'])
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
    if args.stage == 'revised':
        t0 = time.time()
        h1 = stage_rev3()
        q1, pp_rows = stage_rev4()
        g2 = stage_rev5(pp_rows)
        stage_rev_finalize(h1, q1, g2, pp_rows)
        log(f"revised jobs TOTAL {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
