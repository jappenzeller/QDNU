#!/usr/bin/env python3
"""
================================================================================
PROMPT P2REV-02 - The classical re-run for the Paper 2 revision
================================================================================

One pass over the classical LOSO pipeline (Tier 2A / 2B / 3, 22 CHB-MIT
patients, 20 s windows, fronto-central CH8 montage) producing:

  Job 1  two labellings (original: ictal+preictal=1; strict: ictal only)
  Job 2  10 interictal seeds, per-patient 95 % bootstrap CIs, paired tests
  Job 3  matched-dimension curves (PCA on Tier 2B, d in {8,24,36,72,144,336})
  Job 4  geometric polarity sign pi_p from the tangent space, + controls
  Job 5  rebuilds the P2REV-01 inventory requires

Feature extraction is REUSED from sagemaker/train_full_cohort.py (the code
that produced the submitted Table 1): preprocess_eeg (0.5-40 Hz bandpass,
60 Hz notch Q=30, per-channel z-score), Ledoit-Wolf covariance,
log10-sorted-eigenvalues (Tier 2A), MichaelHills 336 with MAX over 1 s
sub-windows (Tier 2B), make_spd covariances (Tier 3). The window builder
follows scripts/prompt037_per_seizure_decomposition.py (non-legacy): each
ictal/preictal segment is preprocessed whole and tiled into non-overlapping
20 s windows; a segment shorter than one tile but >= 10 s (the original
min-samples rule) contributes one whole-segment window. Interictal windows
come from the seizure-free files on disk at 40 s spacing (the original
pipeline's spacing), sampled per seed; the 037 cap_interictal pattern.

No hyperparameter is tuned anywhere in this script. XGBoost uses exactly the
constants of train_full_cohort.make_xgb_model (the run that produced the
submitted numbers); LDA is lsqr + Ledoit-Wolf auto shrinkage; TangentSpace is
pyriemann's with metric='riemann'.

Stages (--stage): cache, loso, matched, geom, stats, all (default).

Outputs: results/p2rev02/ (tracked).

================================================================================
PRE-DECLARED BLOCKS (written before anything ran; evaluated in stage 'stats')
================================================================================

D1 (Job 1, labelling robustness)
    The per-patient polarity sign under Tier 3 - sign(AUC_raw - 0.5), seed 0,
    same folds - agrees between the original and strict labellings in >= 18 of
    22 patients. A patient whose AUC is undefined under either labelling (no
    class-1 windows, or a single class) counts as a disagreement. The
    disagreeing patients are reported by name. PASS iff n_agree >= 18.

S1 (Job 2, R2 section 1.6, equivalence)
    Paired difference Tier 3 - Tier 2A of per-patient ORACLE-CALIBRATED AUC
    over the 22 patients, original labelling, seed 0.
    (i)  95 % percentile bootstrap CI of the mean paired difference, 2000
         resamples over patients, rng seed 42.
    (ii) TOST with margin +/-0.05: two one-sided paired t-tests against
         -0.05 and +0.05; tost_p = max of the two one-sided p-values.
    Equivalence is claimed ONLY if the bootstrap CI lies entirely inside
    (-0.05, +0.05). Alongside: Wilcoxon signed-rank p (zero_method='pratt';
    method='exact' when computable, otherwise the normal approximation, and
    which one is used is recorded) and the matched-pairs rank-biserial
    r = (W+ - W-)/(W+ + W-) on pratt ranks, for every pairwise comparison the
    manuscript makes: T3 vs T2A, T3 vs T2B, T2A vs T2B (calibrated AUC; raw
    also reported), both labellings.

M1 (Job 3, R2 section 1.3 / R3 section 4, dimensionality)
    AUC below means mean per-patient oracle-calibrated AUC, original
    labelling, evaluated on seed 0 (across-seed mean and SD reported
    alongside). The manuscript's dimensionality claim holds iff BOTH
    (a) Spearman rho over the six points (d, AUC(PCA_d(Tier2B) + LDA)),
        d in {8, 24, 36, 72, 144, 336}, is > 0.8, AND
    (b) at d = 36 under LDA, |AUC(PCA_36(Tier2B)) - AUC(Tier3 tangent-36)|
        <= 0.05.
    If (b) fails with Tier 2B above Tier 3 at equal dimension, the
    manuscript's claim is withdrawn and replaced by: "the Tier 2B
    representation performs better under the conditions tested."

G1 (Job 4, R2 section 1.1 / R3 sections 1-2, geometric sign)
    pi_p = sign<s_p, r> with the tangent space fit on the training cohort's
    interictal windows (seed-0 interictal sample), r = unit-normalised mean
    ictal-minus-interictal shift over the 21 training patients, s_p = the
    held-out patient's class-1-minus-class-0 mean tangent shift (uses the
    held-out patient's labels; it is a geometric quantity, not a classifier).
    Among patients whose Tier 3 seed-0 bootstrap CI excludes 0.5, pi_p agrees
    with sign(AUC_raw - 0.5) in >= 75 %. Agreement over all 22 and the cosine
    |<s_p, r>| / ||s_p|| are reported per patient. Both labellings. PASS iff
    agreement >= 75 % among CI-decided patients under the original labelling.

Controls (R3 section 2, declared with Job 4): (c1) label-flip - for every
patient x method, flipping the class labels maps AUC to exactly 1 - AUC
(tolerance 1e-9); (c2) per-patient permutation test on Tier 3 raw AUC, 1000
within-patient label permutations, p_below = (1 + #{AUC_perm <= AUC_obs}) /
1001, reported per patient.
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sagemaker.train_full_cohort import (  # noqa: E402  (reused, not rewritten)
    QUANTUM_CHANNELS, compute_covariance_ledoit_wolf, extract_eigenvalues,
    extract_tier2b_features, make_spd, normalize_channel_label,
    parse_summary_file, preprocess_eeg, read_edf_segment,
)

# =============================================================================
# CONFIGURATION (constants; nothing is selected on test data)
# =============================================================================

DATA_ROOT = Path(os.environ.get("QDNU_DATA_ROOT", "H:/Data/PythonDNU/EEG/chbmit"))
OUT_DIR = PROJECT_ROOT / "results" / "p2rev02"
CACHE_DIR = OUT_DIR / "cache"
FIG_DIR = OUT_DIR / "figures"

WINDOW_SEC = 20.0
PREICTAL_SEC = 60.0
MIN_PARTIAL_FRAC = 0.5          # original min-samples rule: >= 10 s keeps a window
INTERICTAL_SPACING = 40.0       # original pipeline: window starts every 40 s
MAX_POOL_PER_FILE = 90          # pool cap per seizure-free file
EXCLUDE_SUBJECTS = {'chb12', 'chb24'}
HW_SUBSET = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']
HW_INVERTED_195 = {'chb03', 'chb11', 'chb21'}   # results/calibration/calibrated_gap_inputs.json

SEEDS = list(range(10))
LABELLINGS = ['original', 'strict']
D_GRID = [8, 24, 36, 72, 144, 336]
N_BOOT = 2000
N_PERM = 1000
TOST_MARGIN = 0.05
BOOT_SEED = 42

# XGBoost: exactly train_full_cohort.make_xgb_model (the submitted run).
# NOTE the manuscript (section 3.3) states max_depth=5, learning_rate=0.05;
# the code that produced the submitted numbers used the values below.
XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, eval_metric='logloss', n_jobs=-1,
)
LDA_PARAMS = dict(solver='lsqr', shrinkage='auto')

# Submitted Table 1 values (for the "what changes" section of SUMMARY.md)
SUBMITTED = {
    'tier2b': dict(raw=0.952, cal=0.952, inv=0),
    'tier3':  dict(raw=0.536, cal=0.659, inv=6),
    'tier2a': dict(raw=0.521, cal=0.628, inv=10),
    'wilcoxon_t3_t2a_p': 0.4358,
}
SUBMITTED_TABLE4_QSIM = dict(chb01=-1, chb03=-1, chb05=+1, chb07=+1,
                             chb11=-1, chb14=+1, chb21=-1)

KIND_INTER, KIND_ICTAL, KIND_PREICTAL = 0, 1, 2

PALETTE = {'tier2a': '#0072B2', 'tier2b': '#D55E00', 'tier3': '#009E73',
           'lda': '#0072B2', 'xgb': '#D55E00', 'neutral': '#666666'}


def log(msg=""):
    print(f"{datetime.now():%H:%M:%S} | {msg}", flush=True)


# =============================================================================
# STAGE: cache  (one EDF pass; features for every candidate window)
# =============================================================================

def tile_segment(start: float, end: float):
    """Non-overlapping WINDOW_SEC tiles; one whole-segment window if the
    segment has no full tile but is >= MIN_PARTIAL_FRAC * WINDOW_SEC."""
    dur = end - start
    n_full = int(dur // WINDOW_SEC)
    if n_full == 0:
        return [(start, end)] if dur >= MIN_PARTIAL_FRAC * WINDOW_SEC else []
    return [(start + i * WINDOW_SEC, start + (i + 1) * WINDOW_SEC)
            for i in range(n_full)]


def edf_duration(path: Path) -> float:
    import pyedflib
    with pyedflib.EdfReader(str(path)) as f:
        return float(f.getFileDuration())


def edf_channel_info(path: Path):
    import pyedflib
    with pyedflib.EdfReader(str(path)) as f:
        labels = [normalize_channel_label(f.getLabel(i))
                  for i in range(f.signals_in_file)]
    montage = [normalize_channel_label(c) for c in QUANTUM_CHANNELS]
    return len(labels), sum(1 for m in montage if m in labels)


def cache_subject(subject_dir: Path):
    """Extract features for all candidate windows of one subject."""
    subject = subject_dir.name
    summary = list(subject_dir.glob('*-summary.txt'))
    if not summary:
        return None
    seizures = parse_summary_file(summary[0])
    seizure_files = {}
    for sz in seizures:
        seizure_files.setdefault(sz['file'], []).append(sz)

    plan = []   # (file, start, end, kind, seizure_id)
    for i, sz in enumerate(seizures, 1):
        if not (subject_dir / sz['file']).exists():
            continue
        for (a, b) in tile_segment(sz['start'], sz['end']):
            plan.append((sz['file'], a, b, KIND_ICTAL, i))
        pre_a = max(0.0, sz['start'] - PREICTAL_SEC)
        for (a, b) in tile_segment(pre_a, sz['start']):
            plan.append((sz['file'], a, b, KIND_PREICTAL, i))

    free_files = sorted(f for f in subject_dir.glob('*.edf')
                        if f.name not in seizure_files)
    for f in free_files:
        try:
            dur = edf_duration(f)
        except Exception:
            dur = 3600.0
        n = 0
        t = 0.0
        while t + WINDOW_SEC <= dur and n < MAX_POOL_PER_FILE:
            plan.append((f.name, t, t + WINDOW_SEC, KIND_INTER, 0))
            t += INTERICTAL_SPACING
            n += 1

    X2a, X2b, covs, kinds, sids, files, starts = [], [], [], [], [], [], []
    # group reads by file; segments were already tiled, so each plan row is
    # one window; preprocess each window (segment tiles were cut from the
    # whole preprocessed segment in 037 - here each tile is preprocessed
    # individually, which matches the original interictal path; filtering a
    # 20 s window vs cutting from a longer filtered segment differs only at
    # the filtfilt edges)
    for fname, a, b, kind, sid in plan:
        data, fs = read_edf_segment(str(subject_dir / fname), a, b,
                                    target_channels=QUANTUM_CHANNELS)
        if data is None or data.size == 0:
            continue
        if data.shape[1] < int(WINDOW_SEC * fs * MIN_PARTIAL_FRAC):
            continue
        data = preprocess_eeg(data, fs=int(fs))
        cov = compute_covariance_ledoit_wolf(data)
        X2a.append(extract_eigenvalues(cov))
        X2b.append(extract_tier2b_features(data, fs=int(fs), aggregation='max'))
        covs.append(make_spd(cov))
        kinds.append(kind)
        sids.append(sid)
        files.append(fname)
        starts.append(a)

    if not X2a:
        return None
    first_edf = sorted(subject_dir.glob('*.edf'))[0]
    n_ch_total, n_ch_montage = edf_channel_info(first_edf)
    return dict(
        X2a=np.array(X2a), X2b=np.nan_to_num(np.array(X2b), nan=0.0,
                                             posinf=10.0, neginf=-10.0),
        covs=np.array(covs), kind=np.array(kinds), seizure_id=np.array(sids),
        src_file=np.array(files), start=np.array(starts),
        n_seizures=len(seizures), total_ictal_s=float(
            sum(z['end'] - z['start'] for z in seizures)),
        n_ch_total=n_ch_total, n_ch_montage=n_ch_montage,
    )


def stage_cache(subjects):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for sd in subjects:
        out = CACHE_DIR / f"{sd.name}.npz"
        if out.exists():
            log(f"cache: {sd.name} already cached")
            continue
        d = cache_subject(sd)
        if d is None:
            log(f"cache: {sd.name} FAILED (no windows)")
            continue
        np.savez_compressed(out, **d)
        k = d['kind']
        log(f"cache: {sd.name}  ictal={int((k==KIND_ICTAL).sum())} "
            f"preictal={int((k==KIND_PREICTAL).sum())} "
            f"pool={int((k==KIND_INTER).sum())}")
    log(f"cache stage done in {(time.time()-t0)/60:.1f} min")


def load_cache(subjects):
    out = {}
    for sd in subjects:
        p = CACHE_DIR / f"{sd.name}.npz"
        if p.exists():
            z = np.load(p, allow_pickle=True)
            out[sd.name] = {k: z[k] for k in z.files}
    return out


# =============================================================================
# Window selection per (labelling, seed)
# =============================================================================

def n_class1_original(d):
    return int(((d['kind'] == KIND_ICTAL) | (d['kind'] == KIND_PREICTAL)).sum())


def select_windows(d, labelling: str, seed: int):
    """Indices + labels for one patient under one labelling and seed.
    Interictal: n_keep = min(pool, n_class1_original), sampled with
    rng(seed); the same interictal set serves both labellings."""
    kind = d['kind']
    pool = np.where(kind == KIND_INTER)[0]
    n_keep = min(len(pool), n_class1_original(d))
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(pool, n_keep, replace=False)) \
        if len(pool) > n_keep else pool
    if labelling == 'original':
        cls1 = np.where((kind == KIND_ICTAL) | (kind == KIND_PREICTAL))[0]
    else:
        cls1 = np.where(kind == KIND_ICTAL)[0]
    idx = np.concatenate([cls1, keep]).astype(int)
    y = np.concatenate([np.ones(len(cls1), int), np.zeros(len(keep), int)])
    return idx, y


# =============================================================================
# STAGE: loso  (Jobs 1-2 base runs: T2A, T2B, T3 x labelling x seed)
# =============================================================================

def auc_safe(y, s):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, s))


def run_fold_xgb(Xtr, ytr, Xte):
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    sc = StandardScaler()
    Xtr = sc.fit_transform(Xtr)
    Xte = sc.transform(Xte)
    m = XGBClassifier(**XGB_PARAMS)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def run_fold_lda(Xtr, ytr, Xte):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis(**LDA_PARAMS)
    lda.fit(Xtr, ytr)
    return lda.predict_proba(Xte)[:, 1]


def tangent_features(covs_tr, covs_te):
    from pyriemann.tangentspace import TangentSpace
    ts = TangentSpace(metric='riemann')
    return ts.fit_transform(covs_tr), ts.transform(covs_te)


def stage_loso(cache):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subjects = sorted(cache.keys())
    rows = []
    scores_store = {}   # (lab, seed) -> {subj: {'y':, 't2a':, 't2b':, 't3':}}
    t0 = time.time()
    for lab in LABELLINGS:
        for seed in SEEDS:
            sel = {s: select_windows(cache[s], lab, seed) for s in subjects}
            store = {}
            for test in subjects:
                tr = [s for s in subjects if s != test]
                itr = {s: sel[s] for s in tr}
                Xte_idx, yte = sel[test]
                if len(np.unique(yte)) < 2 or len(yte) == 0:
                    store[test] = dict(y=yte, t2a=None, t2b=None, t3=None)
                    continue
                ytr = np.concatenate([itr[s][1] for s in tr])
                out = {'y': yte}
                for meth, key in [('t2a', 'X2a'), ('t2b', 'X2b')]:
                    Xtr = np.vstack([cache[s][key][itr[s][0]] for s in tr])
                    Xte = cache[test][key][Xte_idx]
                    out[meth] = run_fold_xgb(Xtr, ytr, Xte)
                Ctr = np.concatenate([cache[s]['covs'][itr[s][0]] for s in tr])
                Cte = cache[test]['covs'][Xte_idx]
                try:
                    Ftr, Fte = tangent_features(Ctr, Cte)
                    out['t3'] = run_fold_lda(Ftr, ytr, Fte)
                except Exception as e:
                    log(f"loso: T3 failed {lab}/s{seed}/{test}: {e}")
                    out['t3'] = None
                store[test] = out
                for meth, dnat, clf in [('tier2a', 8, 'xgb'),
                                        ('tier2b', 336, 'xgb'),
                                        ('tier3', 36, 'lda')]:
                    s = out[{'tier2a': 't2a', 'tier2b': 't2b',
                             'tier3': 't3'}[meth]]
                    a = auc_safe(yte, s) if s is not None else None
                    rows.append(dict(patient=test, labelling=lab, method=meth,
                                     classifier=clf, d=dnat, seed=seed,
                                     auc=a))
            scores_store[(lab, seed)] = store
            aucs = [r['auc'] for r in rows
                    if r['labelling'] == lab and r['seed'] == seed
                    and r['method'] == 'tier3' and r['auc'] is not None]
            log(f"loso: {lab} seed {seed} done "
                f"(T3 mean raw {np.mean(aucs):.3f})")
    # persist
    np.save(OUT_DIR / 'loso_scores.npy', np.array([scores_store], dtype=object))
    with open(OUT_DIR / 'loso_base_auc.json', 'w') as f:
        json.dump(rows, f)
    log(f"loso stage done in {(time.time()-t0)/60:.1f} min")
    return rows, scores_store


# =============================================================================
# STAGE: matched  (Job 3)
# =============================================================================

def stage_matched(cache):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    subjects = sorted(cache.keys())
    lab = 'original'
    rows = []
    t0 = time.time()
    for seed in SEEDS:
        sel = {s: select_windows(cache[s], lab, seed) for s in subjects}
        # tangent features once per fold (shared by t3-xgb evaluation)
        for test in subjects:
            tr = [s for s in subjects if s != test]
            ytr = np.concatenate([sel[s][1] for s in tr])
            yte = sel[test][1]
            if len(np.unique(yte)) < 2:
                continue
            X2b_tr = np.vstack([cache[s]['X2b'][sel[s][0]] for s in tr])
            X2b_te = cache[test]['X2b'][sel[test][0]]
            sc = StandardScaler()
            X2b_tr_s = sc.fit_transform(X2b_tr)
            X2b_te_s = sc.transform(X2b_te)
            for d in D_GRID:
                pca = PCA(n_components=min(d, X2b_tr_s.shape[0]),
                          random_state=0)
                Ptr = pca.fit_transform(X2b_tr_s)
                Pte = pca.transform(X2b_te_s)
                for clf, fn in [('lda', run_fold_lda), ('xgb', run_fold_xgb)]:
                    s = fn(Ptr, ytr, Pte)
                    rows.append(dict(patient=test, labelling=lab,
                                     method='tier2b_pca', classifier=clf,
                                     d=d, seed=seed, auc=auc_safe(yte, s)))
            # cross pairings at native dimension
            Ctr = np.concatenate([cache[s]['covs'][sel[s][0]] for s in tr])
            Cte = cache[test]['covs'][sel[test][0]]
            try:
                Ftr, Fte = tangent_features(Ctr, Cte)
                s = run_fold_xgb(Ftr, ytr, Fte)
                rows.append(dict(patient=test, labelling=lab,
                                 method='tier3_tangent', classifier='xgb',
                                 d=36, seed=seed, auc=auc_safe(yte, s)))
            except Exception:
                pass
            X2a_tr = np.vstack([cache[s]['X2a'][sel[s][0]] for s in tr])
            X2a_te = cache[test]['X2a'][sel[test][0]]
            sc2 = StandardScaler()
            s = run_fold_lda(sc2.fit_transform(X2a_tr), ytr,
                             sc2.transform(X2a_te))
            rows.append(dict(patient=test, labelling=lab, method='tier2a',
                             classifier='lda', d=8, seed=seed,
                             auc=auc_safe(yte, s)))
        log(f"matched: seed {seed} done ({(time.time()-t0)/60:.1f} min)")
    with open(OUT_DIR / 'matched_auc.json', 'w') as f:
        json.dump(rows, f)
    log(f"matched stage done in {(time.time()-t0)/60:.1f} min")
    return rows


# =============================================================================
# STAGE: geom  (Job 4)
# =============================================================================

def stage_geom(cache):
    from pyriemann.tangentspace import TangentSpace
    subjects = sorted(cache.keys())
    out_rows = []
    t0 = time.time()
    for lab in LABELLINGS:
        sel = {s: select_windows(cache[s], lab, 0) for s in subjects}
        for test in subjects:
            tr = [s for s in subjects if s != test]
            inter_tr = np.concatenate([
                cache[s]['covs'][sel[s][0]][sel[s][1] == 0] for s in tr])
            ts = TangentSpace(metric='riemann')
            ts.fit(inter_tr)
            shifts = []
            for s in tr:
                idx, y = sel[s]
                if (y == 1).sum() == 0 or (y == 0).sum() == 0:
                    continue
                V = ts.transform(cache[s]['covs'][idx])
                shifts.append(V[y == 1].mean(0) - V[y == 0].mean(0))
            r = np.mean(shifts, axis=0)
            r = r / np.linalg.norm(r)
            idx, y = sel[test]
            if (y == 1).sum() == 0 or (y == 0).sum() == 0:
                out_rows.append(dict(patient=test, labelling=lab, pi=None,
                                     cosine=None, n_train_shifts=len(shifts)))
                continue
            V = ts.transform(cache[test]['covs'][idx])
            s_p = V[y == 1].mean(0) - V[y == 0].mean(0)
            dot = float(np.dot(s_p, r))
            out_rows.append(dict(
                patient=test, labelling=lab,
                pi=int(np.sign(dot)) if dot != 0 else 0,
                cosine=float(abs(dot) / np.linalg.norm(s_p)),
                n_train_shifts=len(shifts)))
        log(f"geom: {lab} done ({(time.time()-t0)/60:.1f} min)")
    with open(OUT_DIR / 'geom_raw.json', 'w') as f:
        json.dump(out_rows, f)
    return out_rows


# =============================================================================
# STAGE: stats  (bootstraps, tests, rebuilds, CSVs, figures, SUMMARY)
# =============================================================================

def bootstrap_auc_ci(y, s, n_boot=N_BOOT, seed=BOOT_SEED):
    """95 % percentile bootstrap over windows, stratified by class."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    s = np.asarray(s, float)
    i0 = np.where(y == 0)[0]
    i1 = np.where(y == 1)[0]
    if len(i0) == 0 or len(i1) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        b0 = rng.choice(i0, len(i0), replace=True)
        b1 = rng.choice(i1, len(i1), replace=True)
        yy = np.concatenate([np.zeros(len(b0)), np.ones(len(b1))])
        ss = np.concatenate([s[b0], s[b1]])
        if len(np.unique(ss)) < 2:
            vals[b] = 0.5
        else:
            vals[b] = roc_auc_score(yy, ss)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def perm_p_below(y, s, n_perm=N_PERM, seed=BOOT_SEED):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y)
    obs = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n_perm):
        yp = rng.permutation(y)
        if len(np.unique(yp)) < 2:
            continue
        if roc_auc_score(yp, s) <= obs:
            cnt += 1
    return float((1 + cnt) / (n_perm + 1)), float(obs)


def pratt_rank_biserial(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    from scipy.stats import rankdata
    ranks = rankdata(np.abs(d))
    wp = float(ranks[d > 0].sum())
    wm = float(ranks[d < 0].sum())
    return (wp - wm) / (wp + wm) if (wp + wm) > 0 else 0.0


def wilcoxon_pratt(a, b):
    from scipy.stats import wilcoxon
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    try:
        res = wilcoxon(a, b, zero_method='pratt', alternative='two-sided',
                       method='exact')
        used = 'exact'
    except Exception:
        res = wilcoxon(a, b, zero_method='pratt', alternative='two-sided',
                       method='approx')
        used = 'approx'
    return float(res.pvalue), used


def tost_paired(diffs, margin=TOST_MARGIN):
    from scipy import stats
    diffs = np.asarray(diffs, float)
    n = len(diffs)
    m = diffs.mean()
    se = diffs.std(ddof=1) / np.sqrt(n)
    t_low = (m - (-margin)) / se
    t_up = (m - margin) / se
    p_low = 1 - stats.t.cdf(t_low, n - 1)   # H0: m <= -margin
    p_up = stats.t.cdf(t_up, n - 1)         # H0: m >= +margin
    return float(max(p_low, p_up)), float(m), float(se)


def boot_mean_ci(diffs, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(diffs, float)
    means = [rng.choice(diffs, len(diffs), replace=True).mean()
             for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cal(a):
    return None if a is None else max(a, 1 - a)


def sign_of(a):
    return None if a is None else (1 if a >= 0.5 else -1)


def stage_stats(cache, base_rows, scores_store, matched_rows, geom_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import scipy

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    subjects = sorted(cache.keys())
    tests = {}

    def base_auc(lab, seed, method, patient):
        for r in base_rows:
            if (r['labelling'], r['seed'], r['method'], r['patient']) == \
                    (lab, seed, method, patient):
                return r['auc']
        return None

    # ---- bootstrap CIs + permutation + label-flip (seed 0, both labellings)
    ci = {}       # (lab, method, patient) -> (lo, hi)
    permp = {}    # (lab, patient) -> p
    flip_ok, flip_max_dev = True, 0.0
    from sklearn.metrics import roc_auc_score
    for lab in LABELLINGS:
        store = scores_store[(lab, 0)]
        for p in subjects:
            d = store[p]
            y = np.asarray(d['y'])
            for meth, key in [('tier2a', 't2a'), ('tier2b', 't2b'),
                              ('tier3', 't3')]:
                s = d[key]
                if s is None or len(np.unique(y)) < 2:
                    ci[(lab, meth, p)] = (None, None)
                    continue
                ci[(lab, meth, p)] = bootstrap_auc_ci(y, s)
                a1 = roc_auc_score(y, s)
                a2 = roc_auc_score(1 - y, s)
                dev = abs(a2 - (1 - a1))
                flip_max_dev = max(flip_max_dev, dev)
                if dev > 1e-9:
                    flip_ok = False
            if d['t3'] is not None and len(np.unique(y)) > 1:
                permp[(lab, p)] = perm_p_below(y, d['t3'])[0]

    # ---- D1 ---------------------------------------------------------------
    d1_disagree = []
    n_def = 0
    for p in subjects:
        a_o = base_auc('original', 0, 'tier3', p)
        a_s = base_auc('strict', 0, 'tier3', p)
        if a_o is None or a_s is None:
            d1_disagree.append(p + ' (undefined)')
            continue
        n_def += 1
        if sign_of(a_o) != sign_of(a_s):
            d1_disagree.append(p)
    d1_agree = len(subjects) - len(d1_disagree)
    tests['D1'] = dict(
        declared='Tier 3 polarity sign agrees original vs strict in >= 18/22; '
                 'undefined counts as disagreement',
        n_agree=d1_agree, n_total=len(subjects),
        disagreeing_patients=d1_disagree,
        passed=bool(d1_agree >= 18))

    # ---- S1 ---------------------------------------------------------------
    paired_rows = []
    s1 = None
    for lab in LABELLINGS:
        vals = {}
        for meth in ['tier2a', 'tier2b', 'tier3']:
            vals[meth] = {}
            for p in subjects:
                a = base_auc(lab, 0, meth, p)
                if a is not None:
                    vals[meth][p] = a
        common = sorted(set(vals['tier2a']) & set(vals['tier2b'])
                        & set(vals['tier3']))
        for name, (ma, mb) in {
                't3_vs_t2a': ('tier3', 'tier2a'),
                't3_vs_t2b': ('tier3', 'tier2b'),
                't2a_vs_t2b': ('tier2a', 'tier2b')}.items():
            for kind_, f in [('calibrated', cal), ('raw', lambda x: x)]:
                a = [f(vals[ma][p]) for p in common]
                b = [f(vals[mb][p]) for p in common]
                diffs = np.array(a) - np.array(b)
                lo, hi = boot_mean_ci(diffs)
                tp, m, se = tost_paired(diffs)
                wp, wmethod = wilcoxon_pratt(a, b)
                rb = pratt_rank_biserial(a, b)
                row = dict(comparison=name, labelling=lab, auc_kind=kind_,
                           n=len(common), mean_diff=round(m, 4),
                           ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                           tost_p=round(tp, 5), wilcoxon_p=round(wp, 5),
                           wilcoxon_method=wmethod,
                           rank_biserial=round(rb, 3))
                paired_rows.append(row)
                if (name, lab, kind_) == ('t3_vs_t2a', 'original',
                                          'calibrated'):
                    s1 = dict(row)
                    s1['equivalent'] = bool(-TOST_MARGIN < lo and
                                            hi < TOST_MARGIN)
    tests['S1'] = dict(
        declared='T3-T2A calibrated paired diff, original seed 0: equivalence '
                 'iff 95% bootstrap CI inside (-0.05, +0.05); TOST margin '
                 '0.05; Wilcoxon pratt + rank-biserial for all comparisons',
        **s1, passed=None)   # S1 is a measurement, pass = CI reported;
    tests['S1']['passed'] = bool(s1 is not None)

    # ---- M1 ---------------------------------------------------------------
    def matched_mean(method, clf, d, seed=None, kind_='calibrated'):
        f = cal if kind_ == 'calibrated' else (lambda x: x)
        sel_ = [r for r in matched_rows
                if r['method'] == method and r['classifier'] == clf
                and r['d'] == d and r['auc'] is not None
                and (seed is None or r['seed'] == seed)]
        if seed is not None:
            return float(np.mean([f(r['auc']) for r in sel_])) if sel_ else None
        by_seed = {}
        for r in sel_:
            by_seed.setdefault(r['seed'], []).append(f(r['auc']))
        ms = [np.mean(v) for v in by_seed.values()]
        return (float(np.mean(ms)), float(np.std(ms))) if ms else (None, None)

    curve0 = [matched_mean('tier2b_pca', 'lda', d, seed=0) for d in D_GRID]
    rho = float(scipy.stats.spearmanr(D_GRID, curve0).statistic)
    t3_lda_cal0 = float(np.mean(
        [cal(base_auc('original', 0, 'tier3', p)) for p in subjects
         if base_auc('original', 0, 'tier3', p) is not None]))
    t2b36_lda0 = matched_mean('tier2b_pca', 'lda', 36, seed=0)
    gap36 = abs(t2b36_lda0 - t3_lda_cal0)
    m1a = rho > 0.8
    m1b = gap36 <= 0.05
    tests['M1'] = dict(
        declared='(a) Spearman rho(d, cal AUC) over 6 points for '
                 'PCA(T2B)+LDA > 0.8; (b) |T2B_pca36+LDA - T3_36+LDA| <= '
                 '0.05 calibrated, original seed 0',
        spearman_rho=round(rho, 3), curve_seed0=[round(c, 4) for c in curve0],
        t2b_pca36_lda=round(t2b36_lda0, 4), t3_lda=round(t3_lda_cal0, 4),
        gap_at_36=round(gap36, 4), a_passed=bool(m1a), b_passed=bool(m1b),
        passed=bool(m1a and m1b),
        replacement_sentence=(None if (m1a and m1b) else
                              'the Tier 2B representation performs better '
                              'under the conditions tested'))

    # ---- G1 ---------------------------------------------------------------
    geom_csv_rows = []
    g1 = {}
    for lab in LABELLINGS:
        agree_all, n_all, agree_dec, n_dec = 0, 0, 0, 0
        for g in geom_rows:
            if g['labelling'] != lab:
                continue
            p = g['patient']
            t3a = base_auc(lab, 0, 'tier3', p)
            t2aa = base_auc(lab, 0, 'tier2a', p)
            t3s = sign_of(t3a)
            row = dict(patient=p, labelling=lab, pi=g['pi'],
                       cosine=(None if g['cosine'] is None
                               else round(g['cosine'], 4)),
                       tier3_sign=t3s, tier2a_sign=sign_of(t2aa),
                       hw_sign=(-1 if p in HW_INVERTED_195 else
                                (1 if p in HW_SUBSET else None)),
                       agree_tier3=(None if (g['pi'] is None or t3s is None)
                                    else int(g['pi'] == t3s)))
            geom_csv_rows.append(row)
            if g['pi'] is not None and t3s is not None:
                n_all += 1
                agree_all += int(g['pi'] == t3s)
                lo, hi = ci[(lab, 'tier3', p)]
                if lo is not None and (lo > 0.5 or hi < 0.5):
                    n_dec += 1
                    agree_dec += int(g['pi'] == t3s)
        g1[lab] = dict(agree_all=agree_all, n_all=n_all,
                       agree_decided=agree_dec, n_decided=n_dec,
                       rate_decided=(agree_dec / n_dec if n_dec else None))
    tests['G1'] = dict(
        declared='pi_p agrees with Tier 3 AUC side in >= 75% of patients '
                 'whose seed-0 bootstrap CI excludes 0.5 (original '
                 'labelling); all-22 agreement and cosine reported; both '
                 'labellings',
        **{f'{lab}_{k}': v for lab in LABELLINGS
           for k, v in g1[lab].items()},
        passed=bool(g1['original']['n_decided'] > 0 and
                    g1['original']['rate_decided'] is not None and
                    g1['original']['rate_decided'] >= 0.75))

    # ---- controls ---------------------------------------------------------
    tests['controls'] = dict(
        label_flip_passed=bool(flip_ok),
        label_flip_max_deviation=float(flip_max_dev),
        permutation='per-patient p_below in per_patient_auc.csv (tier3 rows)')

    # ---- hardware / simulation bootstrap from saved scores ----------------
    hw_boot_rows = []
    hw20_path = PROJECT_ROOT / 'results/window_analysis/quantum_20s_hardware.json'
    if hw20_path.exists():
        hw = json.load(open(hw20_path, encoding='utf-8'))['per_subject']
        for p, d in hw.items():
            if 'scores' in d and 'y_true' in d:
                lo, hi = bootstrap_auc_ci(d['y_true'], d['scores'])
                hw_boot_rows.append(dict(
                    patient=p, labelling='as-run', method='agate_hw_20s',
                    classifier='fixed-circuit', d=8, seed='n/a',
                    auc=round(float(d['raw_auc']), 4),
                    ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                    polarity_sign=sign_of(d['raw_auc']), perm_p=None))
    tests['hw_sim_bootstrap'] = dict(
        bootstrapped=['agate_hw_20s (results/window_analysis/'
                      'quantum_20s_hardware.json per-subject scores+y_true)'],
        not_available=[
            'agate_hw_1.95s: ch8_heron_loso_results.json stores per-patient '
            'AUC only; would need per-segment expectation values or raw shot '
            'counts from the ibm_torino jobs',
            'agate_sim_1.95s (Table 1 row): quantum_theta_alpha_results.json '
            'stores per-patient AUC and mean scores only',
            'agate_sim_20s: quantum_20s_simulation.json stores per-patient '
            'AUC only',
            'reviewer_response/predictions.npz holds per-segment scores for '
            'the V3 theta-band sim (q_v3), which is NOT the submitted '
            'theta-alpha row, so it was not used'])

    # ---- Job 5 rebuilds ----------------------------------------------------
    qsc = json.load(open(PROJECT_ROOT /
                         'results/scaling/quantum_channel_scaling.json',
                         encoding='utf-8'))
    ch8 = qsc['ch8']['per_subject']
    t4_rows = []
    for p in HW_SUBSET:
        a = float(ch8[p]['auc'])
        s = sign_of(a)
        t4_rows.append(dict(
            patient=p, qsim_auc=a, qsim_sign=('inverted' if s < 0
                                              else 'standard'),
            submitted_sign=('inverted' if SUBMITTED_TABLE4_QSIM[p] < 0
                            else 'standard'),
            differs=int(s != SUBMITTED_TABLE4_QSIM[p]),
            window_s=1.95,
            source='results/scaling/quantum_channel_scaling.json ch8 '
                   '(overall 0.4595 = simulation_ch8_auc in '
                   'ch8_heron_loso_results.json; 1.95 s, PLV 4-13 Hz, MAX)'))
    sim_cal_scaling = float(np.mean([cal(float(ch8[p]['auc']))
                                     for p in HW_SUBSET]))
    v3 = json.load(open(PROJECT_ROOT / 'analysis_results/quantum_loso_v3/'
                        'quantum_theta_alpha_results.json', encoding='utf-8'))
    sim_cal_v3 = float(np.mean([cal(float(v3['per_subject'][p]['auc']))
                                for p in HW_SUBSET]))
    tests['rebuilds'] = dict(
        table4_qsim_differs=[r['patient'] for r in t4_rows if r['differs']],
        sim195_calibrated_pooled_submitted=0.540,
        sim195_calibrated_per_patient_oracle_v3file=round(sim_cal_v3, 4),
        sim195_calibrated_per_patient_oracle_scalingfile=round(
            sim_cal_scaling, 4),
        seizure_count=142)

    # ---- cohort table -------------------------------------------------------
    cohort_rows = []
    for p in subjects:
        d = cache[p]
        k = d['kind']
        pool = int((k == KIND_INTER).sum())
        n_ict = int((k == KIND_ICTAL).sum())
        n_pre = int((k == KIND_PREICTAL).sum())
        cohort_rows.append(dict(
            patient=p, n_seizures=int(d['n_seizures']),
            total_ictal_s=float(d['total_ictal_s']),
            n_ictal_windows_20s=n_ict, n_preictal_windows_20s=n_pre,
            n_class1_original=n_ict + n_pre, n_class1_strict=n_ict,
            n_interictal_pool=pool,
            n_interictal_used=min(pool, n_ict + n_pre),
            channels_available=int(d['n_ch_montage']),
            channels_in_file=int(d['n_ch_total']),
            in_hw_subset=p in HW_SUBSET))
    bal_o = (sum(r['n_class1_original'] for r in cohort_rows),
             sum(r['n_interictal_used'] for r in cohort_rows))
    bal_s = (sum(r['n_class1_strict'] for r in cohort_rows),
             sum(r['n_interictal_used'] for r in cohort_rows))
    tests['class_balance'] = dict(original=f'{bal_o[0]}:{bal_o[1]}',
                                  strict=f'{bal_s[0]}:{bal_s[1]}')

    # ---- write CSVs ---------------------------------------------------------
    def wcsv(name, rows, header):
        with open(OUT_DIR / name, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in header})

    pp_header = ['patient', 'labelling', 'method', 'classifier', 'd', 'seed',
                 'auc', 'ci_lo', 'ci_hi', 'polarity_sign', 'perm_p']
    pp_rows = []
    for r in base_rows:
        a = r['auc']
        lo, hi = (ci.get((r['labelling'], r['method'], r['patient']),
                         (None, None)) if r['seed'] == 0 else (None, None))
        pp_rows.append(dict(
            **r, ci_lo=(None if lo is None else round(lo, 4)),
            ci_hi=(None if hi is None else round(hi, 4)),
            polarity_sign=sign_of(a),
            perm_p=(round(permp[(r['labelling'], r['patient'])], 4)
                    if r['method'] == 'tier3' and r['seed'] == 0
                    and (r['labelling'], r['patient']) in permp else None)))
        pp_rows[-1]['auc'] = None if a is None else round(a, 4)
    for r in matched_rows:
        pp_rows.append(dict(**r, ci_lo=None, ci_hi=None,
                            polarity_sign=sign_of(r['auc']), perm_p=None))
        pp_rows[-1]['auc'] = (None if r['auc'] is None
                              else round(r['auc'], 4))
    pp_rows.extend(hw_boot_rows)
    wcsv('per_patient_auc.csv', pp_rows, pp_header)

    ov_rows = []
    for lab in LABELLINGS:
        for meth, clf, dnat in [('tier2a', 'xgb', 8), ('tier2b', 'xgb', 336),
                                ('tier3', 'lda', 36)]:
            for kind_, f in [('raw', lambda x: x), ('calibrated', cal)]:
                by_seed = []
                for seed in SEEDS:
                    vs = [f(r['auc']) for r in base_rows
                          if (r['labelling'], r['seed'], r['method']) ==
                          (lab, seed, meth) and r['auc'] is not None]
                    if vs:
                        by_seed.append(np.mean(vs))
                ov_rows.append(dict(
                    labelling=lab, method=meth + '_' + kind_, classifier=clf,
                    d=dnat, mean_over_seeds=round(float(np.mean(by_seed)), 4),
                    sd_over_seeds=round(float(np.std(by_seed)), 4)))
    wcsv('overall_auc.csv', ov_rows,
         ['labelling', 'method', 'classifier', 'd', 'mean_over_seeds',
          'sd_over_seeds'])

    md_rows = []
    for clf in ['lda', 'xgb']:
        for d in D_GRID:
            mean_, sd_ = matched_mean('tier2b_pca', clf, d)
            inv_by_seed = []
            for seed in SEEDS:
                inv = sum(1 for r in matched_rows
                          if (r['method'], r['classifier'], r['d'],
                              r['seed']) == ('tier2b_pca', clf, d, seed)
                          and r['auc'] is not None and r['auc'] < 0.5)
                inv_by_seed.append(inv)
            md_rows.append(dict(labelling='original', classifier=clf, d=d,
                                auc_mean=round(mean_, 4),
                                auc_sd=round(sd_, 4),
                                n_inversions=int(np.round(
                                    np.mean(inv_by_seed)))))
    wcsv('matched_dimension.csv', md_rows,
         ['labelling', 'classifier', 'd', 'auc_mean', 'auc_sd',
          'n_inversions'])

    wcsv('geometric_sign.csv', geom_csv_rows,
         ['patient', 'labelling', 'pi', 'cosine', 'tier3_sign', 'tier2a_sign',
          'hw_sign', 'agree_tier3'])
    wcsv('paired_tests.csv', paired_rows,
         ['comparison', 'labelling', 'auc_kind', 'n', 'mean_diff', 'ci_lo',
          'ci_hi', 'tost_p', 'wilcoxon_p', 'wilcoxon_method',
          'rank_biserial'])
    wcsv('table4_rebuilt.csv', t4_rows,
         ['patient', 'qsim_auc', 'qsim_sign', 'submitted_sign', 'differs',
          'window_s', 'source'])
    wcsv('cohort_table.csv', cohort_rows, list(cohort_rows[0].keys()))

    with open(OUT_DIR / 'tests.json', 'w', encoding='utf-8') as f:
        json.dump(dict(pre_declarations_see='scripts/p2rev02_classical_rerun'
                                            '.py module docstring',
                       xgb_params=XGB_PARAMS, lda_params=LDA_PARAMS,
                       scipy_version=scipy.__version__, **tests), f, indent=2)

    # ---- figures ------------------------------------------------------------
    make_figures(base_rows, matched_rows, geom_csv_rows, ci, plt)

    # ---- SUMMARY ------------------------------------------------------------
    write_summary(tests, ov_rows, md_rows, paired_rows, geom_csv_rows,
                  cohort_rows, t4_rows, base_rows, ci)
    log("stats stage done")
    return tests


def make_figures(base_rows, matched_rows, geom_rows, ci, plt):
    subjects = sorted({r['patient'] for r in base_rows})

    # 1. matched dimension curve (LDA vs XGB), calibrated, seed-mean
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for clf in ['lda', 'xgb']:
        ys, sds = [], []
        for d in D_GRID:
            vals = []
            for seed in SEEDS:
                v = [cal(r['auc']) for r in matched_rows
                     if (r['method'], r['classifier'], r['d'], r['seed']) ==
                     ('tier2b_pca', clf, d, seed) and r['auc'] is not None]
                if v:
                    vals.append(np.mean(v))
            ys.append(np.mean(vals))
            sds.append(np.std(vals))
        ax.errorbar(D_GRID, ys, yerr=sds, marker='o', ms=5, lw=2,
                    color=PALETTE[clf], label=f'PCA(Tier 2B) + {clf.upper()}')
        ax.annotate(f'{ys[-1]:.3f}', (D_GRID[-1], ys[-1]),
                    textcoords='offset points', xytext=(6, 4), fontsize=9,
                    color=PALETTE[clf])
    ax.set_xscale('log')
    ax.set_xticks(D_GRID)
    ax.set_xticklabels(D_GRID)
    ax.set_xlabel('PCA dimension d (log scale)')
    ax.set_ylabel('mean per-patient calibrated AUC')
    ax.set_title('Matched dimension, original labelling\n'
                 '(mean ± SD over 10 interictal seeds)')
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'matched_dimension_curve.png', dpi=150)
    plt.close(fig)

    # 2. geometric sign vs Tier 3 AUC (original labelling)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    xs, ys_, names = [], [], []
    t3 = {(r['patient']): r['auc'] for r in base_rows
          if (r['labelling'], r['seed'], r['method']) ==
          ('original', 0, 'tier3')}
    for g in geom_rows:
        if g['labelling'] != 'original' or g['pi'] is None:
            continue
        p = g['patient']
        if t3.get(p) is None:
            continue
        xs.append(g['pi'] * g['cosine'])
        ys_.append(t3[p])
        names.append(p)
    ax.axhline(0.5, color=PALETTE['neutral'], lw=1, ls='--')
    ax.axvline(0.0, color=PALETTE['neutral'], lw=1, ls='--')
    ax.scatter(xs, ys_, s=55, color=PALETTE['tier3'], zorder=3,
               edgecolor='white', linewidth=1.5)
    for x, y, p in zip(xs, ys_, names):
        if (x > 0) != (y > 0.5):   # label the disagreements
            ax.annotate(p, (x, y), textcoords='offset points', xytext=(5, 4),
                        fontsize=8, color=PALETTE['neutral'])
    ax.set_xlabel('signed geometric projection  $\\pi_p \\cdot '
                  '|\\langle \\bar{s}_p, r\\rangle| / \\|\\bar{s}_p\\|$')
    ax.set_ylabel('Tier 3 raw AUC (seed 0)')
    ax.set_title('Geometric polarity sign vs Tier 3 AUC side '
                 '(22 patients, original labelling)')
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'geometric_sign_vs_auc.png', dpi=150)
    plt.close(fig)

    # 3. per-patient CI (original, seed 0), three tiers
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    offs = {'tier2a': -0.25, 'tier2b': 0.0, 'tier3': 0.25}
    base0 = {(r['method'], r['patient']): r['auc'] for r in base_rows
             if (r['labelling'], r['seed']) == ('original', 0)}
    for meth in ['tier2a', 'tier2b', 'tier3']:
        xs_, ys2, lo_, hi_ = [], [], [], []
        for i, p in enumerate(subjects):
            a = base0.get((meth, p))
            c = ci.get(('original', meth, p), (None, None))
            if a is None or c[0] is None:
                continue
            xs_.append(i + offs[meth])
            ys2.append(a)
            lo_.append(a - c[0])
            hi_.append(c[1] - a)
        ax.errorbar(xs_, ys2, yerr=[lo_, hi_], fmt='o', ms=4, lw=1.3,
                    capsize=2, color=PALETTE[meth],
                    label={'tier2a': 'Tier 2A (8, XGB)',
                           'tier2b': 'Tier 2B (336, XGB)',
                           'tier3': 'Tier 3 (36, LDA)'}[meth])
    ax.axhline(0.5, color=PALETTE['neutral'], lw=1, ls='--')
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=90, fontsize=8)
    ax.set_ylabel('raw AUC with 95 % bootstrap CI')
    ax.set_title('Per-patient raw AUC, original labelling, seed 0 '
                 '(2000-resample stratified bootstrap)')
    ax.grid(alpha=0.25, lw=0.5, axis='y')
    ax.legend(frameon=False, ncol=3, loc='lower right')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'per_patient_ci.png', dpi=150)
    plt.close(fig)


def write_summary(tests, ov_rows, md_rows, paired_rows, geom_rows,
                  cohort_rows, t4_rows, base_rows, ci):
    def ov(lab, meth):
        for r in ov_rows:
            if r['labelling'] == lab and r['method'] == meth:
                return r
        return {}

    L = []
    L.append("# PROMPT P2REV-02 - Classical re-run for the Paper 2 revision\n")
    L.append(f"**Generated:** {datetime.now():%Y-%m-%d}")
    L.append("**Script:** `scripts/p2rev02_classical_rerun.py`")
    L.append("**Depends on:** P2REV-01 inventory; feature code "
             "`sagemaker/train_full_cohort.py`; window builder after "
             "`scripts/prompt037_per_seizure_decomposition.py`\n")
    L.append("Conventions: CHB-MIT, 22 patients (chb12, chb24 excluded), "
             "fronto-central CH8 montage (FP1-F7, F7-T7, FP1-F3, F3-C3, "
             "FP2-F8, F8-T8, FP2-F4, F4-C4 - the montage the submitted runs "
             "actually used), $20\\,$s windows, LOSO. Two labellings: "
             "*original* (class 1 = annotated seizure + 60 s preceding, as "
             "submitted) and *strict* (class 1 = annotated seizure only; "
             "preictal windows dropped). Interictal windows from the "
             "seizure-free files at 40 s spacing, per-patient count matched "
             "to the class-1 count, resampled under 10 seeds. Calibrated "
             "means per-patient oracle $\\max(\\mathrm{AUC}, "
             "1-\\mathrm{AUC})$ throughout, named as such. XGBoost "
             "hyperparameters are the submitted run's constants "
             "(n_estimators=200, max_depth=6, learning_rate=0.1, "
             "subsample=0.8, colsample_bytree=0.8, min_child_weight=3, "
             "reg_alpha=0.1, reg_lambda=1.0); nothing is tuned. "
             "Pre-declared: **D1, S1, M1, G1** (verbatim in the script "
             "docstring).\n")
    L.append("---\n")
    L.append("## Headline AUC table (mean over seeds $\\pm$ SD; mean "
             "per-patient AUC)\n")
    L.append("| labelling | method | raw | calibrated |")
    L.append("|---|---|---|---|")
    for lab in LABELLINGS:
        for meth, label in [('tier2a', 'Tier 2A (8, XGB)'),
                            ('tier2b', 'Tier 2B (336, XGB)'),
                            ('tier3', 'Tier 3 (36, LDA)')]:
            r_ = ov(lab, meth + '_raw')
            c_ = ov(lab, meth + '_calibrated')
            L.append(f"| {lab} | {label} | "
                     f"{r_.get('mean_over_seeds')} ± {r_.get('sd_over_seeds')} | "
                     f"{c_.get('mean_over_seeds')} ± {c_.get('sd_over_seeds')} |")
    L.append("")
    L.append(f"Submitted Table 1 for comparison: Tier 2B 0.952/0.952, "
             f"Tier 3 0.536/0.659, Tier 2A 0.521/0.628 (one seed, whole "
             f"segments rather than tiled windows, 10 interictal/patient).\n")

    L.append("## D1 - polarity sign, original vs strict\n")
    d1 = tests['D1']
    L.append(f"Agreement {d1['n_agree']}/{d1['n_total']}; disagreeing: "
             f"{', '.join(d1['disagreeing_patients']) or 'none'}.\n")
    L.append(f"**Verdict: D1 {'PASS' if d1['passed'] else 'FAIL'}** "
             f"(threshold 18/22).\n")

    L.append("## S1 - equivalence Tier 3 vs Tier 2A\n")
    s1 = tests['S1']
    L.append(f"Mean paired difference (calibrated, original, seed 0) "
             f"$\\Delta = {s1['mean_diff']}$, 95 % bootstrap CI "
             f"[{s1['ci_lo']}, {s1['ci_hi']}], TOST $p = {s1['tost_p']}$ "
             f"(margin $\\pm 0.05$), Wilcoxon ({s1['wilcoxon_method']}, "
             f"pratt) $p = {s1['wilcoxon_p']}$, rank-biserial "
             f"$r = {s1['rank_biserial']}$.\n")
    L.append(f"**Verdict: S1 - equivalence "
             f"{'CLAIMED' if s1.get('equivalent') else 'NOT claimed'}** "
             f"(CI inside the margin: {s1.get('equivalent')}). Full table "
             f"in `paired_tests.csv`.\n")

    L.append("## M1 - matched dimension\n")
    m1 = tests['M1']
    L.append("| d | " + " | ".join(str(d) for d in D_GRID) + " |")
    L.append("|---|" + "---|" * len(D_GRID))
    L.append("| PCA(T2B)+LDA cal AUC (seed 0) | " +
             " | ".join(f"{c:.3f}" for c in m1['curve_seed0']) + " |")
    L.append("")
    L.append(f"Spearman $\\rho = {m1['spearman_rho']}$ (a: "
             f"{'PASS' if m1['a_passed'] else 'FAIL'}); at $d=36$: "
             f"PCA(T2B)+LDA {m1['t2b_pca36_lda']} vs Tier 3+LDA "
             f"{m1['t3_lda']}, gap {m1['gap_at_36']} (b: "
             f"{'PASS' if m1['b_passed'] else 'FAIL'}).\n")
    L.append(f"**Verdict: M1 {'PASS' if m1['passed'] else 'FAIL'}.**" +
             ("" if m1['passed'] else f" The manuscript's dimensionality "
              f"claim is withdrawn and replaced by: "
              f"\"{m1['replacement_sentence']}\"") + "\n")

    L.append("## G1 - geometric sign as predictor\n")
    g1k = tests['G1']
    L.append(f"Original labelling: decided patients (CI excludes 0.5) "
             f"{g1k['original_n_decided']}, agreement "
             f"{g1k['original_agree_decided']}/{g1k['original_n_decided']}; "
             f"all 22: {g1k['original_agree_all']}/{g1k['original_n_all']}. "
             f"Strict: {g1k['strict_agree_decided']}/"
             f"{g1k['strict_n_decided']} decided, "
             f"{g1k['strict_agree_all']}/{g1k['strict_n_all']} overall. "
             f"Per-patient $\\pi_p$ and cosines in `geometric_sign.csv`.\n")
    L.append(f"**Verdict: G1 {'PASS' if g1k['passed'] else 'FAIL'}** "
             f"(threshold 75 % among decided, original labelling).\n")

    L.append("## Controls\n")
    c = tests['controls']
    L.append(f"Label-flip: every patient x method maps AUC to exactly "
             f"$1-\\mathrm{{AUC}}$ (max deviation "
             f"{c['label_flip_max_deviation']:.2e}) - "
             f"{'PASS' if c['label_flip_passed'] else 'FAIL'}. Per-patient "
             f"permutation p-values (1000 within-patient label permutations, "
             f"Tier 3) are the `perm_p` column of `per_patient_auc.csv`.\n")

    L.append("## Rebuilds (Job 5)\n")
    rb = tests['rebuilds']
    L.append(f"- Table 4 QSim column rebuilt from the 1.95 s statevector "
             f"run (`quantum_channel_scaling.json` ch8): signs differ from "
             f"the submitted column for "
             f"{', '.join(rb['table4_qsim_differs']) or 'no patient'} "
             f"(`table4_rebuilt.csv`).")
    L.append(f"- Sim-1.95 s calibrated: submitted pooled flip "
             f"{rb['sim195_calibrated_pooled_submitted']}; per-patient "
             f"oracle {rb['sim195_calibrated_per_patient_oracle_v3file']} "
             f"(theta-alpha file) / "
             f"{rb['sim195_calibrated_per_patient_oracle_scalingfile']} "
             f"(scaling file).")
    L.append(f"- Cohort table: `cohort_table.csv` (per-patient windows per "
             f"class, pool sizes, montage channels). Class balance "
             f"(class1:interictal): original "
             f"{tests['class_balance']['original']}, strict "
             f"{tests['class_balance']['strict']}.")
    L.append(f"- Seizure count: {rb['seizure_count']} (per-annotation; "
             f"the submitted \"163\" remains unsourced - see results doc).\n")

    L.append("![matched dimension](figures/matched_dimension_curve.png)")
    L.append("![geometric sign](figures/geometric_sign_vs_auc.png)")
    L.append("![per-patient CI](figures/per_patient_ci.png)\n")

    L.append("## What changes in the manuscript\n")
    L.append("*(placeholder - filled in by the analysis of this run; see "
             "docs/PROMPT_P2REV_02_RESULTS.md)*\n")

    with open(OUT_DIR / 'SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(L))


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['cache', 'loso', 'matched', 'geom', 'stats',
                             'all'])
    ap.add_argument('--data-root', default=str(DATA_ROOT))
    args = ap.parse_args()
    t0 = time.time()
    data_root = Path(args.data_root)
    subjects = sorted(d for d in data_root.iterdir()
                      if d.is_dir() and d.name.startswith('chb')
                      and d.name not in EXCLUDE_SUBJECTS)
    log(f"P2REV-02 stage={args.stage}  subjects={len(subjects)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.stage in ('cache', 'all'):
        stage_cache(subjects)
    cache = load_cache(subjects)
    log(f"cache loaded: {len(cache)} subjects")

    base_rows = scores_store = matched_rows = geom_rows = None
    if args.stage in ('loso', 'all'):
        base_rows, scores_store = stage_loso(cache)
    if args.stage in ('matched', 'all'):
        matched_rows = stage_matched(cache)
    if args.stage in ('geom', 'all'):
        geom_rows = stage_geom(cache)
    if args.stage in ('stats', 'all'):
        if base_rows is None:
            base_rows = json.load(open(OUT_DIR / 'loso_base_auc.json'))
            scores_store = np.load(OUT_DIR / 'loso_scores.npy',
                                   allow_pickle=True)[0]
        if matched_rows is None:
            matched_rows = json.load(open(OUT_DIR / 'matched_auc.json'))
        if geom_rows is None:
            geom_rows = json.load(open(OUT_DIR / 'geom_raw.json'))
        stage_stats(cache, base_rows, scores_store, matched_rows, geom_rows)

    log(f"TOTAL runtime {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
