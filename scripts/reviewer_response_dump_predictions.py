#!/usr/bin/env python3
"""
REV-001 Task 0/3 helper: re-run all Paper 1 LOSO pipelines using the cached
quantum_8ch EEG (and feature) caches and dump per-segment scores so we can
compute DeLong / permutation / bootstrap stats with paired arrays.

Outputs to results/reviewer_response/predictions.npz:
    seg_subject : (N,) string
    seg_label   : (N,) int in {0,1}
    seg_index   : (N,) int (cache index)
    q_v1        : (N,) float quantum V1 PN-dynamics fidelity (not yet
                  ictal-template-vs-interictal differenced; this is the
                  single-fidelity score the V1 script uses for AUC)
    q_v2        : (N,) float quantum V2 dual-template (fid_ictal - fid_inter)
    q_v3        : (N,) float quantum V3 PLV theta band, MAX subwindow agg
    cls_8ch     : (N,) float bag-mean proba (5 bags) for classical 8ch V2
    cls_8ch_bagN: (N,) float per-bag proba for bag N (N=0..4)
    cls_18ch   : (N18,) float for the 18ch pipeline (smaller cache, 486)
    cls_18ch_seg_subject, cls_18ch_seg_label, cls_18ch_seg_index, etc.

Per-fold metadata is implicit in the LOSO subject grouping.

Re-uses the exact LOSO/template code from the Paper 1 scripts.
"""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'QA1'))

from sagemaker.train_chbmit import FS

# Import re-usable pieces from existing scripts
from scripts.run_quantum_loso import (
    PN_LAMBDA_A, PN_LAMBDA_C, PN_DT,
    normalize_eeg, eeg_to_statevector, train_template,
)
from scripts.run_quantum_loso_v2 import (
    extract_pn_params_bandpower, build_template as build_dual_template_v2,
    compute_fidelity as compute_fid_v2,
    extract_all_features_v2,
)
from scripts.run_quantum_loso_v3 import (
    extract_plv_params_subwindows, build_plv_template,
    compute_fidelity as compute_fid_v3,
    BANDS,
)
from QA1.pn_dynamics import PNDynamics
from QA1.multichannel_circuit import create_multichannel_circuit, get_statevector, compute_fidelity as qa1_compute_fidelity

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                    stream=sys.stdout, force=True)
log = logging.getLogger('rev001-dump')

# Numba-accelerated PN dynamics (clamp mode) to make V1 LOSO tractable
try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False


def _evolve_clamp_py(eeg_abs, lambda_a, lambda_c, dt):
    """Reference clamp evolution. eeg_abs: (n_ch, T) abs of normalized eeg."""
    n_ch, T = eeg_abs.shape
    out = np.zeros((n_ch, 3), dtype=np.float64)
    TWOPI = 2.0 * np.pi
    for ch in range(n_ch):
        a = 0.0; b = 0.0; c = 0.0
        for t in range(T):
            ft = eeg_abs[ch, t]
            a = a + dt * (-lambda_a * a + ft * (1.0 - a))
            if a < 0.0: a = 0.0
            elif a > 1.0: a = 1.0
            b = b + dt * (ft * (1.0 - b))
            if b < 0.0: b = 0.0
            elif b > TWOPI: b = TWOPI
            c = c + dt * (lambda_c * c + ft * (1.0 - c))
            if c < 0.0: c = 0.0
            elif c > 1.0: c = 1.0
        out[ch, 0] = a; out[ch, 1] = b; out[ch, 2] = c
    return out

if HAS_NUMBA:
    _evolve_clamp_fast = njit(cache=True)(_evolve_clamp_py)
else:
    _evolve_clamp_fast = _evolve_clamp_py


def fast_evolve_multichannel(eeg_norm, lambda_a, lambda_c, dt):
    """Drop-in replacement for PNDynamics.evolve_multichannel (clamp mode)."""
    abs_eeg = np.ascontiguousarray(np.abs(eeg_norm, dtype=np.float64))
    arr = _evolve_clamp_fast(abs_eeg, float(lambda_a), float(lambda_c), float(dt))
    return [(float(arr[i, 0]), float(arr[i, 1]), float(arr[i, 2])) for i in range(arr.shape[0])]

CACHE_EEG = ROOT / 'analysis_results' / 'quantum_8ch_cache.npz'
CACHE_FEAT = ROOT / 'analysis_results' / 'feature_cache.npz'
OUT_DIR = ROOT / 'results' / 'reviewer_response'

LABEL_MAP = {'interictal': 0, 'preictal': 1, 'ictal': 1}

# V3 best band per published summary
V3_BAND_NAME = 'theta'
V3_BAND = BANDS[V3_BAND_NAME]


def load_eeg_cache():
    log.info(f'Loading EEG cache: {CACHE_EEG}')
    d = np.load(CACHE_EEG, allow_pickle=True)
    eeg = list(d['eeg'])
    labels = list(d['labels'])
    subjects = list(d['subjects'])
    log.info(f'  {len(eeg)} segments, {len(set(subjects))} subjects')
    return eeg, labels, subjects


def _eeg_to_sv_fast(eeg, lambda_a, lambda_c, dt):
    """Numba-accelerated V1 eeg -> statevector. Matches run_quantum_loso.eeg_to_statevector."""
    eeg_norm = normalize_eeg(eeg)
    params = fast_evolve_multichannel(eeg_norm, lambda_a, lambda_c, dt)
    circuit = create_multichannel_circuit(params)
    return get_statevector(circuit), params


def _build_template_fast(preictal_eeg, lambda_a, lambda_c, dt):
    """Build V1 template circuit using fast PN evolution; matches train_template()."""
    all_params = []
    for eeg in preictal_eeg:
        eeg_norm = normalize_eeg(eeg)
        params = fast_evolve_multichannel(eeg_norm, lambda_a, lambda_c, dt)
        all_params.append(params)
    n_channels = len(all_params[0])
    avg_params = []
    for ch in range(n_channels):
        a_avg = np.mean([p[ch][0] for p in all_params])
        b_avg = np.mean([p[ch][1] for p in all_params])
        c_avg = np.mean([p[ch][2] for p in all_params])
        avg_params.append((a_avg, b_avg, c_avg))
    return get_statevector(create_multichannel_circuit(avg_params))


def dump_quantum_v1(eeg_list, labels, subjects):
    """Reproduce V1 LOSO and dump per-segment fidelity score (the AUC-used scalar).

    Uses a numba-accelerated PN dynamics that is mathematically identical to
    QA1.pn_dynamics.PNDynamics (clamp mode). The published V1 AUC = 0.444
    must reproduce within tolerance for this substitution to be valid.
    """
    from sklearn.metrics import roc_auc_score
    print(f'[V1] using PN dynamics: lambda_a={PN_LAMBDA_A}, lambda_c={PN_LAMBDA_C}, dt={PN_DT}, '
          f'numba={"yes" if HAS_NUMBA else "no"}', flush=True)
    y = np.array([LABEL_MAP.get(l, 0) for l in labels])
    unique_subjects = sorted(set(subjects))
    scores = np.full(len(eeg_list), np.nan)
    for si, ts in enumerate(unique_subjects):
        train_mask = np.array([s != ts for s in subjects])
        test_mask = np.array([s == ts for s in subjects])
        train_eeg = [eeg_list[i] for i in range(len(eeg_list)) if train_mask[i]]
        train_labels = [labels[i] for i in range(len(labels)) if train_mask[i]]
        y_test = y[test_mask]
        if len(np.unique(y_test)) < 2:
            print(f'[V1] {ts}: skipped (single class)', flush=True)
            continue
        preictal_eeg = [eeg for eeg, lbl in zip(train_eeg, train_labels) if lbl == 'preictal']
        if not preictal_eeg:
            preictal_eeg = [eeg for eeg, lbl in zip(train_eeg, train_labels) if lbl == 'ictal']
        if not preictal_eeg:
            print(f'[V1] {ts}: skipped (no positive)', flush=True)
            continue
        try:
            template_sv = _build_template_fast(preictal_eeg, PN_LAMBDA_A, PN_LAMBDA_C, PN_DT)
        except Exception as e:
            print(f'[V1] {ts}: template failed: {e}', flush=True)
            continue
        test_indices = np.where(test_mask)[0]
        for gi in test_indices:
            try:
                test_sv, _ = _eeg_to_sv_fast(eeg_list[gi], PN_LAMBDA_A, PN_LAMBDA_C, PN_DT)
                scores[gi] = qa1_compute_fidelity(template_sv, test_sv)
            except Exception:
                scores[gi] = 0.5
        print(f'[V1] {ts} done ({si+1}/{len(unique_subjects)})', flush=True)
    used_mask = ~np.isnan(scores)
    auc_overall = roc_auc_score(y[used_mask], scores[used_mask])
    print(f'[V1] reproduced AUC = {auc_overall:.4f} (n={used_mask.sum()})', flush=True)
    return scores, used_mask, auc_overall


def dump_quantum_v2(eeg_list, labels, subjects):
    """Reproduce V2 dual-template LOSO; dump (fid_ictal - fid_inter)."""
    from sklearn.metrics import roc_auc_score
    print(f'[V2] dual-template band-power PN encoding', flush=True)
    y = np.array([LABEL_MAP.get(l, 0) for l in labels])
    unique_subjects = sorted(set(subjects))
    scores = np.full(len(eeg_list), np.nan)
    for si, ts in enumerate(unique_subjects):
        train_mask = np.array([s != ts for s in subjects])
        test_mask = np.array([s == ts for s in subjects])
        train_eeg = [eeg_list[i] for i in range(len(eeg_list)) if train_mask[i]]
        train_labels = [labels[i] for i in range(len(labels)) if train_mask[i]]
        y_test = y[test_mask]
        if len(np.unique(y_test)) < 2:
            continue
        ictal_segs = [eeg for eeg, lbl in zip(train_eeg, train_labels) if lbl in ('ictal', 'preictal')]
        inter_segs = [eeg for eeg, lbl in zip(train_eeg, train_labels) if lbl == 'interictal']
        if not ictal_segs or not inter_segs:
            continue
        _, ictal_sv, _ = build_dual_template_v2(ictal_segs, fs=FS)
        _, inter_sv, _ = build_dual_template_v2(inter_segs, fs=FS)
        test_indices = np.where(test_mask)[0]
        for gi in test_indices:
            params = extract_pn_params_bandpower(eeg_list[gi], fs=FS)
            test_sv = get_statevector(create_multichannel_circuit(params))
            fid_i = compute_fid_v2(ictal_sv, test_sv)
            fid_n = compute_fid_v2(inter_sv, test_sv)
            scores[gi] = float(fid_i - fid_n)
        print(f'[V2] {ts} done ({si+1}/{len(unique_subjects)})', flush=True)
    used_mask = ~np.isnan(scores)
    auc_overall = roc_auc_score(y[used_mask], scores[used_mask])
    print(f'[V2] reproduced AUC = {auc_overall:.4f} (n={used_mask.sum()})', flush=True)
    return scores, used_mask, auc_overall


def dump_quantum_v3(eeg_list, labels, subjects, band=V3_BAND):
    """Reproduce V3 PLV theta band LOSO; dump MAX-aggregated score."""
    from sklearn.metrics import roc_auc_score
    print(f'[V3] PLV band {band[0]}-{band[1]} Hz, MAX subwindow aggregation', flush=True)
    y = np.array([LABEL_MAP.get(l, 0) for l in labels])
    unique_subjects = sorted(set(subjects))
    scores = np.full(len(eeg_list), np.nan)
    for si, ts in enumerate(unique_subjects):
        train_mask = np.array([s != ts for s in subjects])
        test_mask = np.array([s == ts for s in subjects])
        train_eeg = [eeg_list[i] for i in range(len(eeg_list)) if train_mask[i]]
        train_labels = [labels[i] for i in range(len(labels)) if train_mask[i]]
        y_test = y[test_mask]
        if len(np.unique(y_test)) < 2:
            continue
        ictal_segs = [eeg for eeg, lbl in zip(train_eeg, train_labels) if lbl in ('ictal', 'preictal')]
        inter_segs = [eeg for eeg, lbl in zip(train_eeg, train_labels) if lbl == 'interictal']
        if not ictal_segs or not inter_segs:
            continue
        _, ictal_sv = build_plv_template(ictal_segs, fs=FS, band=band)
        _, inter_sv = build_plv_template(inter_segs, fs=FS, band=band)
        if ictal_sv is None or inter_sv is None:
            continue
        test_indices = np.where(test_mask)[0]
        for gi in test_indices:
            sw_params_list = extract_plv_params_subwindows(eeg_list[gi], fs=FS, band=band)
            sub_scores = []
            for sw_params in sw_params_list:
                test_sv = get_statevector(create_multichannel_circuit(sw_params))
                fid_i = compute_fid_v3(ictal_sv, test_sv)
                fid_n = compute_fid_v3(inter_sv, test_sv)
                sub_scores.append(fid_i - fid_n)
            scores[gi] = float(max(sub_scores)) if sub_scores else 0.0
        print(f'[V3] {ts} done ({si+1}/{len(unique_subjects)})', flush=True)
    used_mask = ~np.isnan(scores)
    auc_overall = roc_auc_score(y[used_mask], scores[used_mask])
    print(f'[V3] reproduced AUC = {auc_overall:.4f} (n={used_mask.sum()})', flush=True)
    return scores, used_mask, auc_overall


def dump_classical_8ch(eeg_list, labels, subjects, n_bags=5):
    """Reproduce the V2 classical 8-ch XGBoost LOSO; return per-bag probas + mean."""
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    print('[cls8] extracting V2 features (8 channels)', flush=True)
    feats = np.array([extract_all_features_v2(eeg, FS) for eeg in eeg_list])
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    print(f'[cls8] features shape: {feats.shape}', flush=True)
    y = np.array([LABEL_MAP.get(l, 0) for l in labels])
    unique_subjects = sorted(set(subjects))

    per_bag = np.full((n_bags, len(eeg_list)), np.nan)
    for si, ts in enumerate(unique_subjects):
        train_mask = np.array([s != ts for s in subjects])
        test_mask = np.array([s == ts for s in subjects])
        X_train = feats[train_mask]
        y_train = y[train_mask]
        X_test = feats[test_mask]
        y_test = y[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f'[cls8] {ts}: skipped (single class)', flush=True)
            continue
        for bag in range(n_bags):
            rng = np.random.RandomState(42 + bag)
            idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            model = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42 + bag,
                eval_metric='logloss', n_jobs=-1,
            )
            model.fit(X_train[idx], y_train[idx])
            proba = model.predict_proba(X_test)[:, 1]
            test_idx = np.where(test_mask)[0]
            per_bag[bag, test_idx] = proba
        print(f'[cls8] {ts} done ({si+1}/{len(unique_subjects)})', flush=True)
    mean_proba = np.nanmean(per_bag, axis=0)
    used = ~np.isnan(mean_proba)
    auc = roc_auc_score(y[used], mean_proba[used])
    print(f'[cls8] reproduced AUC = {auc:.4f} (n={used.sum()})', flush=True)
    per_bag_aucs = []
    for bag in range(n_bags):
        m = ~np.isnan(per_bag[bag])
        per_bag_aucs.append(roc_auc_score(y[m], per_bag[bag, m]))
    print(f'[cls8] per-bag AUCs: {per_bag_aucs} SD={np.std(per_bag_aucs, ddof=1):.4f}', flush=True)
    return mean_proba, per_bag, used, auc, per_bag_aucs


def dump_classical_18ch(n_bags=5):
    """Reproduce 18-ch LOSO using feature_cache; dump bag-mean proba."""
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    print(f'[cls18] loading {CACHE_FEAT}', flush=True)
    d = np.load(CACHE_FEAT, allow_pickle=True)
    feats = d['features']
    labels = d['labels']
    subjects = d['subjects']
    feats = np.nan_to_num(feats, nan=0.0, posinf=10.0, neginf=-10.0)
    y = np.array([LABEL_MAP.get(l, 0) for l in labels])
    unique_subjects = sorted(set(subjects))
    print(f'[cls18] {feats.shape}, {len(unique_subjects)} subjects', flush=True)
    per_bag = np.full((n_bags, len(feats)), np.nan)
    for si, ts in enumerate(unique_subjects):
        train_mask = np.array([s != ts for s in subjects])
        test_mask = np.array([s == ts for s in subjects])
        X_train = feats[train_mask]
        y_train = y[train_mask]
        X_test = feats[test_mask]
        y_test = y[test_mask]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        for bag in range(n_bags):
            rng = np.random.RandomState(42 + bag)
            idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            model = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42 + bag,
                eval_metric='logloss', n_jobs=-1,
            )
            model.fit(X_train[idx], y_train[idx])
            proba = model.predict_proba(X_test)[:, 1]
            per_bag[bag, np.where(test_mask)[0]] = proba
        print(f'[cls18] {ts} done ({si+1}/{len(unique_subjects)})', flush=True)
    mean_proba = np.nanmean(per_bag, axis=0)
    used = ~np.isnan(mean_proba)
    auc = roc_auc_score(y[used], mean_proba[used])
    print(f'[cls18] reproduced AUC = {auc:.4f} (n={used.sum()})', flush=True)
    per_bag_aucs = []
    for bag in range(n_bags):
        m = ~np.isnan(per_bag[bag])
        per_bag_aucs.append(roc_auc_score(y[m], per_bag[bag, m]))
    print(f'[cls18] per-bag AUCs: {per_bag_aucs} SD={np.std(per_bag_aucs, ddof=1):.4f}', flush=True)
    return mean_proba, per_bag, used, auc, per_bag_aucs, labels, subjects


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eeg, labels, subjects = load_eeg_cache()
    y = np.array([LABEL_MAP.get(l, 0) for l in labels])
    subj_arr = np.array(subjects)

    # Run each dump
    v1_scores, v1_used, v1_auc = dump_quantum_v1(eeg, labels, subjects)
    v2_scores, v2_used, v2_auc = dump_quantum_v2(eeg, labels, subjects)
    v3_scores, v3_used, v3_auc = dump_quantum_v3(eeg, labels, subjects, band=V3_BAND)
    cls8_mean, cls8_bags, cls8_used, cls8_auc, cls8_bag_aucs = dump_classical_8ch(eeg, labels, subjects)

    cls18_mean, cls18_bags, cls18_used, cls18_auc, cls18_bag_aucs, lab18, subj18 = dump_classical_18ch()

    # Per-subject AUC for 8ch (to reproduce Table C1)
    from sklearn.metrics import roc_auc_score
    per_subj_cls8 = {}
    for ts in sorted(set(subjects)):
        m = (subj_arr == ts) & cls8_used
        if m.sum() < 2 or len(np.unique(y[m])) < 2:
            continue
        per_subj_cls8[ts] = float(roc_auc_score(y[m], cls8_mean[m]))

    out = OUT_DIR / 'predictions.npz'
    np.savez_compressed(
        out,
        # 504-segment cache
        seg_subject=subj_arr,
        seg_label=y,
        q_v1=v1_scores,
        q_v1_used=v1_used,
        q_v2=v2_scores,
        q_v2_used=v2_used,
        q_v3=v3_scores,
        q_v3_used=v3_used,
        cls_8ch=cls8_mean,
        cls_8ch_used=cls8_used,
        cls_8ch_bags=cls8_bags,
        cls_8ch_bag_aucs=np.array(cls8_bag_aucs),
        # 486-segment 18ch cache
        seg_18_subject=np.array(subj18),
        seg_18_label=np.array([LABEL_MAP.get(l, 0) for l in lab18]),
        cls_18ch=cls18_mean,
        cls_18ch_used=cls18_used,
        cls_18ch_bags=cls18_bags,
        cls_18ch_bag_aucs=np.array(cls18_bag_aucs),
    )
    log.info(f'Wrote {out}')

    summary = {
        'reproduction_check': {
            'V1':     {'published': 0.444, 'reproduced': float(v1_auc), 'delta': float(v1_auc - 0.444)},
            'V2':     {'published': 0.534, 'reproduced': float(v2_auc), 'delta': float(v2_auc - 0.534)},
            'V3':     {'published': 0.529, 'reproduced': float(v3_auc), 'delta': float(v3_auc - 0.529)},
            'cls_8ch':{'published': 0.625, 'reproduced': float(cls8_auc), 'delta': float(cls8_auc - 0.625)},
            'cls_18ch':{'published': 0.820, 'reproduced': float(cls18_auc), 'delta': float(cls18_auc - 0.820)},
        },
        'cls_8ch_per_bag_auc': [float(x) for x in cls8_bag_aucs],
        'cls_8ch_bag_sd': float(np.std(cls8_bag_aucs, ddof=1)),
        'per_subject_cls8_auc': per_subj_cls8,
        'cls_8ch_per_subject_min': min(per_subj_cls8.values()),
        'cls_8ch_per_subject_max': max(per_subj_cls8.values()),
        'cls_8ch_per_subject_n_subjects': len(per_subj_cls8),
        'alignment_check': {
            'v2_quantum_segments_used': int(v2_used.sum()),
            'cls8_segments_used': int(cls8_used.sum()),
            'paired_segments': int((v2_used & cls8_used).sum()),
            'note': 'V2 and cls8 share the same 8-ch EEG cache, same ordering; paired by index.',
        },
        'timestamp': datetime.now().isoformat(),
    }
    with open(OUT_DIR / 'reproduction_check.json', 'w') as f:
        json.dump(summary, f, indent=2)
    log.info('Done.')


if __name__ == '__main__':
    main()
