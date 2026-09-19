#!/usr/bin/env python3
"""
================================================================================
PROMPT 037 - Per-seizure decomposition of the geometric polarity measures
================================================================================

Disconfirmation run for the PROMPT 035 geometric addendum. The addendum's two
flagged correlations (||s_p|| vs n_seizures, bw_dist vs mean_seizure_duration)
have closed-form mechanical explanations (Step 0, prompt037_step0_diagnostics.py):
1/sqrt(K) averaging of a mean shift vector, and an ictal-window dropout in the
one-window-per-segment loader. This script rebuilds the measures per seizure
event so that the two mechanisms can be measured directly rather than inferred.

Stages (select with --stage; default runs all):
    cache     EDF pass -> results/prompt037/cache/W{W}/chbNN.npz
              (tiled W-second windows, each tagged (seizure_id, phase))
    analyze   Steps 2-5 from the cache -> CSV/JSON/PNG deliverables
    test      Unit tests 1 (identity) and 3 (synthetic recovery); no EDF reads
    legacy    Unit test 2: re-implement the addendum loader and reproduce
              results/prompt035/geometric_measures.csv s_norm / bw_dist

Conventions carried over from prompt035_geometric_addendum.py unchanged:
LOSO_CHANNELS (CH8), preprocess_eeg (demean, 0.5-128 Hz, 60 Hz notch),
Covariances(estimator='lwf'), EXCLUDE_SUBJECTS (chb12, chb24), 22 patients,
TangentSpace(metric='riemann') fit on all of the patient's covariances.

Author: Claude Code
Date: 2026-09-03
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
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr, norm, binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATA_ROOT = Path(os.environ.get("QDNU_DATA_ROOT",
                                        "H:/Data/PythonDNU/EEG/chbmit"))
OUT_DIR = PROJECT_ROOT / "results" / "prompt037"
P035_DIR = PROJECT_ROOT / "results" / "prompt035"

LOSO_CHANNELS = ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
                 "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]
EXCLUDE_SUBJECTS = {'chb12', 'chb24'}

N_INTERICTAL = 10          # cap kept for comparability with 035
SEED = 37
N_PERM = 2000
N_BOOT_FIXED = 200
RETENTION_MIN = 0.95
W20_RETENTION_MIN = 0.80

CLINICAL_VARS = ['age_years', 'n_seizures', 'mean_seizure_duration_sec',
                 'median_seizure_duration_sec', 'max_seizure_duration_sec',
                 'n_seizure_files', 'total_recording_hours',
                 'seizure_density_per_hour']

PRIMARY_TESTS = [
    ('P1', 'L_corr', 'n_seizures'),
    ('P2', 'c_corr', 'n_seizures'),
    ('P3', 'D_bar', 'mean_seizure_duration_sec'),
    ('P4', 'L_corr', 'mean_seizure_duration_sec'),
]
PRIMARY_MEASURES = ['L_corr', 'c_corr', 'D_bar']
SECONDARY_MEASURES = ['s_bar_norm', 'd_pooled', 't3_polarity_mag']


def log(msg=""):
    print(msg, flush=True)


# =============================================================================
# STEP 1 - LOADER WITH UNIFORM ICTAL REPRESENTATION
# =============================================================================

def build_segments(subject_dir: Path):
    """Segments with a seizure ordinal.  Reuses the 035 segment extractor and
    attaches seizure_id = ordinal in chbNN-summary.txt (1-based)."""
    from sagemaker.train_chbmit import (extract_segments_for_subject,
                                        parse_summary_file)
    summary = list(subject_dir.glob('*-summary.txt'))[0]
    seizures = parse_summary_file(summary)
    ordinal = {(z['file'], z['start']): i + 1 for i, z in enumerate(seizures)}
    durations = {i + 1: float(z['end'] - z['start'])
                 for i, z in enumerate(seizures)}
    segs = extract_segments_for_subject(subject_dir)
    out = []
    for seg in segs:
        if seg.label == 'ictal':
            sid = ordinal[(seg.source_file, int(seg.start_sec))]
        elif seg.label == 'preictal':
            sid = ordinal[(seg.source_file, int(seg.end_sec))]
        else:
            sid = 0
        out.append((seg, sid))
    return out, durations


def load_patient_windows(subject_id, data_root, window_sec, legacy=False):
    """Returns dict with covs (n,8,8), phase (str), seizure_id (int, 0 =
    interictal), window_idx (ordinal within its segment), durations dict.

    legacy=True reproduces the 035 loader exactly: one window per segment,
    first window_sec seconds, kept only if the segment holds >= window_sec.
    Interictal is *not* capped in legacy mode (035 did not cap; the 10 came
    from the files present on disk)."""
    from sagemaker.train_chbmit import preprocess_eeg, read_edf_segment
    from pyriemann.estimation import Covariances

    subject_dir = data_root / subject_id
    segs, durations = build_segments(subject_dir)
    windows, phase, sid_list, widx = [], [], [], []
    for seg, sid in segs:
        edf_path = data_root / seg.subject / seg.source_file
        if not edf_path.exists():
            continue
        data, fs = read_edf_segment(str(edf_path), seg.start_sec, seg.end_sec,
                                    target_channels=LOSO_CHANNELS)
        if data is None or data.size == 0:
            continue
        expected = int(window_sec * fs)
        if legacy and data.shape[1] < int(window_sec * fs * 0.5):
            continue
        data = preprocess_eeg(data, fs=int(fs))
        if legacy:
            if data.shape[1] >= expected:
                windows.append(data[:, :expected])
                phase.append(seg.label); sid_list.append(sid); widx.append(0)
        else:
            n_full = data.shape[1] // expected
            for i in range(n_full):
                windows.append(data[:, i * expected:(i + 1) * expected])
                phase.append(seg.label); sid_list.append(sid); widx.append(i)
    if not windows:
        return None
    windows = np.array(windows)
    covs = Covariances(estimator='lwf').transform(windows)
    valid = np.array([bool(np.all(np.linalg.eigvalsh(c) > 0)
                           and not np.isnan(c).any()) for c in covs])
    return dict(covs=covs[valid], phase=np.array(phase)[valid],
                seizure_id=np.array(sid_list)[valid],
                window_idx=np.array(widx)[valid], durations=durations)


def cap_interictal(d, n_keep, seed):
    """Deterministic interictal subsample. Returns mask + the kept indices."""
    inter = np.where(d['phase'] == 'interictal')[0]
    if len(inter) <= n_keep:
        keep = inter
    else:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(inter, n_keep, replace=False))
    mask = d['phase'] != 'interictal'
    mask[keep] = True
    return mask, keep


def cache_path(window_sec):
    return OUT_DIR / 'cache' / f'W{window_sec:g}'


def stage_cache(subjects, data_root, window_sec):
    cdir = cache_path(window_sec)
    cdir.mkdir(parents=True, exist_ok=True)
    for sid in subjects:
        f = cdir / f'{sid}.npz'
        if f.exists():
            log(f"  {sid}: cached")
            continue
        t0 = time.time()
        d = load_patient_windows(sid, data_root, window_sec)
        if d is None:
            log(f"  {sid}: NO WINDOWS"); continue
        mask, kept = cap_interictal(d, N_INTERICTAL, SEED)
        dur = d['durations']
        np.savez(f, covs=d['covs'][mask], phase=d['phase'][mask],
                 seizure_id=d['seizure_id'][mask],
                 window_idx=d['window_idx'][mask],
                 interictal_kept=kept,
                 n_interictal_total=int((d['phase'] == 'interictal').sum()),
                 seizure_ids=np.array(sorted(dur)),
                 durations=np.array([dur[k] for k in sorted(dur)]))
        n_ict = int((d['phase'][mask] == 'ictal').sum())
        log(f"  {sid}: {mask.sum()} windows ({n_ict} ictal, "
            f"{len(dur)} seizures) in {time.time()-t0:.1f}s")


def load_cache(sid, window_sec):
    z = np.load(cache_path(window_sec) / f'{sid}.npz', allow_pickle=False)
    return {k: z[k] for k in z.files}


# =============================================================================
# STEP 2 - PER-SEIZURE SHIFT VECTORS AND THE EXACT DECOMPOSITION
# =============================================================================

def tangent_vectors(covs):
    from pyriemann.tangentspace import TangentSpace
    return TangentSpace(metric='riemann').fit_transform(covs)


def decompose(tv_sz, sz_ids, tv_int, seizure_ids_all):
    """Core estimator.  tv_sz: tangent vectors of the seizure-phase windows,
    sz_ids: their seizure ordinal, tv_int: interictal tangent vectors,
    seizure_ids_all: ordered list of seizure ordinals to report (those with
    zero windows are reported as NaN and excluded from the decomposition).

    Returns per-seizure table + patient-level quantities, corrected and raw."""
    d = tv_sz.shape[1]
    n_int = len(tv_int)
    u_int = tv_int.mean(axis=0)
    b_int = tv_int.var(axis=0, ddof=1).sum() / n_int if n_int > 1 else 0.0

    ks = [k for k in seizure_ids_all if np.any(sz_ids == k)]
    K = len(ks)
    S, nks, bks, tr_within = [], [], [], []
    for k in ks:
        x = tv_sz[sz_ids == k]
        S.append(x.mean(axis=0) - u_int)
        nks.append(len(x))
        tr_within.append(x.var(axis=0, ddof=1).sum() if len(x) > 1 else np.nan)
    S = np.array(S); nks = np.array(nks); tr_within = np.array(tr_within)

    # within-seizure noise: per-seizure when n_k >= 2, else pooled over the
    # patient's multi-window seizures; if none exist fall back to the total
    # within-ictal variance (includes between-seizure spread -> conservative,
    # over-corrects Q).  Flag which fallback was used.
    multi = nks > 1
    if multi.any():
        pooled_tr = float(np.sum((nks[multi] - 1) * tr_within[multi]) /
                          np.sum(nks[multi] - 1))
        noise_source = 'pooled_within' if not multi.all() else 'per_seizure'
    else:
        pooled_tr = float(tv_sz.var(axis=0, ddof=1).sum()) if len(tv_sz) > 1 else 0.0
        noise_source = 'total_ictal_fallback'
    tr_k = np.where(multi, tr_within, pooled_tr)
    bks = tr_k / nks

    sq = np.sum(S ** 2, axis=1)                       # ||s^(k)||^2 raw
    sq_corr = sq - bks - b_int
    s_bar = S.mean(axis=0)
    s_bar_sq = float(np.sum(s_bar ** 2))

    G = S @ S.T
    off = G[~np.eye(K, dtype=bool)]
    Q = float(sq.mean())
    P = float(off.mean()) if K > 1 else np.nan
    Q_corr = float(sq_corr.mean())
    P_corr = float((off - b_int).mean()) if K > 1 else np.nan
    # identity: K^2 ||s_bar||^2 = sum ||s^(k)||^2 + sum_{j!=k} <s^(j),s^(k)>
    lhs = K ** 2 * s_bar_sq
    rhs = float(sq.sum() + off.sum())
    residual = abs(lhs - rhs) / max(abs(lhs), 1e-12)

    c = P / Q if K > 1 and Q > 0 else np.nan
    c_corr = P_corr / Q_corr if K > 1 and Q_corr > 0 else np.nan
    return dict(
        ks=ks, K=K, n_k=nks, s_vecs=S, sq=sq, sq_corr=sq_corr, b_k=bks,
        b_int=float(b_int), n_int=n_int, noise_source=noise_source,
        Q=Q, Q_corr=Q_corr, L_bar=float(np.sqrt(max(Q, 0))),
        L_corr=float(np.sqrt(max(Q_corr, 0))), P=P, P_corr=P_corr,
        c=c, c_corr=c_corr, s_bar_norm=float(np.sqrt(s_bar_sq)),
        ratio_sbar_Q=(s_bar_sq / Q if Q > 0 else np.nan),
        ratio_sbar_Qcorr=(s_bar_sq / Q_corr if Q_corr > 0 else np.nan),
        identity_residual=residual, Qcorr_nonpositive=bool(Q_corr <= 0),
        naive_mean_cos=(float(np.mean(
            [G[i, j] / np.sqrt(G[i, i] * G[j, j])
             for i in range(K) for j in range(K) if i != j])) if K > 1 else np.nan),
    )


def bw_per_seizure(covs_sz, sz_ids, covs_int, ks):
    from pyriemann.utils.mean import mean_wasserstein
    from pyriemann.utils.distance import distance_wasserstein
    C_int = mean_wasserstein(covs_int)
    dks = []
    for k in ks:
        x = covs_sz[sz_ids == k]
        C_k = mean_wasserstein(x) if len(x) > 1 else x[0]
        dks.append(float(distance_wasserstein(C_k, C_int)))
    d_pooled = float(distance_wasserstein(mean_wasserstein(covs_sz), C_int))
    return np.array(dks), d_pooled


def analyze_patient(sid, window_sec, arm='ictal'):
    z = load_cache(sid, window_sec)
    covs, phase, szid = z['covs'], z['phase'], z['seizure_id']
    tv = tangent_vectors(covs)                 # reference = patient mean
    m_int = phase == 'interictal'
    m_sz = phase == arm
    ks_all = [int(k) for k in z['seizure_ids']]
    dur = dict(zip([int(k) for k in z['seizure_ids']], z['durations']))
    if m_sz.sum() == 0 or m_int.sum() < 2:
        return None
    dec = decompose(tv[m_sz], szid[m_sz], tv[m_int], ks_all)
    dks, d_pooled = bw_per_seizure(covs[m_sz], szid[m_sz], covs[m_int], dec['ks'])
    dec.update(d_k=dks, D_bar=float(dks.mean()), d_pooled=d_pooled,
               bw_shrinkage=d_pooled / dks.mean() if dks.mean() > 0 else np.nan,
               durations=np.array([dur[k] for k in dec['ks']]),
               n_seizures_total=len(ks_all),
               retention=len(dec['ks']) / len(ks_all),
               n_windows_phase=int(m_sz.sum()), tv=tv, phase=phase, szid=szid,
               ks_all=ks_all, dur_all=dur)
    return dec


# =============================================================================
# STEP 3 - PERMUTATION OF WINDOW -> SEIZURE ASSIGNMENT
# =============================================================================

def permutation_coherence(dec, n_perm, seed):
    """Shuffle ictal windows among seizures holding n_k fixed."""
    tv, phase = dec['tv'], dec['phase']
    m_int = phase == 'interictal'
    tv_sz = dec['_tv_sz']; ids = dec['_ids']      # arm windows, seizures in ks
    if np.all(dec['n_k'] == 1):
        return None
    rng = np.random.default_rng(seed)
    obs = dec['c_corr']
    null = np.empty(n_perm)
    for i in range(n_perm):
        perm_ids = rng.permutation(ids)
        r = decompose(tv_sz, perm_ids, tv[m_int], dec['ks'])
        null[i] = r['c_corr']
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) < 100:
        return None
    ge = (np.sum(null >= obs) + 1) / (len(null) + 1)
    le = (np.sum(null <= obs) + 1) / (len(null) + 1)
    p_two = min(1.0, 2 * min(ge, le))
    direction = 1 if obs > np.median(null) else -1
    z = direction * norm.isf(p_two / 2)
    return dict(c_obs=obs, null_mean=float(np.median(null)),
                null_sd=float(np.subtract(*np.quantile(null, [.75, .25]))),
                null_q025=float(np.quantile(null, .025)),
                null_q975=float(np.quantile(null, .975)), p_two=p_two, z=z,
                direction=direction, n_perm=len(null))


# =============================================================================
# STATS HELPERS
# =============================================================================

def spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4:
        return np.nan, np.nan, int(ok.sum())
    r, p = spearmanr(x[ok], y[ok])
    return float(r), float(p), int(ok.sum())


def rank_partial(x, y, z):
    """Rank partial correlation of x,y given z (Spearman on residuals of
    ranks). Returns r, p (t-approx, df = n-3)."""
    from scipy.stats import rankdata, t as tdist
    x, y, z = map(lambda a: np.asarray(a, float), (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    n = len(x)
    if n < 5:
        return np.nan, np.nan, n
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy = np.corrcoef(rx, ry)[0, 1]; rxz = np.corrcoef(rx, rz)[0, 1]
    ryz = np.corrcoef(ry, rz)[0, 1]
    r = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    tstat = r * np.sqrt((n - 3) / (1 - r ** 2))
    p = 2 * tdist.sf(abs(tstat), n - 3)
    return float(r), float(p), n


def holm(pvals):
    p = np.asarray(pvals, float); m = len(p)
    order = np.argsort(p); adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def read_csv_dict(path):
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            out[row['patient']] = row
    return out


def fnum(x):
    try:
        v = float(x); return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def write_csv(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow([('' if (isinstance(v, float) and not np.isfinite(v)) else
                         (f"{v:.6g}" if isinstance(v, float) else v)) for v in r])


# =============================================================================
# STEP 4 / 5 - CROSS-PATIENT AND WITHIN-PATIENT TESTS
# =============================================================================

def cross_patient_tests(results, clinical, magnitude, out_dir, subjects, tag=''):
    def cvals(var):
        return np.array([fnum(clinical.get(s, {}).get(var)) for s in subjects])
    meas = {}
    for m in PRIMARY_MEASURES + ['s_bar_norm', 'd_pooled']:
        meas[m] = np.array([results[s][m] if s in results else np.nan for s in subjects])
    meas['t3_polarity_mag'] = np.array([fnum(magnitude.get(s, {}).get('t3_polarity_mag'))
                                        for s in subjects])
    retention = np.array([results[s]['retention'] if s in results else np.nan
                          for s in subjects])
    # exclude patients with Q_corr <= 0 from c tests
    c_ok = np.array([not results[s]['Qcorr_nonpositive'] if s in results else False
                     for s in subjects])
    meas_c = meas['c_corr'].copy(); meas_c[~c_ok] = np.nan
    meas['c_corr'] = meas_c

    prim_rows, pvals = [], []
    for pid, m, var in PRIMARY_TESTS:
        r, p, n = spear(meas[m], cvals(var))
        other = 'mean_seizure_duration_sec' if var == 'n_seizures' else 'n_seizures'
        pr, pp, pn = rank_partial(meas[m], cvals(var), cvals(other))
        rr, rp, rn = rank_partial(meas[m], cvals(var), retention)
        prim_rows.append([pid, m, var, r, p, n, pr, pp, other, rr, rp])
        pvals.append(p if np.isfinite(p) else 1.0)
    adj = holm(pvals)
    for row, a in zip(prim_rows, adj):
        row.append(float(a)); row.append(int(a < 0.05))
    write_csv(out_dir / f'primary_tests{tag}.csv',
              ['id', 'measure', 'clinical_variable', 'spearman_rho', 'p', 'n',
               'partial_rho_given_other', 'partial_p', 'other_variable',
               'partial_rho_given_retention', 'partial_p_retention',
               'holm_p', 'survives_holm'], prim_rows)

    expl = []
    all_meas = PRIMARY_MEASURES + SECONDARY_MEASURES
    for var in CLINICAL_VARS:
        for m in all_meas:
            r, p, n = spear(meas[m], cvals(var))
            expl.append([var, m, r, p, n])
    n_tests = len(expl); alpha = 0.05 / n_tests
    for row in expl:
        row.append(int(np.isfinite(row[3]) and row[3] < alpha))
    write_csv(out_dir / f'exploratory_correlations{tag}.csv',
              ['clinical_variable', 'measure', 'spearman_rho', 'p', 'n',
               f'survives_bonferroni_alpha_{alpha:.5f}'], expl)
    return prim_rows, expl, n_tests, alpha, meas


def within_patient_duration(results, subjects, out_dir, tag='', rng_seed=SEED):
    """Step 5. Per-patient Spearman of seizure duration vs d^(k) and vs
    ||s^(k)||^2_corr for K >= 4, Stouffer with weights sqrt(K-1), mixed model,
    and the fixed-window-count control."""
    rows, per = [], []
    for s in subjects:
        if s not in results:
            continue
        r = results[s]
        K = r['K']
        if K < 4:
            continue
        dur = r['durations']
        rd, pd_, _ = spear(dur, r['d_k'])
        rs, ps, _ = spear(dur, r['sq_corr'])
        rows.append([s, K, rd, pd_, rs, ps])
        per.append((s, K, rd, rs))
    write_csv(out_dir / f'within_patient_duration{tag}.csv',
              ['patient', 'K', 'rho_duration_vs_d_k', 'p_d', 'rho_duration_vs_sqnorm_corr',
               'p_s'], rows)

    def stouffer(vals):
        zs, ws = [], []
        for s, K, r in vals:
            if not np.isfinite(r) or K < 4:
                continue
            r = np.clip(r, -0.999, 0.999)
            zs.append(np.arctanh(r) * np.sqrt((K - 3) / 1.06)); ws.append(np.sqrt(K - 1))
        zs, ws = np.array(zs), np.array(ws)
        if len(zs) == 0:
            return np.nan, np.nan, 0
        Z = float(np.sum(ws * zs) / np.sqrt(np.sum(ws ** 2)))
        return Z, float(2 * norm.sf(abs(Z))), len(zs)

    Zd, pZd, nd = stouffer([(s, K, rd) for s, K, rd, rs in per])
    Zs, pZs, ns = stouffer([(s, K, rs) for s, K, rd, rs in per])

    # mixed model log d ~ log dur + (1|patient)
    mm = {}
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
        recs = []
        for s in subjects:
            if s in results:
                r = results[s]
                for k, dur, dk in zip(r['ks'], r['durations'], r['d_k']):
                    if dk > 0 and dur > 0:
                        recs.append(dict(patient=s, log_dur=np.log(dur), log_d=np.log(dk)))
        df = pd.DataFrame(recs)
        md = smf.mixedlm("log_d ~ log_dur", df, groups=df["patient"]).fit(reml=True)
        mm = dict(slope=float(md.params['log_dur']), se=float(md.bse['log_dur']),
                  p=float(md.pvalues['log_dur']), n_seizures=int(len(df)),
                  n_patients=int(df.patient.nunique()), converged=bool(md.converged))
        # Within/between decomposition (Mundlak): the random-intercept slope
        # above is a precision-weighted blend of the within-patient and the
        # between-patient slope, so it can inherit the cross-patient duration
        # effect (P3).  Split log_dur into patient mean + deviation.
        df['log_dur_between'] = df.groupby('patient')['log_dur'].transform('mean')
        df['log_dur_within'] = df['log_dur'] - df['log_dur_between']
        md_wb = smf.mixedlm("log_d ~ log_dur_within + log_dur_between", df,
                            groups=df["patient"]).fit(reml=True)
        mm['within_between'] = dict(
            within_slope=float(md_wb.params['log_dur_within']),
            within_se=float(md_wb.bse['log_dur_within']),
            within_p=float(md_wb.pvalues['log_dur_within']),
            between_slope=float(md_wb.params['log_dur_between']),
            between_se=float(md_wb.bse['log_dur_between']),
            between_p=float(md_wb.pvalues['log_dur_between']),
            converged=bool(md_wb.converged))
        try:
            md2 = smf.mixedlm("log_d ~ log_dur", df, groups=df["patient"],
                              re_formula="~log_dur").fit(reml=True)
            if md2.converged:
                mm['random_slope'] = dict(slope=float(md2.params['log_dur']),
                                          se=float(md2.bse['log_dur']),
                                          p=float(md2.pvalues['log_dur']))
        except Exception as e:
            mm['random_slope'] = f'did not converge: {e}'
    except Exception as e:
        mm = dict(error=str(e))

    # fixed window count control: 1 window per seizure, N_BOOT_FIXED draws
    rng = np.random.default_rng(rng_seed)
    from pyriemann.utils.mean import mean_wasserstein
    from pyriemann.utils.distance import distance_wasserstein
    boot_Zd, boot_rhod = [], []
    for b in range(N_BOOT_FIXED):
        vals = []
        for s in subjects:
            if s not in results or results[s]['K'] < 4:
                continue
            r = results[s]
            z = load_cache(s, r['window_sec'])
            covs, phase, szid = z['covs'], z['phase'], z['seizure_id']
            m_int = phase == 'interictal'
            C_int = r['_C_int']
            dks = []
            for k in r['ks']:
                idx = np.where((szid == k) & (phase == r['arm']))[0]
                j = rng.choice(idx)
                dks.append(distance_wasserstein(covs[j], C_int))
            rd, _, _ = spear(r['durations'], dks)
            vals.append((s, r['K'], rd))
        Z, _, _ = stouffer(vals)
        boot_Zd.append(Z); boot_rhod.append(np.nanmean([v[2] for v in vals]))
    fixed = dict(mean_stouffer_z=float(np.nanmean(boot_Zd)),
                 p_from_mean_z=float(2 * norm.sf(abs(np.nanmean(boot_Zd)))),
                 sd_stouffer_z=float(np.nanstd(boot_Zd)),
                 mean_per_patient_rho=float(np.nanmean(boot_rhod)),
                 n_boot=N_BOOT_FIXED)

    combined = dict(
        n_patients_K_ge_4=nd, n_seizures=int(sum(K for _, K, _, _ in per)),
        mean_rho_duration_vs_d=float(np.nanmean([rd for _, _, rd, _ in per])),
        stouffer_z_d=Zd, stouffer_p_d=pZd,
        mean_rho_duration_vs_sqnorm_corr=float(np.nanmean([rs for _, _, _, rs in per])),
        stouffer_z_sqnorm=Zs, stouffer_p_sqnorm=pZs,
        mixed_model_log_d_on_log_dur=mm, fixed_window_count_control=fixed,
        power_note=("Stouffer over 16 patients at K~7: true within-patient rho 0.40 "
                    "-> z~3.3; 0.25 -> z~2.0. Underpowered below rho~0.25."))
    with open(out_dir / f'within_patient_combined{tag}.json', 'w') as f:
        json.dump(combined, f, indent=2)
    return rows, combined


# =============================================================================
# FIGURES
# =============================================================================

def scatter(x, y, labels, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.scatter(x, y, s=45, alpha=0.8)
    for xi, yi, l in zip(x, y, labels):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(l.replace('chb', ''), (xi, yi), fontsize=7,
                        xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def within_panel(results, subjects, path):
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    cmap = plt.get_cmap('tab20')
    i = 0
    for s in subjects:
        if s not in results or results[s]['K'] < 4:
            continue
        r = results[s]
        ax.scatter(r['durations'], r['d_k'], s=28, alpha=0.85, color=cmap(i % 20),
                   label=s)
        i += 1
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('seizure duration (s)'); ax.set_ylabel('per-seizure BW distance d^(k)')
    ax.set_title('Within-patient duration vs d^(k), patients with K >= 4', fontsize=10)
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3, which='both')
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# =============================================================================
# ANALYZE STAGE
# =============================================================================

def run_analysis(subjects, window_sec, arm, out_dir, tag, clinical, magnitude,
                 do_perm=True, do_within=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    from pyriemann.utils.mean import mean_wasserstein
    results = {}
    inv_rows, sz_rows, pat_rows = [], [], []
    for s in subjects:
        if not (cache_path(window_sec) / f'{s}.npz').exists():
            log(f"  {s}: no cache"); continue
        dec = analyze_patient(s, window_sec, arm)
        if dec is None:
            log(f"  {s}: SKIP ({arm} arm empty)"); continue
        dec['window_sec'] = window_sec; dec['arm'] = arm
        z = load_cache(s, window_sec)
        m_int = z['phase'] == 'interictal'
        dec['_C_int'] = mean_wasserstein(z['covs'][m_int])
        m_sz = (z['phase'] == arm) & np.isin(z['seizure_id'], dec['ks'])
        dec['_tv_sz'] = dec['tv'][m_sz]; dec['_ids'] = z['seizure_id'][m_sz]
        results[s] = dec
        # inventory
        ph = z['phase']
        nk_ict = [int(((z['seizure_id'] == k) & (ph == 'ictal')).sum()) for k in dec['ks_all']]
        inv_rows.append([s, len(dec['ks_all']), int((ph == 'ictal').sum()),
                         int((ph == 'preictal').sum()), int(m_int.sum()),
                         int(z['n_interictal_total']),
                         min(nk_ict), float(np.median(nk_ict)), max(nk_ict),
                         float(np.mean(np.array(nk_ict) > 0)),
                         ' '.join(f"{d:g}" for d in z['durations'])])
        for k, nk, dur, sq, sqc, dk, bk in zip(dec['ks'], dec['n_k'], dec['durations'],
                                               dec['sq'], dec['sq_corr'], dec['d_k'], dec['b_k']):
            sz_rows.append([s, k, dur, nk, float(np.sqrt(sq)), float(sqc), float(bk), dk])
        pat_rows.append([s, dec['K'], dec['n_seizures_total'], dec['retention'],
                         dec['n_windows_phase'], dec['n_int'], dec['b_int'],
                         dec['noise_source'], dec['Q'], dec['Q_corr'], dec['L_bar'],
                         dec['L_corr'], dec['P'], dec['P_corr'], dec['c'], dec['c_corr'],
                         dec['naive_mean_cos'], dec['s_bar_norm'], dec['ratio_sbar_Q'],
                         dec['ratio_sbar_Qcorr'], 1.0 / dec['K'], dec['D_bar'],
                         dec['d_pooled'], dec['bw_shrinkage'], dec['identity_residual'],
                         int(dec['Qcorr_nonpositive'])])
        log(f"  {s}: K={dec['K']}/{dec['n_seizures_total']} L_corr={dec['L_corr']:.3f} "
            f"c_corr={dec['c_corr']:+.3f} |s_bar|^2/Q_corr={dec['ratio_sbar_Qcorr']:.3f} "
            f"(1/K={1/dec['K']:.3f}) D_bar={dec['D_bar']:.1f} d_pooled={dec['d_pooled']:.1f} "
            f"resid={dec['identity_residual']:.1e}")

    write_csv(out_dir / f'window_inventory{tag}.csv',
              ['patient', 'K', 'n_ictal_windows', 'n_preictal_windows', 'n_interictal_kept',
               'n_interictal_available', 'ictal_win_per_seizure_min', 'median', 'max',
               'ictal_retention_fraction', 'seizure_durations_sec'], inv_rows)
    write_csv(out_dir / f'per_seizure_shifts{tag}.csv',
              ['patient', 'seizure_id', 'duration_sec', 'n_windows', 's_k_norm',
               's_k_sqnorm_corr', 'b_k', 'd_k'], sz_rows)
    write_csv(out_dir / f'patient_decomposition{tag}.csv',
              ['patient', 'K', 'n_seizures_total', 'retention', f'n_{arm}_windows', 'n_int',
               'b_int', 'noise_source', 'Q', 'Q_corr', 'L_bar', 'L_corr', 'P', 'P_corr',
               'c', 'c_corr', 'naive_mean_cos', 's_bar_norm', 'sbar_sq_over_Q',
               'sbar_sq_over_Qcorr', 'one_over_K', 'D_bar', 'd_pooled', 'bw_shrinkage',
               'identity_residual', 'Qcorr_nonpositive'], pat_rows)

    # unit test 1
    worst = max(r['identity_residual'] for r in results.values())
    log(f"  identity residual (max, relative): {worst:.2e}  "
        f"{'OK' if worst < 1e-10 else 'FAIL'}")

    perm_rows, perm = [], {}
    if do_perm:
        log("Step 3: permutation of window -> seizure assignment")
        for s in subjects:
            if s not in results:
                continue
            pr = permutation_coherence(results[s], N_PERM, SEED)
            if pr is None:
                perm_rows.append([s, results[s]['K'], results[s]['c_corr'], '', '', '', '',
                                  '', '', 'excluded (nothing to shuffle or c undefined)'])
                continue
            perm[s] = pr
            perm_rows.append([s, results[s]['K'], pr['c_obs'], pr['null_mean'], pr['null_sd'],
                              pr['null_q025'], pr['null_q975'], pr['p_two'], pr['z'], ''])
            log(f"  {s}: c_corr={pr['c_obs']:+.3f} null median={pr['null_mean']:+.3f} "
                f"IQR={pr['null_sd']:.3f} p={pr['p_two']:.3f}")
        if perm:
            zs = np.array([p['z'] for p in perm.values()])
            Zc = float(zs.sum() / np.sqrt(len(zs)))
            n_above = int(sum(p['direction'] > 0 for p in perm.values()))
            sign_p = float(binomtest(n_above, len(perm), 0.5).pvalue)
            perm_rows.append(['COHORT', len(perm), '', '', '', '', '', '',
                              Zc, f'stouffer_z={Zc:.3f} p={2*norm.sf(abs(Zc)):.4f}; '
                              f'sign test {n_above}/{len(perm)} above null median p={sign_p:.3f}'])
            perm['_cohort'] = dict(stouffer_z=Zc, stouffer_p=float(2 * norm.sf(abs(Zc))),
                                   n_above=n_above, n=len(perm), sign_p=sign_p)
        write_csv(out_dir / f'permutation_coherence{tag}.csv',
                  ['patient', 'K', 'c_corr_obs', 'null_median', 'null_iqr', 'null_q025',
                   'null_q975', 'p_two_sided', 'z', 'note'], perm_rows)

    log("Step 4: cross-patient tests")
    prim, expl, n_tests, alpha, meas = cross_patient_tests(results, clinical, magnitude,
                                                           out_dir, subjects, tag)
    for row in prim:
        log(f"  {row[0]} {row[1]} vs {row[2]}: rho={row[3]:+.3f} p={row[4]:.4f} "
            f"holm={row[11]:.4f} partial|{row[8]}={row[6]:+.3f} (p={row[7]:.3f})")
    sc = out_dir / 'scatters'; sc.mkdir(exist_ok=True)
    for pid, m, var in PRIMARY_TESTS:
        x = np.array([fnum(clinical.get(s, {}).get(var)) for s in subjects])
        scatter(x, meas[m], subjects, var, m, f'{pid}{tag}: {m} vs {var}',
                sc / f'{pid}_{m}_vs_{var}{tag}.png')

    within = None
    if do_within:
        log("Step 5: within-patient duration test")
        rows, within = within_patient_duration(results, subjects, out_dir, tag)
        mm = within['mixed_model_log_d_on_log_dur']; wb = mm.get('within_between', {})
        log(f"  Stouffer z (d_k) = {within['stouffer_z_d']:+.3f} p={within['stouffer_p_d']:.4f}; "
            f"mixed-model slope = {mm.get('slope', float('nan')):+.3f} p={mm.get('p', float('nan')):.4f}; "
            f"within slope = {wb.get('within_slope', float('nan')):+.3f} p={wb.get('within_p', float('nan')):.4f}; "
            f"between slope = {wb.get('between_slope', float('nan')):+.3f} p={wb.get('between_p', float('nan')):.4f}; "
            f"fixed-count z = {within['fixed_window_count_control']['mean_stouffer_z']:+.3f}")
        within_panel(results, subjects, sc / f'within_patient_duration{tag}.png')
    return results, perm, prim, expl, n_tests, alpha, within, inv_rows


# =============================================================================
# UNIT TEST 3 - SYNTHETIC RECOVERY
# =============================================================================

SYNTH_TABLE = [  # K, n_k, L, c, expect Q_corr, expect P/Q, expect naive cos
    (7, 3, 3, 0.00, 9.0, -0.02, 0.14),
    (7, 3, 3, 0.29, 9.0, 0.28, 0.25),
    (7, 3, 5, 0.00, 25.0, -0.01, 0.09),
    (4, 2, 3, 0.00, 9.0, -0.01, 0.12),
    (7, 8, 3, 0.29, 9.0, 0.28, 0.36),
]


def synthetic_test(n_draws=4000, d=36, n_int=10, sigma=1.0, tol=0.03, seed=SEED):
    rng = np.random.default_rng(seed)
    rows, ok_all = [], True
    for K, nk, L, c, eQ, ePQ, eCos in SYNTH_TABLE:
        Qc, Pc, PQ, cosn = [], [], [], []
        for _ in range(n_draws):
            g = rng.standard_normal(d); g /= np.linalg.norm(g)
            S = []
            for k in range(K):
                h = rng.standard_normal(d); h /= np.linalg.norm(h)
                S.append(L * (np.sqrt(c) * g + np.sqrt(1 - c) * h))
            mu_int = rng.standard_normal(d)
            tv_int = mu_int + sigma * rng.standard_normal((n_int, d))
            tv_sz = np.vstack([mu_int + S[k] + sigma * rng.standard_normal((nk, d))
                               for k in range(K)])
            ids = np.repeat(np.arange(1, K + 1), nk)
            r = decompose(tv_sz, ids, tv_int, list(range(1, K + 1)))
            Qc.append(r['Q_corr']); Pc.append(r['P_corr'])
            PQ.append(r['P_corr'] / r['Q_corr'] if r['Q_corr'] > 0 else np.nan)
            cosn.append(r['naive_mean_cos'])
        mQ, mCos = np.mean(Qc), np.mean(cosn)
        # The estimator under test is "correct Q and P, then divide".  Across
        # draws the unbiased summary is the ratio of the mean corrected P to
        # the mean corrected Q; the mean of per-draw ratios is reported too
        # but carries a Jensen bias that grows as K*n_k shrinks.
        mPQ = float(np.mean(Pc) / np.mean(Qc))
        mPQ_perdraw = float(np.nanmean(PQ))
        ok = (abs(mQ - eQ) <= tol * eQ) and (abs(mPQ - ePQ) <= tol) and (abs(mCos - eCos) <= tol)
        ok_all &= ok
        rows.append([K, nk, L, c, eQ, mQ, ePQ, mPQ, mPQ_perdraw, eCos, mCos, int(ok)])
        log(f"  K={K} n_k={nk} L={L} c={c:.2f}: Q_corr={mQ:.3f} (exp {eQ})  "
            f"P/Q={mPQ:+.3f} (exp {ePQ:+.2f}; per-draw mean {mPQ_perdraw:+.3f})  "
            f"naive={mCos:+.3f} (exp {eCos:+.2f})  {'OK' if ok else 'FAIL'}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / 'synthetic_recovery.csv',
              ['K', 'n_k', 'L', 'true_c', 'expect_Q_corr', 'Q_corr', 'expect_P_over_Q',
               'P_over_Q_ratio_of_means', 'P_over_Q_mean_of_ratios', 'expect_naive_cos',
               'naive_cos', 'ok'], rows)
    return ok_all


# =============================================================================
# UNIT TEST 2 - LEGACY REPRODUCTION
# =============================================================================

def legacy_test(subjects, data_root):
    from pyriemann.utils.mean import mean_wasserstein
    from pyriemann.utils.distance import distance_wasserstein
    ref = read_csv_dict(P035_DIR / 'geometric_measures.csv')
    rows, n_ok = [], 0
    for s in subjects:
        d = load_patient_windows(s, data_root, 20.0, legacy=True)
        if d is None:
            rows.append([s, '', '', '', '', 0, 'no windows']); continue
        labels = (d['phase'] != 'interictal').astype(int)
        tv = tangent_vectors(d['covs'])
        s_p = tv[labels == 1].mean(0) - tv[labels == 0].mean(0)
        s_norm = float(np.linalg.norm(s_p))
        bw = float(distance_wasserstein(mean_wasserstein(d['covs'][labels == 1]),
                                        mean_wasserstein(d['covs'][labels == 0])))
        r_s, r_b = fnum(ref[s]['s_norm']), fnum(ref[s]['bw_dist'])
        ok = abs(s_norm - r_s) < 1e-6 and abs(bw - r_b) < 1e-6
        # 035 CSV carries 6 decimals -> 1e-6 is the resolution; use 5e-6
        ok = abs(s_norm - r_s) < 5e-6 and abs(bw - r_b) < 5e-6
        n_ok += ok
        rows.append([s, s_norm, r_s, bw, r_b, int(ok),
                     f"n={len(labels)} nsz={int(labels.sum())} (035: {ref[s]['n_windows']}/{ref[s]['n_seizure_windows']})"])
        log(f"  {s}: s_norm {s_norm:.6f} vs {r_s:.6f}  bw {bw:.6f} vs {r_b:.6f}  "
            f"{'OK' if ok else 'MISMATCH'}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / 'legacy_reproduction.csv',
              ['patient', 's_norm_legacy', 's_norm_035', 'bw_legacy', 'bw_035', 'ok', 'note'], rows)
    log(f"  legacy reproduction: {n_ok}/{len(subjects)} within 5e-6 "
        f"({'OK' if n_ok >= 20 else 'FAIL'})")
    return n_ok


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='all',
                    choices=['all', 'cache', 'analyze', 'test', 'legacy'])
    ap.add_argument('--data-root', default=str(DEFAULT_DATA_ROOT))
    ap.add_argument('--window', type=float, default=8.0)
    ap.add_argument('--subjects', default='', help='comma list; default all 22')
    ap.add_argument('--exclude', default='', help='comma list of patients to drop')
    ap.add_argument('--no-perm', action='store_true')
    ap.add_argument('--no-within', action='store_true')
    ap.add_argument('--arm', default='ictal', choices=['ictal', 'preictal'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default='', help='override output dir')
    args = ap.parse_args()

    data_root = Path(args.data_root)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.subjects:
        subjects = args.subjects.split(',')
    elif data_root.exists():
        subjects = sorted(d.name for d in data_root.iterdir()
                          if d.is_dir() and d.name.startswith('chb')
                          and d.name not in EXCLUDE_SUBJECTS)
    else:
        subjects = sorted(read_csv_dict(P035_DIR / 'geometric_measures.csv'))
    if args.exclude:
        subjects = [s for s in subjects if s not in args.exclude.split(',')]
    out_dir = Path(args.out) if args.out else OUT_DIR
    t0 = time.time()

    if args.stage in ('test',):
        log("Unit test 3: synthetic recovery")
        ok = synthetic_test()
        log(f"synthetic recovery: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    if args.stage == 'legacy':
        n_ok = legacy_test(subjects, data_root)
        sys.exit(0 if n_ok >= 20 else 1)

    if args.stage in ('all', 'cache'):
        log(f"Step 1: cache W={args.window:g}s from {data_root}")
        stage_cache(subjects, data_root, args.window)
        if args.stage == 'cache':
            log(f"Done in {time.time()-t0:.0f}s"); return

    clinical = read_csv_dict(P035_DIR / 'clinical_metadata.csv')
    magnitude = read_csv_dict(P035_DIR / 'polarity_magnitude.csv')
    log(f"Step 2: decomposition, W={args.window:g}s, arm={args.arm}")
    res = run_analysis(subjects, args.window, args.arm, out_dir, args.tag,
                       clinical, magnitude, do_perm=not args.no_perm,
                       do_within=not args.no_within)
    with open(out_dir / f'run_meta{args.tag}.json', 'w') as f:
        json.dump(dict(generated=datetime.now().isoformat(timespec='seconds'),
                       window_sec=args.window, arm=args.arm, subjects=subjects,
                       n_perm=N_PERM, n_boot_fixed=N_BOOT_FIXED, seed=SEED,
                       n_interictal_cap=N_INTERICTAL,
                       perm_cohort=(res[1].get('_cohort') if res[1] else None),
                       n_exploratory_tests=res[4], bonferroni_alpha=res[5]), f, indent=2)
    log(f"Done in {time.time()-t0:.0f}s. Outputs in {out_dir}")


if __name__ == '__main__':
    main()
