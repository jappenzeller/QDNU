#!/usr/bin/env python3
"""
================================================================================
PROMPT 036 - Mean phase lag Delta_ij as a polarity correlate
================================================================================

PROMPT 019 showed single-channel mean phase b is noise (pooled AUC 0.508).
This prompt tests the reference-independent quantity instead: the argument of
the complex PLV per channel pair,

    z_ij = (1/T) sum_t exp(i(phi_i(t) - phi_j(t)))
    r_ij = |z_ij|          (existing PLV magnitude)
    Delta_ij = arg(z_ij)   (mean phase lag, in (-pi, pi])

Hypothesis: standard-polarity and inverted-polarity patients show
opposite-sign ictal-vs-interictal phase-lag shifts delta_ij for a subset of
channel pairs.

Pipeline conventions (matching PROMPT 019 / b_parameter_investigation.py):
- 7-patient hardware subset, CH8 montage from classical_channel_scaling.json
- Segments from sagemaker.train_chbmit.extract_segments_for_subject;
  each segment is read as a full 30 s block from segment start
- Broadband 0.5-40 Hz bandpass, then theta (4-8 Hz) and alpha (8-15 Hz)
  band filters (butter order 4, filtfilt) + scipy.signal.hilbert.
  Filtering/Hilbert run on the full 30 s block; phases are then sliced into
  1.95 s and 20 s non-overlapping windows (avoids per-window edge artifacts).
- The delta_ij shift analysis uses STRICT ictal vs interictal windows
  (preictal excluded), as in PROMPT 019 Task 1.
- The LOSO XGBoost check (Step 4) also uses strict ictal vs interictal at
  20 s windows so that r-alone / Delta-alone / combined are internally
  comparable. PROMPT 019's b-alone 0.508 is an external reference point.

Classical feasibility check only - does NOT touch the A-Gate circuit or any
hardware baselines.

Outputs to results/prompt036/:
    cache/<subject>.npz              per-window complex-PLV features
    delta_table_<ws>.csv             28 pairs x 2 bands main table (output 1)
    delta_long_<ws>.csv              per-patient long format with concentrations
    heatmap_delta_<ws>.png           standard vs inverted heatmaps (output 2)
    loso_auc.csv                     LOSO AUC table (output 3)
    strong_weak.csv                  strong vs weak polar comparison (output 4)
    permutation_test.csv             global label-permutation null
    prompt036_results.json           machine-readable everything
    SUMMARY.md                       tables + verdict (output 5)

Author: Claude Code
Date: 2026-09-02
================================================================================
"""

from __future__ import annotations

import csv
import json
import sys
import time
import warnings
from itertools import combinations
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import circmean, circvar, mannwhitneyu, pearsonr
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sagemaker.train_chbmit import (          # noqa: E402
    FS,
    WINDOW_SEC,
    normalize_channel_label,
    extract_segments_for_subject,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
OUT_DIR = PROJECT_ROOT / "results" / "prompt036"
CACHE_DIR = OUT_DIR / "cache"
CLASSICAL_RESULTS = PROJECT_ROOT / "results" / "scaling" / "classical_channel_scaling.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']

# A-Gate hardware polarity at CH8-1.95s (reference labels from the prompt)
STANDARD = {'chb01', 'chb05', 'chb07', 'chb14'}
INVERTED = {'chb03', 'chb11', 'chb21'}
STRONG_POLAR = {'chb03', 'chb21'}
WEAK_POLAR = {'chb01', 'chb05', 'chb07', 'chb14'}   # chb11 excluded from Step 5

BROADBAND = (0.5, 40.0)
BANDS = {'theta': (4.0, 8.0), 'alpha': (8.0, 15.0)}
BAND_NAMES = list(BANDS.keys())

WINDOW_SIZES = {'1.95s': int(1.95 * FS), '20s': int(20.0 * FS)}   # 499, 5120
SEGMENT_SAMPLES = int(WINDOW_SEC * FS)                             # 7680

N_CH = 8
PAIRS = list(combinations(range(N_CH), 2))    # 28 pairs, i < j
N_PAIRS = len(PAIRS)

B_ALONE_AUC = 0.5083     # PROMPT 019 pooled b-alone reference
MIN_WINDOWS = 2          # minimum windows per condition for circmean

XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, random_state=42,
    eval_metric='logloss', n_jobs=-1,
)

LABEL_CODE = {'interictal': 0, 'preictal': 1, 'ictal': 2}


# =============================================================================
# HELPERS
# =============================================================================

def load_channels():
    with open(CLASSICAL_RESULTS, 'r') as f:
        return json.load(f)['ch8']['channels']


def bandpass(data: np.ndarray, lo: float, hi: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, [lo / nyq, min(hi / nyq, 0.99)], btype='band')
    return filtfilt(b, a, data, axis=-1)


def circ_diff(a, b):
    """Wrapped difference a - b in (-pi, pi]."""
    return np.angle(np.exp(1j * (np.asarray(a) - np.asarray(b))))


def circ_linear_corr(theta: np.ndarray, x: np.ndarray) -> float:
    """Mardia circular-linear correlation coefficient (0..1)."""
    c, s = np.cos(theta), np.sin(theta)
    try:
        rcx = pearsonr(c, x)[0]
        rsx = pearsonr(s, x)[0]
        rcs = pearsonr(c, s)[0]
    except Exception:
        return float('nan')
    if any(np.isnan(v) for v in (rcx, rsx, rcs)) or abs(rcs) >= 1.0:
        return float('nan')
    r2 = (rcx ** 2 + rsx ** 2 - 2 * rcx * rsx * rcs) / (1 - rcs ** 2)
    return float(np.sqrt(max(r2, 0.0)))


# =============================================================================
# STEP 1: PER-WINDOW COMPLEX PLV
# =============================================================================

def compute_subject_features(subject: str, channels) -> dict:
    """Compute per-window (r_ij, Delta_ij) for both bands and window sizes.

    Returns dict with, per window-size key ws:
        f'{ws}_r'     : (n_windows, n_bands, n_pairs) float32
        f'{ws}_delta' : (n_windows, n_bands, n_pairs) float32
        f'{ws}_label' : (n_windows,) int8  (0 interictal, 1 preictal, 2 ictal)
    """
    import pyedflib

    cache_path = CACHE_DIR / f"{subject}.npz"
    if cache_path.exists():
        with np.load(cache_path) as z:
            return {k: z[k] for k in z.files}

    subject_dir = DATA_DIR / subject
    segments = extract_segments_for_subject(subject_dir)

    # Group segments by source file so each EDF is opened/read once
    by_file: dict = {}
    for seg in segments:
        by_file.setdefault(seg.source_file, []).append(seg)

    norm_targets = [normalize_channel_label(ch) for ch in channels]

    out = {ws: {'r': [], 'delta': [], 'label': []} for ws in WINDOW_SIZES}
    t0 = time.time()

    for fname, segs in sorted(by_file.items()):
        edf_path = subject_dir / fname
        if not edf_path.exists():
            continue
        try:
            with pyedflib.EdfReader(str(edf_path)) as f:
                file_channels = [normalize_channel_label(f.getLabel(i))
                                 for i in range(f.signals_in_file)]
                idxs = []
                for t in norm_targets:
                    if t in file_channels:
                        idxs.append(file_channels.index(t))
                if len(idxs) != N_CH:
                    continue
                sig_len = f.getNSamples()[idxs[0]]
                raw = np.empty((N_CH, sig_len))
                for k, ci in enumerate(idxs):
                    raw[k] = f.readSignal(ci)
        except Exception as e:
            print(f"  [{subject}] skip {fname}: {e}")
            continue

        for seg in segs:
            start = int(seg.start_sec * FS)
            end = start + SEGMENT_SAMPLES
            if end > raw.shape[1]:
                continue
            block = raw[:, start:end]

            # broadband then per-band phases on the full 30 s block
            try:
                broad = bandpass(block, *BROADBAND, FS)
            except Exception:
                continue

            phases = np.empty((len(BANDS), N_CH, SEGMENT_SAMPLES))
            ok = True
            for bi, (lo, hi) in enumerate(BANDS.values()):
                try:
                    filt = bandpass(broad, lo, hi, FS)
                    phases[bi] = np.angle(hilbert(filt, axis=-1))
                except Exception:
                    ok = False
                    break
            if not ok:
                continue

            lab = LABEL_CODE[seg.label]
            for ws, wsamp in WINDOW_SIZES.items():
                n_win = SEGMENT_SAMPLES // wsamp
                for w in range(n_win):
                    sl = phases[:, :, w * wsamp:(w + 1) * wsamp]
                    r_row = np.empty((len(BANDS), N_PAIRS), dtype=np.float32)
                    d_row = np.empty((len(BANDS), N_PAIRS), dtype=np.float32)
                    for bi in range(len(BANDS)):
                        E = np.exp(1j * sl[bi])                     # (8, T)
                        Z = (E @ E.conj().T) / sl.shape[-1]         # (8, 8)
                        for pi_, (i, j) in enumerate(PAIRS):
                            z = Z[i, j]
                            r_row[bi, pi_] = np.abs(z)
                            d_row[bi, pi_] = np.angle(z)
                    out[ws]['r'].append(r_row)
                    out[ws]['delta'].append(d_row)
                    out[ws]['label'].append(lab)

    result = {}
    for ws in WINDOW_SIZES:
        result[f'{ws}_r'] = np.array(out[ws]['r'], dtype=np.float32)
        result[f'{ws}_delta'] = np.array(out[ws]['delta'], dtype=np.float32)
        result[f'{ws}_label'] = np.array(out[ws]['label'], dtype=np.int8)

    np.savez_compressed(cache_path, **result)
    n20 = len(result['20s_label'])
    n195 = len(result['1.95s_label'])
    print(f"  [{subject}] {n195} x 1.95s windows, {n20} x 20s windows "
          f"({time.time() - t0:.0f}s)")
    return result


# =============================================================================
# STEP 2: ICTAL VS INTERICTAL SHIFT
# =============================================================================

def compute_shifts(features: dict) -> dict:
    """Per window-size: per-subject delta_ij shift + concentrations.

    Returns shifts[ws] = {
        'delta':      (n_subj, n_bands, n_pairs)  wrapped shift, NaN if too few
        'conc_ictal': (n_subj, n_bands, n_pairs)  1 - circular variance
        'conc_inter': (n_subj, n_bands, n_pairs)
        'n_ictal':    (n_subj,), 'n_inter': (n_subj,)
    }
    """
    shifts = {}
    for ws in WINDOW_SIZES:
        n_s = len(SUBJECTS)
        d = np.full((n_s, len(BANDS), N_PAIRS), np.nan)
        ci = np.full_like(d, np.nan)
        cx = np.full_like(d, np.nan)
        n_ict = np.zeros(n_s, dtype=int)
        n_int = np.zeros(n_s, dtype=int)
        for si, subj in enumerate(SUBJECTS):
            lab = features[subj][f'{ws}_label']
            delta = features[subj][f'{ws}_delta']
            m_ict = lab == LABEL_CODE['ictal']
            m_int = lab == LABEL_CODE['interictal']
            n_ict[si], n_int[si] = int(m_ict.sum()), int(m_int.sum())
            if n_ict[si] < MIN_WINDOWS or n_int[si] < MIN_WINDOWS:
                continue
            for bi in range(len(BANDS)):
                for pi_ in range(N_PAIRS):
                    a_ict = delta[m_ict, bi, pi_].astype(np.float64)
                    a_int = delta[m_int, bi, pi_].astype(np.float64)
                    mu_ict = circmean(a_ict, high=np.pi, low=-np.pi)
                    mu_int = circmean(a_int, high=np.pi, low=-np.pi)
                    d[si, bi, pi_] = circ_diff(mu_ict, mu_int)
                    ci[si, bi, pi_] = 1.0 - circvar(a_ict, high=np.pi, low=-np.pi)
                    cx[si, bi, pi_] = 1.0 - circvar(a_int, high=np.pi, low=-np.pi)
        shifts[ws] = {'delta': d, 'conc_ictal': ci, 'conc_inter': cx,
                      'n_ictal': n_ict, 'n_inter': n_int}
    return shifts


# =============================================================================
# STEP 3: POLARITY CORRELATION
# =============================================================================

POL = np.array([+1.0 if s in STANDARD else -1.0 for s in SUBJECTS])


def pair_polarity_stats(deltas: np.ndarray, pol: np.ndarray) -> dict:
    """deltas: (7,) shifts for one pair/band. pol: +1 standard / -1 inverted."""
    valid = ~np.isnan(deltas)
    d, p = deltas[valid], pol[valid]
    n = int(valid.sum())
    if n < 4 or len(np.unique(p)) < 2:
        return dict(n=n, frac_std_pos=np.nan, frac_inv_pos=np.nan,
                    agreement=np.nan, r_pb=np.nan, p_pb=np.nan, r_cl=np.nan)
    std_mask, inv_mask = p > 0, p < 0
    frac_std_pos = float(np.mean(d[std_mask] > 0))
    frac_inv_pos = float(np.mean(d[inv_mask] > 0))
    k = int(np.sum((d > 0) == (p > 0)))       # standard->positive orientation
    agreement = max(k, n - k)                 # best of the two orientations
    r_pb, p_pb = pearsonr(p, d)
    r_cl = circ_linear_corr(d, p)
    return dict(n=n, frac_std_pos=frac_std_pos, frac_inv_pos=frac_inv_pos,
                agreement=agreement, r_pb=float(r_pb), p_pb=float(p_pb),
                r_cl=r_cl)


def polarity_analysis(shift: dict) -> list:
    """Returns list of row dicts, one per (band, pair)."""
    rows = []
    for bi, band in enumerate(BAND_NAMES):
        for pi_, (i, j) in enumerate(PAIRS):
            deltas = shift['delta'][:, bi, pi_]
            st = pair_polarity_stats(deltas, POL)
            flag = (not np.isnan(st['agreement']) and st['agreement'] >= 6) or \
                   (not np.isnan(st['r_pb']) and abs(st['r_pb']) > 0.6)
            rows.append(dict(band=band, pair_idx=pi_, i=i, j=j,
                             deltas=deltas, flag=flag,
                             conc_ictal_mean=float(np.nanmean(shift['conc_ictal'][:, bi, pi_])),
                             conc_inter_mean=float(np.nanmean(shift['conc_inter'][:, bi, pi_])),
                             **st))
    return rows


def permutation_test(shift: dict) -> dict:
    """Global null: all C(7,3) assignments of 3 'inverted' patients.

    Statistics per assignment: (a) number of (band, pair) tests with
    agreement >= 6/7, (b) mean |r_pb| across all 56 tests.
    """
    all_assignments = list(combinations(range(len(SUBJECTS)), 3))
    observed_inv = tuple(sorted(i for i, s in enumerate(SUBJECTS) if s in INVERTED))
    stats_flags, stats_meanr = [], []
    for assign in all_assignments:
        pol = np.array([-1.0 if i in assign else +1.0 for i in range(len(SUBJECTS))])
        n_flag, rs = 0, []
        for bi in range(len(BANDS)):
            for pi_ in range(N_PAIRS):
                st = pair_polarity_stats(shift['delta'][:, bi, pi_], pol)
                if not np.isnan(st['agreement']) and st['agreement'] >= 6:
                    n_flag += 1
                if not np.isnan(st['r_pb']):
                    rs.append(abs(st['r_pb']))
        stats_flags.append(n_flag)
        stats_meanr.append(float(np.mean(rs)) if rs else np.nan)
    obs_idx = all_assignments.index(observed_inv)
    obs_flags, obs_meanr = stats_flags[obs_idx], stats_meanr[obs_idx]
    p_flags = float(np.mean([s >= obs_flags for s in stats_flags]))
    p_meanr = float(np.mean([s >= obs_meanr for s in stats_meanr]))
    return dict(observed_n_flagged=obs_flags, observed_mean_abs_r=obs_meanr,
                p_n_flagged=p_flags, p_mean_abs_r=p_meanr,
                null_flags=stats_flags, null_meanr=stats_meanr)


# =============================================================================
# STEP 4: LOSO XGBOOST (20 s windows, strict ictal vs interictal)
# =============================================================================

def build_dataset(features: dict, ws: str = '20s'):
    X_r, X_cs, y, groups = [], [], [], []
    for si, subj in enumerate(SUBJECTS):
        lab = features[subj][f'{ws}_label']
        mask = (lab == LABEL_CODE['ictal']) | (lab == LABEL_CODE['interictal'])
        r = features[subj][f'{ws}_r'][mask].reshape(int(mask.sum()), -1)       # 56
        d = features[subj][f'{ws}_delta'][mask].reshape(int(mask.sum()), -1)
        cs = np.concatenate([np.cos(d), np.sin(d)], axis=1)                    # 112
        X_r.append(r)
        X_cs.append(cs)
        y.append((lab[mask] == LABEL_CODE['ictal']).astype(int))
        groups.append(np.full(int(mask.sum()), si))
    return (np.vstack(X_r), np.vstack(X_cs),
            np.concatenate(y), np.concatenate(groups))


def loso_xgb(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    from xgboost import XGBClassifier
    per_subj, all_true, all_score = {}, [], []
    for si, subj in enumerate(SUBJECTS):
        te = groups == si
        tr = ~te
        if len(np.unique(y[te])) < 2:
            continue
        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        per_subj[subj] = float(roc_auc_score(y[te], proba))
        all_true.append(y[te])
        all_score.append(proba)
    pooled = float(roc_auc_score(np.concatenate(all_true), np.concatenate(all_score)))
    return dict(pooled_auc=pooled, per_subject=per_subj,
                mean_auc=float(np.mean(list(per_subj.values()))))


# =============================================================================
# STEP 5: STRONG VS WEAK POLAR
# =============================================================================

def strong_weak_comparison(shift: dict) -> dict:
    res = {}
    for bi, band in enumerate(BAND_NAMES + ['both']):
        if band == 'both':
            mag = np.nanmean(np.abs(shift['delta']), axis=(1, 2))    # (7,)
        else:
            mag = np.nanmean(np.abs(shift['delta'][:, bi, :]), axis=1)
        per_subj = {s: float(mag[si]) for si, s in enumerate(SUBJECTS)}
        strong = [per_subj[s] for s in SUBJECTS if s in STRONG_POLAR]
        weak = [per_subj[s] for s in SUBJECTS if s in WEAK_POLAR]
        try:
            u_g, p_g = mannwhitneyu(strong, weak, alternative='greater')
            u_2, p_2 = mannwhitneyu(strong, weak, alternative='two-sided')
        except Exception:
            u_g = p_g = u_2 = p_2 = np.nan
        res[band] = dict(per_subject=per_subj,
                         strong_mean=float(np.mean(strong)),
                         weak_mean=float(np.mean(weak)),
                         mw_u=float(u_g), p_greater=float(p_g),
                         p_two_sided=float(p_2))
    return res


# =============================================================================
# OUTPUT WRITERS
# =============================================================================

def pair_name(i, j, channels):
    return f"{channels[i]}|{channels[j]}"


def write_delta_tables(ws: str, rows: list, shift: dict, channels):
    tag = ws.replace('.', 'p')
    main_path = OUT_DIR / f"delta_table_{tag}.csv"
    with open(main_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['band', 'pair'] + [f'delta_{s}' for s in SUBJECTS] +
                   ['conc_ictal_mean', 'conc_inter_mean', 'n_std_pos', 'n_inv_pos',
                    'sign_agreement', 'r_pointbiserial', 'p_pointbiserial',
                    'r_circlinear', 'flagged'])
        for r in rows:
            w.writerow([r['band'], pair_name(r['i'], r['j'], channels)] +
                       [f"{v:+.4f}" if not np.isnan(v) else '' for v in r['deltas']] +
                       [f"{r['conc_ictal_mean']:.4f}", f"{r['conc_inter_mean']:.4f}",
                        f"{r['frac_std_pos'] * 4:.0f}/4" if not np.isnan(r['frac_std_pos']) else '',
                        f"{r['frac_inv_pos'] * 3:.0f}/3" if not np.isnan(r['frac_inv_pos']) else '',
                        f"{r['agreement']:.0f}/7" if not np.isnan(r['agreement']) else '',
                        f"{r['r_pb']:+.4f}" if not np.isnan(r['r_pb']) else '',
                        f"{r['p_pb']:.4f}" if not np.isnan(r['p_pb']) else '',
                        f"{r['r_cl']:.4f}" if not np.isnan(r['r_cl']) else '',
                        int(r['flag'])])

    long_path = OUT_DIR / f"delta_long_{tag}.csv"
    with open(long_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['band', 'pair', 'subject', 'polarity', 'delta_rad',
                    'conc_ictal', 'conc_inter', 'n_ictal_windows', 'n_inter_windows'])
        for bi, band in enumerate(BAND_NAMES):
            for pi_, (i, j) in enumerate(PAIRS):
                for si, subj in enumerate(SUBJECTS):
                    d = shift['delta'][si, bi, pi_]
                    w.writerow([band, pair_name(i, j, channels), subj,
                                'standard' if subj in STANDARD else 'inverted',
                                f"{d:+.4f}" if not np.isnan(d) else '',
                                f"{shift['conc_ictal'][si, bi, pi_]:.4f}",
                                f"{shift['conc_inter'][si, bi, pi_]:.4f}",
                                shift['n_ictal'][si], shift['n_inter'][si]])


def plot_heatmaps(ws: str, shift: dict, channels):
    tag = ws.replace('.', 'p')
    fig, axes = plt.subplots(len(BANDS), 2, figsize=(13, 11))
    groups = [('Standard polarity (n=4)', [SUBJECTS.index(s) for s in sorted(STANDARD)]),
              ('Inverted polarity (n=3)', [SUBJECTS.index(s) for s in sorted(INVERTED)])]
    for bi, band in enumerate(BAND_NAMES):
        for gi, (gname, gidx) in enumerate(groups):
            M = np.full((N_CH, N_CH), np.nan)
            for pi_, (i, j) in enumerate(PAIRS):
                vals = shift['delta'][gidx, bi, pi_]
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    continue
                mu = circmean(vals, high=np.pi, low=-np.pi)
                mu = float(np.angle(np.exp(1j * mu)))
                M[i, j] = mu
                M[j, i] = -mu
            ax = axes[bi, gi]
            im = ax.imshow(M, cmap='coolwarm', vmin=-np.pi, vmax=np.pi)
            ax.set_xticks(range(N_CH))
            ax.set_yticks(range(N_CH))
            ax.set_xticklabels(channels, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(channels, fontsize=8)
            ax.set_title(f"{band} - {gname}", fontsize=11)
            fig.colorbar(im, ax=ax, fraction=0.046, label='circ-mean delta_ij (rad)')
    fig.suptitle(f"PROMPT 036: ictal-interictal phase-lag shift delta_ij, {ws} windows\n"
                 f"(polarity signature = opposite signs in same cells across columns)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_DIR / f"heatmap_delta_{tag}.png", dpi=150)
    plt.close(fig)


# =============================================================================
# SUMMARY
# =============================================================================

def write_summary(pol_rows, perms, loso, sw, shifts, channels):
    lines = []
    a = lines.append
    a("# PROMPT 036 - Mean phase lag Delta_ij as a polarity correlate")
    a("")
    a(f"**Generated:** {datetime.now().isoformat(timespec='seconds')}")
    a("")
    a("Conventions: CH8 montage, 0.5-40 Hz broadband, theta 4-8 / alpha 8-15 Hz,")
    a("30 s blocks per segment (PROMPT 019 loader), strict ictal vs interictal")
    a("(preictal excluded) for both the shift analysis and the LOSO check.")
    a("")

    for ws in WINDOW_SIZES:
        rows = pol_rows[ws]
        flagged = [r for r in rows if r['flag']]
        n_agree6 = sum(1 for r in rows if not np.isnan(r['agreement']) and r['agreement'] >= 6)
        n_r06 = sum(1 for r in rows if not np.isnan(r['r_pb']) and abs(r['r_pb']) > 0.6)
        a(f"## Polarity-discriminating pairs ({ws} windows)")
        a("")
        a(f"- Flagged (agreement >= 6/7 OR |r_pb| > 0.6): **{len(flagged)}/56**")
        a(f"  - by sign agreement >= 6/7: {n_agree6} (chance expectation ~7.0/56, "
          f"since P(>=6/7 best-orientation) = 0.125)")
        a(f"  - by |r_pb| > 0.6: {n_r06} (chance expectation ~8.7/56 at n=7)")
        p = perms[ws]
        a(f"- Label-permutation test over all C(7,3)=35 polarity assignments:")
        a(f"  - n flagged-by-agreement: observed {p['observed_n_flagged']}, "
          f"p = {p['p_n_flagged']:.3f} (floor 0.029)")
        a(f"  - mean |r_pb| across 56 tests: observed {p['observed_mean_abs_r']:.3f}, "
          f"p = {p['p_mean_abs_r']:.3f}")
        a("")
        if flagged:
            a("| band | pair | agreement | r_pb | p | r_circlin | delta by patient (" +
              " ".join(SUBJECTS) + ") |")
            a("|------|------|-----------|------|---|-----------|------------------|")
            for r in sorted(flagged,
                            key=lambda x: -abs(x['r_pb']) if not np.isnan(x['r_pb']) else 0):
                ds = ' '.join(f"{v:+.2f}" for v in r['deltas'])
                a(f"| {r['band']} | {pair_name(r['i'], r['j'], channels)} | "
                  f"{r['agreement']:.0f}/7 | {r['r_pb']:+.3f} | {r['p_pb']:.3f} | "
                  f"{r['r_cl']:.3f} | {ds} |")
            a("")

    a("## LOSO AUC (20 s windows, XGBoost, strict ictal vs interictal)")
    a("")
    a("| feature set | n features | pooled AUC | mean per-patient AUC |")
    a("|-------------|-----------|------------|----------------------|")
    a(f"| b alone (PROMPT 019 reference) | 8 | {B_ALONE_AUC:.4f} | - |")
    for key, nf in [('r_alone', 56), ('delta_alone', 112), ('r_plus_delta', 168)]:
        v = loso[key]
        a(f"| {key} | {nf} | {v['pooled_auc']:.4f} | {v['mean_auc']:.4f} |")
    a("")
    a("Per-patient AUC:")
    a("")
    a("| patient | r_alone | delta_alone | r_plus_delta |")
    a("|---------|---------|-------------|--------------|")
    for s in SUBJECTS:
        a(f"| {s} | " + " | ".join(
            f"{loso[k]['per_subject'].get(s, float('nan')):.4f}"
            for k in ('r_alone', 'delta_alone', 'r_plus_delta')) + " |")
    a("")
    inc = loso['r_plus_delta']['pooled_auc'] - loso['r_alone']['pooled_auc']
    a(f"Increment of r+Delta over r alone (pooled): {inc:+.4f}")
    a("")

    a("## Strong vs weak polar |delta_ij| (Step 5)")
    a("")
    a("| window | band | strong mean (chb03, chb21) | weak mean (chb01/05/07/14) | "
      "MW p (greater) | p (two-sided) |")
    a("|--------|------|---------------------------|----------------------------|"
      "---------------|---------------|")
    for ws in WINDOW_SIZES:
        for band, res in sw[ws].items():
            a(f"| {ws} | {band} | {res['strong_mean']:.4f} | {res['weak_mean']:.4f} | "
              f"{res['p_greater']:.4f} | {res['p_two_sided']:.4f} |")
    a("")
    a("(n=2 vs n=4: the one-sided Mann-Whitney p-value floor is 1/15 = 0.067; "
      "this comparison is directional evidence only.)")
    a("")

    # ---- auto-verdict skeleton (per the interpretation guide) ----
    d_auc = loso['delta_alone']['pooled_auc']
    ws_main = '20s'
    p_flag = perms[ws_main]['p_n_flagged']
    n_flagged = sum(r['flag'] for r in pol_rows[ws_main])
    a("## Verdict")
    a("")
    if d_auc < 0.55 and p_flag > 0.1:
        verdict = ("Delta-alone AUC is at or near chance and the number of "
                   "polarity-'discriminating' pairs is fully consistent with the "
                   "label-permutation null: phase lag is uninformative on this "
                   "subset and polarity shows no phase-lag signature. Direction 1 "
                   "loses its main payoff - report and move on.")
    elif d_auc >= 0.60 and p_flag > 0.1:
        verdict = ("Delta-alone carries real discriminative signal "
                   f"(pooled AUC {d_auc:.3f}) but the polarity sign structure does "
                   "not exceed the permutation null: phase lag is a useful feature, "
                   "yet polarity is not a phase-lag phenomenon. Direction 1 is "
                   "viable as encoding; polarity stays a projection artifact.")
    elif p_flag <= 0.06:
        verdict = (f"{n_flagged}/56 pair-band tests are polarity-discriminating, "
                   "exceeding the permutation null - polarity has a plausible "
                   "physiological reading as seizure phase-lag / propagation "
                   "direction. This supports direction 1 as the Paper 3 spine "
                   "(subject to the n=7 caveat).")
    else:
        verdict = (f"Intermediate result: Delta-alone pooled AUC {d_auc:.3f}, "
                   f"{n_flagged}/56 flagged pairs, permutation p {p_flag:.3f}. "
                   "Neither a clean null nor a clean polarity signature; see "
                   "tables above.")
    a(verdict)
    a("")
    a("*(Auto-generated verdict skeleton; see the reviewed one-paragraph verdict "
      "in the report.)*")

    with open(OUT_DIR / 'SUMMARY.md', 'w') as f:
        f.write('\n'.join(lines))


# =============================================================================
# MAIN
# =============================================================================

def main():
    t_start = time.time()
    channels = load_channels()
    print(f"CH8 montage: {channels}")
    print(f"Subjects: {SUBJECTS}")
    print(f"Bands: {BANDS} | Window samples: {WINDOW_SIZES}")

    # ---- Step 1: features ----
    print("\n=== Step 1: per-window complex PLV ===")
    features = {}
    for subj in SUBJECTS:
        print(f"Processing {subj}...")
        features[subj] = compute_subject_features(subj, channels)

    # ---- Step 2: shifts ----
    print("\n=== Step 2: ictal vs interictal shifts ===")
    shifts = compute_shifts(features)
    for ws in WINDOW_SIZES:
        s = shifts[ws]
        print(f"  {ws}: ictal windows {dict(zip(SUBJECTS, s['n_ictal'].tolist()))}")
        print(f"  {ws}: inter windows {dict(zip(SUBJECTS, s['n_inter'].tolist()))}")

    # ---- Step 3: polarity correlation + permutation ----
    print("\n=== Step 3: polarity correlation ===")
    pol_rows, perms = {}, {}
    for ws in WINDOW_SIZES:
        rows = polarity_analysis(shifts[ws])
        pol_rows[ws] = rows
        n_flag = sum(r['flag'] for r in rows)
        n_agree6 = sum(1 for r in rows
                       if not np.isnan(r['agreement']) and r['agreement'] >= 6)
        print(f"  {ws}: {n_flag}/56 flagged ({n_agree6} by sign agreement >= 6/7; "
              f"chance ~7/56 for that criterion alone)")
        perms[ws] = permutation_test(shifts[ws])
        print(f"  {ws}: permutation p (n_flagged) = {perms[ws]['p_n_flagged']:.3f}, "
              f"p (mean|r|) = {perms[ws]['p_mean_abs_r']:.3f}  [1/35 = 0.029 floor]")
        write_delta_tables(ws, rows, shifts[ws], channels)
        plot_heatmaps(ws, shifts[ws], channels)

    # ---- Step 4: LOSO XGBoost at 20 s ----
    print("\n=== Step 4: LOSO XGBoost (20s, ictal vs interictal) ===")
    X_r, X_cs, y, groups = build_dataset(features, '20s')
    print(f"  dataset: {len(y)} windows, {int(y.sum())} ictal, "
          f"{len(y) - int(y.sum())} interictal")
    loso = {}
    loso['r_alone'] = loso_xgb(X_r, y, groups)
    loso['delta_alone'] = loso_xgb(X_cs, y, groups)
    loso['r_plus_delta'] = loso_xgb(np.hstack([X_r, X_cs]), y, groups)
    for k, v in loso.items():
        print(f"  {k}: pooled AUC {v['pooled_auc']:.4f}, "
              f"mean per-patient {v['mean_auc']:.4f}")

    with open(OUT_DIR / 'loso_auc.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['feature_set', 'n_features', 'pooled_auc', 'mean_per_patient_auc'] +
                   [f'auc_{s}' for s in SUBJECTS])
        w.writerow(['b_alone (PROMPT 019 ref)', 8, B_ALONE_AUC, ''] + [''] * 7)
        for key, nf in [('r_alone', X_r.shape[1]), ('delta_alone', X_cs.shape[1]),
                        ('r_plus_delta', X_r.shape[1] + X_cs.shape[1])]:
            v = loso[key]
            w.writerow([key, nf, f"{v['pooled_auc']:.4f}", f"{v['mean_auc']:.4f}"] +
                       [f"{v['per_subject'].get(s, float('nan')):.4f}" for s in SUBJECTS])

    # ---- Step 5: strong vs weak polar ----
    print("\n=== Step 5: strong vs weak polar ===")
    sw = {}
    for ws in WINDOW_SIZES:
        sw[ws] = strong_weak_comparison(shifts[ws])
        b = sw[ws]['both']
        print(f"  {ws}: strong mean|delta|={b['strong_mean']:.3f} "
              f"weak={b['weak_mean']:.3f} p(greater)={b['p_greater']:.3f}")

    with open(OUT_DIR / 'strong_weak.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['window', 'band', 'group', 'subject', 'mean_abs_delta'])
        for ws in WINDOW_SIZES:
            for band, res in sw[ws].items():
                for s, v in res['per_subject'].items():
                    grp = ('strong' if s in STRONG_POLAR
                           else 'weak' if s in WEAK_POLAR else 'excluded(chb11)')
                    w.writerow([ws, band, grp, s, f"{v:.4f}"])
        w.writerow([])
        w.writerow(['window', 'band', 'strong_mean', 'weak_mean', 'mw_u',
                    'p_greater', 'p_two_sided'])
        for ws in WINDOW_SIZES:
            for band, res in sw[ws].items():
                w.writerow([ws, band, f"{res['strong_mean']:.4f}",
                            f"{res['weak_mean']:.4f}", res['mw_u'],
                            f"{res['p_greater']:.4f}", f"{res['p_two_sided']:.4f}"])

    with open(OUT_DIR / 'permutation_test.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['window', 'observed_n_flagged_agree6', 'p_n_flagged',
                    'observed_mean_abs_r', 'p_mean_abs_r', 'n_permutations'])
        for ws in WINDOW_SIZES:
            p = perms[ws]
            w.writerow([ws, p['observed_n_flagged'], f"{p['p_n_flagged']:.4f}",
                        f"{p['observed_mean_abs_r']:.4f}",
                        f"{p['p_mean_abs_r']:.4f}", 35])

    # ---- JSON dump ----
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o

    results = dict(
        generated=datetime.now().isoformat(),
        config=dict(subjects=SUBJECTS, channels=channels,
                    standard=sorted(STANDARD), inverted=sorted(INVERTED),
                    strong_polar=sorted(STRONG_POLAR), weak_polar=sorted(WEAK_POLAR),
                    bands=BANDS, broadband=BROADBAND,
                    window_samples=WINDOW_SIZES,
                    b_alone_reference=B_ALONE_AUC, xgb_params=XGB_PARAMS,
                    label_convention='strict ictal(1) vs interictal(0); preictal excluded'),
        shifts={ws: {k: shifts[ws][k] for k in
                     ('delta', 'conc_ictal', 'conc_inter', 'n_ictal', 'n_inter')}
                for ws in WINDOW_SIZES},
        polarity={ws: [{k: v for k, v in r.items()} for r in pol_rows[ws]]
                  for ws in WINDOW_SIZES},
        permutation=perms, loso=loso, strong_weak=sw,
    )
    with open(OUT_DIR / 'prompt036_results.json', 'w') as f:
        json.dump(clean(results), f, indent=1)

    write_summary(pol_rows, perms, loso, sw, shifts, channels)
    print(f"\nDone in {time.time() - t_start:.0f}s. Outputs in {OUT_DIR}")


if __name__ == '__main__':
    main()
