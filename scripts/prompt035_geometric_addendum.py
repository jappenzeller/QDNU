#!/usr/bin/env python3
"""
================================================================================
PROMPT 035 (addendum, post PROMPT 036) - Basis-independent geometric polarity
measures vs clinical severity
================================================================================

PROMPT 036 established that polarity has no phase-lag reading. This addendum
adds two basis-INDEPENDENT geometric quantities per patient and makes them the
primary polarity measures for the severity correlation (|AUC - 0.5| becomes
secondary):

1. Shift magnitude ||s_p|| - norm of (mean tangent vector of seizure windows
   minus mean tangent vector of interictal windows) in the Tier 3 tangent
   space. TangentSpace(metric='riemann') is fit on the patient's own
   covariances, so the reference point is that patient's Frechet mean; the
   Euclidean norm in tangent coordinates equals the AIRM norm at the
   reference and is invariant to the measurement basis.

2. Bures-Wasserstein distance between the seizure and interictal Frechet
   means (Wasserstein barycenters via pyriemann mean_wasserstein, distance
   via distance_wasserstein) of the 8x8 covariance matrices.

Conventions (matching the Tier 2A/Tier 3 full-cohort run and
bw_polarity_check.py):
- 22 CHB-MIT patients (chb12, chb24 excluded), CH8 LOSO montage
- 20 s windows (10 s fallback if a class is missing), preprocess_eeg
  (demean, 0.5-128 Hz bandpass, 60 Hz notch), Covariances(estimator='lwf')
- Seizure class = ictal + preictal (label_map of the Tier pipeline), the
  same class definition under which the Tier 3 polarity/AUC was computed.

Outputs (into the existing results/prompt035/):
    geometric_measures.csv      per-patient ||s_p||, BW distance, window info
    polarity_magnitude.csv      REWRITTEN with the two new columns appended
    correlation_matrix.csv      Spearman rho/p for every clinical variable x
                                polarity measure (old + new)
    scatters/geo_*.png          scatter for every flagged geometric pair
    strong_weak_geometric.csv   strong vs weak polar comparison incl. new
                                measures with Mann-Whitney U
    SUMMARY.md                  addendum section appended

Exploratory: no multiple-comparison correction in the primary table; the
number of tests and Bonferroni survivors are reported.

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
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_wasserstein
from pyriemann.utils.distance import distance_wasserstein

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sagemaker.train_chbmit import (          # noqa: E402
    EXCLUDE_SUBJECTS,
    preprocess_eeg,
    read_edf_segment,
    extract_segments_for_subject,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT = Path("H:/Data/PythonDNU/EEG/chbmit")
OUT_DIR = PROJECT_ROOT / "results" / "prompt035"
SCATTER_DIR = OUT_DIR / "scatters"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCATTER_DIR.mkdir(parents=True, exist_ok=True)

LOSO_CHANNELS = ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
                 "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]

WINDOW_SEC = 20.0            # matches the full-cohort Tier 2A/Tier 3 run
FALLBACK_WINDOW_SEC = 10.0

STRONG_POLAR = ['chb03', 'chb21']
WEAK_POLAR = ['chb01', 'chb05', 'chb07', 'chb14']

CLINICAL_VARS = ['age_years', 'n_seizures', 'mean_seizure_duration_sec',
                 'median_seizure_duration_sec', 'max_seizure_duration_sec',
                 'n_seizure_files', 'total_recording_hours',
                 'seizure_density_per_hour']

GEO_MEASURES = ['s_norm', 's_norm_corr', 'bw_dist']
OLD_MEASURES = ['t2a_polarity_mag', 't3_polarity_mag']

FLAG_RHO, FLAG_P = 0.4, 0.05


# =============================================================================
# COVARIANCE LOADING (conventions of bw_polarity_check.py / tier3_loso_cv.py)
# =============================================================================

def load_covariances_for_subject(subject_id, window_sec=WINDOW_SEC):
    subject_dir = DATA_ROOT / subject_id
    if not subject_dir.exists():
        return np.array([]), np.array([])

    segments = extract_segments_for_subject(subject_dir)
    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    windows_list, labels_list = [], []

    for seg in segments:
        try:
            edf_path = DATA_ROOT / seg.subject / seg.source_file
            if not edf_path.exists():
                continue
            data, fs = read_edf_segment(str(edf_path), seg.start_sec,
                                        seg.end_sec, target_channels=LOSO_CHANNELS)
            if data is None or data.size == 0:
                continue
            expected = int(window_sec * fs)
            if data.shape[1] < int(window_sec * fs * 0.5):
                continue
            data = preprocess_eeg(data, fs=int(fs))
            if data.shape[1] >= expected:
                windows_list.append(data[:, :expected])
                labels_list.append(label_map.get(seg.label, 0))
        except Exception:
            continue

    if not windows_list:
        return np.array([]), np.array([])

    windows = np.array(windows_list)
    labels = np.array(labels_list)
    try:
        covs = Covariances(estimator='lwf').transform(windows)
    except Exception:
        return np.array([]), np.array([])

    valid = []
    for cov in covs:
        try:
            eigs = np.linalg.eigvalsh(cov)
            valid.append(bool(np.all(eigs > 0) and not np.isnan(cov).any()))
        except Exception:
            valid.append(False)
    valid = np.array(valid)
    return covs[valid], labels[valid]


def load_with_fallback(subject_id):
    covs, labels = load_covariances_for_subject(subject_id, WINDOW_SEC)
    if len(covs) > 0 and len(np.unique(labels)) == 2:
        return covs, labels, WINDOW_SEC
    covs, labels = load_covariances_for_subject(subject_id, FALLBACK_WINDOW_SEC)
    return covs, labels, FALLBACK_WINDOW_SEC


# =============================================================================
# GEOMETRIC MEASURES
# =============================================================================

def geometric_measures(covs: np.ndarray, labels: np.ndarray) -> dict:
    """||s_p|| in the patient's own tangent space + BW distance between the
    per-class Wasserstein barycenters."""
    m = {}
    ts = TangentSpace(metric='riemann')
    tv = ts.fit_transform(covs)                     # reference = patient mean
    tv_sz, tv_in = tv[labels == 1], tv[labels == 0]
    s_p = tv_sz.mean(axis=0) - tv_in.mean(axis=0)
    m['s_norm'] = float(np.linalg.norm(s_p))

    # Bias-corrected ||s_p||: E[||m_a - m_b||^2] = ||true||^2
    # + tr(Cov_a)/n_a + tr(Cov_b)/n_b, so patients with few seizure windows
    # get an inflated raw norm. Subtract the sampling-variance term.
    bias = (tv_sz.var(axis=0, ddof=1).sum() / len(tv_sz) +
            tv_in.var(axis=0, ddof=1).sum() / len(tv_in))
    m['s_norm_corr'] = float(np.sqrt(max(m['s_norm'] ** 2 - bias, 0.0)))

    c_sz = mean_wasserstein(covs[labels == 1])
    c_in = mean_wasserstein(covs[labels == 0])
    m['bw_dist'] = float(distance_wasserstein(c_sz, c_in))
    return m


# =============================================================================
# CSV HELPERS
# =============================================================================

def read_csv_dict(path: Path) -> dict:
    """Returns {patient: row-dict} keyed on the 'patient' column."""
    out = {}
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            out[row['patient']] = row
    return out


def fnum(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


# =============================================================================
# MAIN
# =============================================================================

def main():
    t0 = time.time()
    subjects = sorted(d.name for d in DATA_ROOT.iterdir()
                      if d.is_dir() and d.name.startswith('chb')
                      and d.name not in EXCLUDE_SUBJECTS)
    print(f"Subjects ({len(subjects)}): {subjects}")

    clinical = read_csv_dict(OUT_DIR / 'clinical_metadata.csv')
    magnitude = read_csv_dict(OUT_DIR / 'polarity_magnitude.csv')

    # ---- geometric measures per patient ----
    geo = {}
    for sid in subjects:
        covs, labels, w_used = load_with_fallback(sid)
        if len(covs) == 0 or len(np.unique(labels)) < 2:
            print(f"  {sid}: SKIP (no two-class covariances)")
            continue
        m = geometric_measures(covs, labels)
        m.update(window_sec=w_used, n_windows=len(labels),
                 n_seizure=int(labels.sum()))
        geo[sid] = m
        print(f"  {sid}: ||s_p||={m['s_norm']:.3f}  BW={m['bw_dist']:.4f}  "
              f"({m['n_windows']} win @ {w_used:.0f}s, {m['n_seizure']} seizure)")

    with open(OUT_DIR / 'geometric_measures.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['patient', 's_norm', 's_norm_corr', 'bw_dist', 'window_sec',
                    'n_windows', 'n_seizure_windows'])
        for sid in subjects:
            if sid in geo:
                g = geo[sid]
                w.writerow([sid, f"{g['s_norm']:.6f}", f"{g['s_norm_corr']:.6f}",
                            f"{g['bw_dist']:.6f}",
                            g['window_sec'], g['n_windows'], g['n_seizure']])

    # ---- rewrite polarity_magnitude.csv with the new columns appended ----
    mag_fields = ['patient', 't2a_raw_auc', 't2a_polarity_mag', 't2a_polarity',
                  't3_raw_auc', 't3_polarity_mag', 't3_polarity',
                  's_norm', 's_norm_corr', 'bw_dist']
    with open(OUT_DIR / 'polarity_magnitude.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(mag_fields)
        for sid in subjects:
            row = magnitude.get(sid, {})
            g = geo.get(sid, {})
            w.writerow([sid] +
                       [row.get(k, '') for k in mag_fields[1:7]] +
                       [f"{g['s_norm']:.6f}" if 's_norm' in g else '',
                        f"{g['s_norm_corr']:.6f}" if 's_norm_corr' in g else '',
                        f"{g['bw_dist']:.6f}" if 'bw_dist' in g else ''])

    # ---- correlation matrix (old + new measures) ----
    measures = {}
    for meas in OLD_MEASURES:
        measures[meas] = {sid: fnum(magnitude.get(sid, {}).get(meas))
                          for sid in subjects}
    for meas in GEO_MEASURES:
        measures[meas] = {sid: geo.get(sid, {}).get(meas, np.nan)
                          for sid in subjects}

    corr_rows, flagged = [], []
    for var in CLINICAL_VARS:
        cvals = {sid: fnum(clinical.get(sid, {}).get(var)) for sid in subjects}
        for meas, mvals in measures.items():
            x = np.array([cvals[s] for s in subjects])
            y = np.array([mvals[s] for s in subjects])
            ok = ~(np.isnan(x) | np.isnan(y))
            n = int(ok.sum())
            if n < 5:
                continue
            rho, p = spearmanr(x[ok], y[ok])
            flag = bool(abs(rho) > FLAG_RHO and p < FLAG_P)
            corr_rows.append(dict(clinical_variable=var, polarity_measure=meas,
                                  spearman_rho=float(rho), p_value=float(p),
                                  n=n, flag=flag))
            if flag and meas in GEO_MEASURES:
                flagged.append((var, meas, x[ok], y[ok], rho, p))

    n_tests = len(corr_rows)
    bonf_alpha = FLAG_P / n_tests
    with open(OUT_DIR / 'correlation_matrix.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['clinical_variable', 'polarity_measure', 'spearman_rho',
                    'p_value', 'n', 'flag', 'survives_bonferroni'])
        for r in sorted(corr_rows, key=lambda r: r['p_value']):
            w.writerow([r['clinical_variable'], r['polarity_measure'],
                        f"{r['spearman_rho']:+.4f}", f"{r['p_value']:.4f}",
                        r['n'], int(r['flag']),
                        int(r['p_value'] < bonf_alpha)])

    n_flag = sum(r['flag'] for r in corr_rows)
    n_bonf = sum(r['p_value'] < bonf_alpha for r in corr_rows)
    print(f"\nCorrelations: {n_tests} tests, {n_flag} flagged "
          f"(|rho|>{FLAG_RHO}, p<{FLAG_P}), {n_bonf} survive Bonferroni "
          f"(alpha={bonf_alpha:.4f})")

    # Confound diagnostic: raw ||s_p|| and BW distance are upward-biased when
    # estimated from few windows, and n_seizure_windows tracks n_seizures.
    diag = {}
    gsubs = [s for s in subjects if s in geo]
    nx = np.array([geo[s]['n_seizure'] for s in gsubs], dtype=float)
    for meas in GEO_MEASURES:
        gy = np.array([geo[s][meas] for s in gsubs])
        rho, p = spearmanr(nx, gy)
        diag[meas] = dict(rho=float(rho), p=float(p))
        print(f"  diagnostic {meas} vs n_seizure_windows: "
              f"rho={rho:+.3f} p={p:.4f}")

    # ---- scatters for flagged geometric pairs ----
    for var, meas, x, y, rho, p in flagged:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(x, y, s=45, alpha=0.8)
        z = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, np.polyval(z, xs), 'r--', lw=1)
        ax.set_xlabel(var)
        ax.set_ylabel(meas)
        ax.set_title(f"{meas} vs {var}\nSpearman rho={rho:+.3f}, p={p:.4f}")
        fig.tight_layout()
        fig.savefig(SCATTER_DIR / f"geo_{meas}_vs_{var}.png", dpi=130)
        plt.close(fig)

    # ---- strong vs weak polar (Step 4, extended with geometric measures) ----
    sw_rows = []
    all_vars = CLINICAL_VARS + list(measures.keys())
    for var in all_vars:
        if var in CLINICAL_VARS:
            vals = {sid: fnum(clinical.get(sid, {}).get(var)) for sid in subjects}
        else:
            vals = measures[var]
        sv = [vals[s] for s in STRONG_POLAR if not np.isnan(vals.get(s, np.nan))]
        wv = [vals[s] for s in WEAK_POLAR if not np.isnan(vals.get(s, np.nan))]
        if len(sv) < 2 or len(wv) < 2:
            continue
        try:
            _, p2 = mannwhitneyu(sv, wv, alternative='two-sided')
        except Exception:
            p2 = np.nan
        sw_rows.append(dict(variable=var,
                            strong_mean=float(np.mean(sv)),
                            weak_mean=float(np.mean(wv)),
                            p_two_sided=float(p2)))
    with open(OUT_DIR / 'strong_weak_geometric.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['variable', 'strong_mean_chb03_chb21',
                    'weak_mean_chb01_05_07_14', 'mw_p_two_sided'])
        for r in sw_rows:
            w.writerow([r['variable'], f"{r['strong_mean']:.4f}",
                        f"{r['weak_mean']:.4f}", f"{r['p_two_sided']:.4f}"])

    # ---- JSON + SUMMARY addendum ----
    with open(OUT_DIR / 'geometric_addendum.json', 'w') as f:
        json.dump(dict(generated=datetime.now().isoformat(),
                       window_sec=WINDOW_SEC, channels=LOSO_CHANNELS,
                       geo={s: geo[s] for s in geo},
                       correlations=corr_rows, n_tests=n_tests,
                       bonferroni_alpha=bonf_alpha,
                       confound_diagnostic_vs_n_seizure_windows=diag,
                       strong_weak=sw_rows), f, indent=1)

    lines = []
    a = lines.append
    a("")
    a("---")
    a("")
    a("## Geometric addendum (post PROMPT 036)")
    a("")
    a(f"**Generated:** {datetime.now().isoformat(timespec='seconds')}")
    a("")
    a("Two basis-independent geometric polarity measures per patient "
      "(primary; |AUC-0.5| is secondary): tangent-space shift magnitude "
      "||s_p|| (TangentSpace metric='riemann', reference = patient's own "
      "Frechet mean, seizure-class mean tangent vector minus interictal mean) "
      "and the Bures-Wasserstein distance between the per-class Wasserstein "
      "barycenters of the 8x8 covariances. 20 s windows, LWF covariance "
      "estimator, seizure class = ictal + preictal, as in the Tier 3 "
      "full-cohort run.")
    a("")
    a("| patient | ||s_p|| | ||s_p|| corr | BW dist | windows |")
    a("|---------|---------|--------------|---------|---------|")
    for sid in subjects:
        if sid in geo:
            g = geo[sid]
            a(f"| {sid} | {g['s_norm']:.3f} | {g['s_norm_corr']:.3f} | "
              f"{g['bw_dist']:.4f} | {g['n_windows']} @ {g['window_sec']:.0f}s |")
    a("")
    a("`s_norm_corr` subtracts the sampling-variance bias term "
      "tr(Cov)/n per class from ||s_p||^2 (raw norms are inflated for "
      "patients with few seizure windows). Confound diagnostic - Spearman "
      "vs n_seizure_windows: " +
      ", ".join(f"{k}: rho={v['rho']:+.3f} (p={v['p']:.3f})"
                for k, v in diag.items()) + ".")
    a("")
    a(f"Spearman correlations vs the {len(CLINICAL_VARS)} clinical variables "
      f"({n_tests} tests total incl. the Tier 2A/Tier 3 magnitudes; "
      f"Bonferroni alpha = {bonf_alpha:.4f}):")
    a("")
    geo_corr = [r for r in corr_rows if r['polarity_measure'] in GEO_MEASURES]
    a("| clinical variable | measure | rho | p | flag |")
    a("|-------------------|---------|-----|---|------|")
    for r in sorted(geo_corr, key=lambda r: r['p_value']):
        a(f"| {r['clinical_variable']} | {r['polarity_measure']} | "
          f"{r['spearman_rho']:+.3f} | {r['p_value']:.4f} | "
          f"{'FLAG' if r['flag'] else ''} |")
    a("")
    n_geo_flag = sum(r['flag'] for r in geo_corr)
    a(f"Flagged geometric correlations (|rho| > {FLAG_RHO}, p < {FLAG_P}): "
      f"**{n_geo_flag}/{len(geo_corr)}**; "
      f"surviving Bonferroni: {sum(r['p_value'] < bonf_alpha for r in geo_corr)}.")
    a("")
    a("Strong vs weak polar (geometric measures; n=2 vs n=4, two-sided "
      "Mann-Whitney):")
    a("")
    a("| variable | strong mean | weak mean | p |")
    a("|----------|-------------|-----------|---|")
    for r in sw_rows:
        if r['variable'] in GEO_MEASURES:
            a(f"| {r['variable']} | {r['strong_mean']:.4f} | "
              f"{r['weak_mean']:.4f} | {r['p_two_sided']:.4f} |")
    a("")
    a("### Addendum verdict")
    a("")
    a("(see reviewed verdict)")
    a("")

    with open(OUT_DIR / 'SUMMARY.md', 'a') as f:
        f.write('\n'.join(lines))

    print(f"\nDone in {time.time() - t0:.0f}s. Outputs in {OUT_DIR}")


if __name__ == '__main__':
    main()
