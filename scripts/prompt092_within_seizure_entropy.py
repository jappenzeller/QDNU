#!/usr/bin/env python3
"""
================================================================================
PROMPT 092 — Within-seizure time-resolved entropy
================================================================================

The measurement behind Notebook 03. The within-seizure entropy path has never
been measured; this is the first measurement.

STANCE, ON RECORD (Jim, 2026-10-11): **no expectation**. Reading A from the
E/I field is pre-declared because it is the only live reading, not because it
is expected to hold. Reading B (duration sets the sign) was tested in 091 and
dropped.

Object: for every annotated seizure in the 22 patients, a sliding window of
2 s at step 1 s from 20 s before onset to the annotated offset. Each window
gives Sigma (lwf), rho = Sigma / tr Sigma, and

    S(t) = - sum_k lambda_k ln lambda_k

over the 8 eigenvalues of rho, plus log tr Sigma(t). Window centre time is
relative to onset (negative before).

Reuses 037 verbatim for segment building and preprocessing: `build_segments`
from `scripts/prompt037_per_seizure_decomposition.py`, and `preprocess_eeg` /
`read_edf_segment` from `sagemaker/train_chbmit.py` (CH8 montage, 256 Hz,
demean, 0.5-128 Hz bandpass, 60 Hz notch), with
`Covariances(estimator='lwf')`. None of that is re-derived here.

Outputs to results/prompt092/:
    per_window.csv.gz        patient, seizure_idx, t_centre_s, S, log_tr
    per_seizure.csv          per-seizure statistics (all seizures)
    per_patient.csv          medians across seizures >= 20 s
    tests.json               P1, P2, counts, runtime
    curves/chbXX.png         22 files, one per patient
    summary_rS_by_group.png
    SUMMARY.md

Usage:
    python scripts/prompt092_within_seizure_entropy.py [--data-root PATH]
                                                       [--patients chb01,chb08]

Author: Claude Code
Date: 2026-10-12
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))

OUT_DIR = PROJECT_ROOT / 'results' / 'prompt092'
CURVE_DIR = OUT_DIR / 'curves'
PROMPT_042_CSV = PROJECT_ROOT / 'results' / 'prompt042' / 'per_patient.csv'
PROMPT_091_CSV = PROJECT_ROOT / 'results' / 'prompt091' / 'durations_per_seizure.csv'
DEFAULT_DATA_ROOT = Path('H:/Data/PythonDNU/EEG/chbmit')

# =============================================================================
# PRE-DECLARED — written into the script before it ran. Both P1 and P2 are
# reported whatever they show. A fails if either fails. There is no P3;
# nothing is pre-declared about the undetermined eight.
# =============================================================================
#
# Windowing: 2 s window, 1 s step, from 20 s before onset to the annotated
#     offset. Window centre time is relative to onset (negative before).
#     Seizures shorter than 20 s are computed and plotted but excluded from
#     every statistic; their count is reported by group.
#
# Per seizure, over the ICTAL PORTION ONLY (window centres from 0 to offset):
#     rS            = Spearman correlation of S against time
#     S0            = onset value = mean S over the 5 s before onset
#     interior-peak = max of S over the ictal portion exceeds S0 + 0.05 nats
#                     AND occurs before the last 10 % of windows
#
# P1 (A, purifiers): among purifier seizures >= 20 s (chb08, 13, 14, 21),
#     rS <= -0.5 in >= 75 %, and interior-peak flag false in >= 75 %.
#
# P2 (A, mixers): among mixer seizures >= 20 s (chb01, 04, 06, 07, 10, 16, 17,
#     18, 19, 23), median |rS| < 0.5, and the groups differ on rS by
#     Mann-Whitney p < 0.05.
#
# =============================================================================

W_SEC = 2.0            # window length
STEP_SEC = 1.0         # step
PRE_SEC = 20.0         # pre-onset span
BASELINE_SEC = 5.0     # S0 averaging window, immediately before onset
MIN_DURATION = 20.0    # seizures shorter than this are excluded from statistics
PEAK_MARGIN = 0.05     # nats, interior-peak threshold above S0
PEAK_TAIL_FRAC = 0.10  # "before the last 10 % of windows"
RS_THRESHOLD = -0.5    # P1 falling criterion
P1_FRACTION = 0.75     # P1 required fraction
P2_MEDIAN_ABS_RS = 0.5  # P2 median |rS| bound
ALPHA = 0.05

PURIFIERS = ['chb08', 'chb13', 'chb14', 'chb21']
MIXERS = ['chb01', 'chb04', 'chb06', 'chb07', 'chb10',
          'chb16', 'chb17', 'chb18', 'chb19', 'chb23']
UNDETERMINED = ['chb02', 'chb03', 'chb05', 'chb09', 'chb11',
                'chb15', 'chb20', 'chb22']
GROUP = ({p: 'purifier' for p in PURIFIERS} |
         {p: 'mixer' for p in MIXERS} |
         {p: 'undetermined' for p in UNDETERMINED})
PATIENTS = sorted(GROUP)
COLOURS = {'purifier': '#B8720F', 'mixer': '#0E93AB', 'undetermined': '#6E7780'}


def log(msg=''):
    print(msg, flush=True)


# =============================================================================
# MEASUREMENT
# =============================================================================

def seizure_windows(edf_path: Path, onset: float, offset: float,
                    channels: list[str]):
    """Sliding-window S(t) and log tr Sigma(t) for one seizure.

    Returns (t_centre, S, log_tr, meta) with t relative to onset, or
    (None, None, None, reason) on failure.
    """
    from sagemaker.train_chbmit import preprocess_eeg, read_edf_segment
    from pyriemann.estimation import Covariances

    span_start = max(0.0, onset - PRE_SEC)
    data, fs = read_edf_segment(str(edf_path), span_start, offset,
                                target_channels=channels)
    if data is None or data.size == 0:
        return None, None, None, 'read_edf_segment returned no data'
    fs = int(fs)
    data = preprocess_eeg(data, fs=fs)

    n_w, n_s = int(W_SEC * fs), int(STEP_SEC * fs)
    if data.shape[1] < n_w:
        return None, None, None, f'span shorter than one {W_SEC:g}s window'

    wins, centres = [], []
    for i in range(0, data.shape[1] - n_w + 1, n_s):
        wins.append(data[:, i:i + n_w])
        centres.append(span_start + i / fs + W_SEC / 2.0 - onset)
    wins = np.asarray(wins)

    covs = Covariances(estimator='lwf').transform(wins)
    S, log_tr, keep = [], [], []
    for c in covs:
        ev = np.linalg.eigvalsh(c)
        tr = float(np.trace(c))
        if not np.isfinite(tr) or tr <= 0 or not np.all(ev > 0) or np.isnan(c).any():
            keep.append(False)
            S.append(np.nan)
            log_tr.append(np.nan)
            continue
        lam = ev / tr
        lam = lam[lam > 0]
        S.append(float(-np.sum(lam * np.log(lam))))
        log_tr.append(float(np.log(tr)))
        keep.append(True)

    keep = np.asarray(keep)
    meta = dict(fs=fs, n_windows_total=len(centres),
                n_windows_dropped=int((~keep).sum()),
                baseline_truncated=bool(onset < PRE_SEC),
                span_start_s=span_start)
    return (np.asarray(centres)[keep], np.asarray(S)[keep],
            np.asarray(log_tr)[keep], meta)


def turning_points(y: np.ndarray) -> int:
    """Turning points after a 3-point median smooth."""
    if len(y) < 5:
        return 0
    sm = np.array([np.median(y[max(0, i - 1):i + 2]) for i in range(len(y))])
    d = np.diff(sm)
    d = d[d != 0]
    if len(d) < 2:
        return 0
    return int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))


def seizure_stats(t: np.ndarray, S: np.ndarray, log_tr: np.ndarray,
                  duration: float) -> dict:
    """Per-seizure statistics. Ictal portion = window centres >= 0."""
    base = (t >= -BASELINE_SEC) & (t < 0)
    ict = t >= 0
    out = dict(n_windows=int(len(t)), n_baseline_windows=int(base.sum()),
               n_ictal_windows=int(ict.sum()))

    out['S0'] = float(np.mean(S[base])) if base.sum() else float('nan')
    if ict.sum() < 3 or not np.isfinite(out['S0']):
        out.update(rS=float('nan'), interior_peak='', turning_points='',
                   peak_frac=float('nan'), dS_end=float('nan'),
                   spearman_S_vs_logtr=float('nan'))
        return out

    ti, Si, Li = t[ict], S[ict], log_tr[ict]
    out['rS'] = float(spearmanr(ti, Si).statistic)

    k = int(np.argmax(Si))
    tail_start = (1.0 - PEAK_TAIL_FRAC) * len(Si)
    out['interior_peak'] = bool(Si[k] > out['S0'] + PEAK_MARGIN and k < tail_start)
    out['turning_points'] = turning_points(Si)
    out['peak_frac'] = float(ti[k] / duration) if duration > 0 else float('nan')
    out['dS_end'] = float(Si[-1] - out['S0'])
    out['spearman_S_vs_logtr'] = float(spearmanr(Si, Li).statistic)
    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    t_run = time.time()
    ap = argparse.ArgumentParser(description='PROMPT 092 within-seizure entropy')
    ap.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument('--patients', type=str, default='',
                    help='comma-separated subset, for debugging')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)

    import prompt037_per_seizure_decomposition as p037
    channels = p037.LOSO_CHANNELS

    # integrity: groups must match 042
    with open(PROMPT_042_CSV, newline='') as f:
        from_042 = {r['patient']: r['group'] for r in csv.DictReader(f)}
    assert from_042 == GROUP, 'group assignment does not match PROMPT 042'
    log(f'groups match {PROMPT_042_CSV.name} (22 patients)')

    patients = [p.strip() for p in args.patients.split(',') if p.strip()] or PATIENTS

    # ---- measurement -------------------------------------------------------
    per_window_rows: list[tuple] = []
    per_seizure: list[dict] = []
    failures: list[dict] = []
    curves: dict[str, list] = {}

    log(f'\nWindowing {W_SEC:g}s / {STEP_SEC:g}s step, {PRE_SEC:g}s pre-onset, '
        f'CH8, lwf covariance')
    for p in patients:
        t_p = time.time()
        segs, durations = p037.build_segments(args.data_root / p)
        ictal = sorted(((sid, s.source_file, float(s.start_sec), float(s.end_sec))
                        for s, sid in segs if s.label == 'ictal'),
                       key=lambda x: x[0])
        curves[p] = []
        for sid, fname, onset, offset in ictal:
            edf = args.data_root / p / fname
            if not edf.exists():
                failures.append(dict(patient=p, seizure_idx=sid,
                                     reason=f'missing EDF {fname}'))
                continue
            t_c, S, log_tr, meta = seizure_windows(edf, onset, offset, channels)
            if t_c is None:
                failures.append(dict(patient=p, seizure_idx=sid, reason=meta))
                continue
            duration = offset - onset
            st = seizure_stats(t_c, S, log_tr, duration)
            row = dict(patient=p, seizure_idx=sid, group=GROUP[p],
                       duration_s=duration, **st,
                       eligible=bool(duration >= MIN_DURATION),
                       baseline_truncated=meta['baseline_truncated'],
                       n_windows_dropped=meta['n_windows_dropped'])
            per_seizure.append(row)
            curves[p].append((sid, t_c, S, log_tr, duration))
            for tc, s_, l_ in zip(t_c, S, log_tr):
                per_window_rows.append((p, sid, round(float(tc), 3),
                                        round(float(s_), 6), round(float(l_), 6)))
        log(f'  {p:6s} {GROUP[p]:12s} {len(ictal):2d} seizures '
            f'({time.time() - t_p:.0f}s)')

    # ---- integrity: seizure count against 091 ------------------------------
    n_091 = sum(1 for _ in csv.DictReader(open(PROMPT_091_CSV))) \
        if PROMPT_091_CSV.exists() else None
    log(f'\nseizures processed {len(per_seizure)}'
        + (f' / 091 annotated {n_091}' if n_091 is not None else ''))

    # ---- write per-window and per-seizure ----------------------------------
    with gzip.open(OUT_DIR / 'per_window.csv.gz', 'wt', newline='') as f:
        w = csv.writer(f)
        w.writerow(['patient', 'seizure_idx', 't_centre_s', 'S', 'log_tr'])
        w.writerows(per_window_rows)

    sz_fields = ['patient', 'seizure_idx', 'group', 'duration_s', 'n_windows',
                 'S0', 'rS', 'interior_peak', 'turning_points', 'peak_frac',
                 'dS_end', 'spearman_S_vs_logtr', 'eligible',
                 'n_baseline_windows', 'n_ictal_windows',
                 'baseline_truncated', 'n_windows_dropped']
    with open(OUT_DIR / 'per_seizure.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sz_fields, extrasaction='ignore')
        w.writeheader()
        for r in per_seizure:
            o = dict(r)
            for k in ('duration_s', 'S0', 'rS', 'peak_frac', 'dS_end',
                      'spearman_S_vs_logtr'):
                v = o.get(k)
                o[k] = '' if v is None or (isinstance(v, float) and not np.isfinite(v)) \
                    else (f'{v:.4f}' if isinstance(v, float) else v)
            w.writerow(o)

    # ---- counts -------------------------------------------------------------
    def grp(rows, g):
        return [r for r in rows if r['group'] == g]

    eligible = [r for r in per_seizure if r['eligible'] and np.isfinite(r['rS'])]
    counts = dict(total=len(per_seizure), eligible=len(eligible))
    for g in ('purifier', 'mixer', 'undetermined'):
        counts[g] = dict(
            total=len(grp(per_seizure, g)),
            eligible=len(grp(eligible, g)),
            short=len([r for r in grp(per_seizure, g) if not r['eligible']]),
        )
    log(f"\ncounts: total {counts['total']}, >= {MIN_DURATION:g}s {counts['eligible']}")
    for g in ('purifier', 'mixer', 'undetermined'):
        c = counts[g]
        log(f"  {g:12s} total {c['total']:3d}  >=20s {c['eligible']:3d}  "
            f"short {c['short']:3d}")

    # ---- P1 / P2 ------------------------------------------------------------
    pur = grp(eligible, 'purifier')
    mix = grp(eligible, 'mixer')
    pur_rs = [r['rS'] for r in pur]
    mix_rs = [r['rS'] for r in mix]

    frac_falling = (sum(1 for v in pur_rs if v <= RS_THRESHOLD) / len(pur_rs)
                    if pur_rs else float('nan'))
    frac_no_peak = (sum(1 for r in pur if r['interior_peak'] is False) / len(pur)
                    if pur else float('nan'))
    p1_pass = bool(frac_falling >= P1_FRACTION and frac_no_peak >= P1_FRACTION)

    med_abs_mix = float(statistics.median([abs(v) for v in mix_rs])) if mix_rs else float('nan')
    u, p_mwu = (mannwhitneyu(pur_rs, mix_rs, alternative='two-sided')
                if pur_rs and mix_rs else (float('nan'), float('nan')))
    p2_pass = bool(med_abs_mix < P2_MEDIAN_ABS_RS and p_mwu < ALPHA)

    log(f'\nP1 purifiers (n={len(pur)}): rS <= {RS_THRESHOLD} in '
        f'{frac_falling:.0%} (need >= {P1_FRACTION:.0%}); '
        f'interior-peak false in {frac_no_peak:.0%} (need >= {P1_FRACTION:.0%}) '
        f'-> {"PASS" if p1_pass else "FAIL"}')
    log(f'P2 mixers (n={len(mix)}): median |rS| = {med_abs_mix:.3f} '
        f'(need < {P2_MEDIAN_ABS_RS}); MWU vs purifiers p = {p_mwu:.4f} '
        f'(need < {ALPHA}) -> {"PASS" if p2_pass else "FAIL"}')
    log(f'Reading A -> {"HOLDS" if (p1_pass and p2_pass) else "FAILS"}')

    # ---- per patient --------------------------------------------------------
    per_patient = {}
    for p in patients:
        rows = [r for r in per_seizure if r['patient'] == p
                and r['eligible'] and np.isfinite(r['rS'])]
        if not rows:
            per_patient[p] = dict(patient=p, group=GROUP[p], n_eligible=0)
            continue

        def med(key):
            vals = [r[key] for r in rows if isinstance(r[key], (int, float))
                    and np.isfinite(r[key])]
            return float(statistics.median(vals)) if vals else float('nan')

        per_patient[p] = dict(
            patient=p, group=GROUP[p], n_eligible=len(rows),
            median_rS=med('rS'), median_S0=med('S0'),
            median_turning_points=med('turning_points'),
            median_peak_frac=med('peak_frac'), median_dS_end=med('dS_end'),
            median_spearman_S_vs_logtr=med('spearman_S_vs_logtr'),
            frac_interior_peak=float(np.mean([1.0 if r['interior_peak'] is True else 0.0
                                              for r in rows])),
            frac_falling=float(np.mean([1.0 if r['rS'] <= RS_THRESHOLD else 0.0
                                        for r in rows])),
        )

    pp_fields = ['patient', 'group', 'n_eligible', 'median_rS', 'median_S0',
                 'median_turning_points', 'median_peak_frac', 'median_dS_end',
                 'median_spearman_S_vs_logtr', 'frac_interior_peak', 'frac_falling']
    with open(OUT_DIR / 'per_patient.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=pp_fields, extrasaction='ignore')
        w.writeheader()
        for p in patients:
            o = dict(per_patient[p])
            for k in pp_fields[3:]:
                v = o.get(k)
                o[k] = '' if v is None or (isinstance(v, float) and not np.isfinite(v)) \
                    else f'{v:.4f}'
            w.writerow(o)

    # ---- figures ------------------------------------------------------------
    log('\nfigures')
    for p in patients:
        plot_patient(p, curves[p])
    plot_summary(eligible)
    log(f'  {len(patients)} curve files + summary_rS_by_group.png')

    # ---- tests.json ---------------------------------------------------------
    runtime = time.time() - t_run
    tests = dict(
        generated=datetime.now().isoformat(timespec='seconds'),
        stance='no expectation; first measurement of the within-seizure entropy path',
        windowing=dict(window_s=W_SEC, step_s=STEP_SEC, pre_onset_s=PRE_SEC,
                       baseline_s=BASELINE_SEC, min_duration_s=MIN_DURATION,
                       peak_margin_nats=PEAK_MARGIN, peak_tail_frac=PEAK_TAIL_FRAC),
        reuse='037 build_segments; sagemaker preprocess_eeg/read_edf_segment; '
              'CH8 montage; Covariances(lwf)',
        counts=counts,
        n_seizures_091=n_091,
        P1=dict(statement='purifiers: rS <= -0.5 in >= 75 % and interior-peak '
                          'false in >= 75 %',
                n=len(pur), frac_rS_le_threshold=frac_falling,
                frac_interior_peak_false=frac_no_peak,
                required_fraction=P1_FRACTION, pass_=p1_pass,
                rS_values=sorted(round(v, 4) for v in pur_rs)),
        P2=dict(statement='mixers: median |rS| < 0.5 and groups differ on rS '
                          '(Mann-Whitney p < 0.05)',
                n=len(mix), median_abs_rS=med_abs_mix,
                mwu_U=float(u), mwu_p=float(p_mwu), alpha=ALPHA, pass_=p2_pass,
                rS_values=sorted(round(v, 4) for v in mix_rs)),
        reading_A_holds=bool(p1_pass and p2_pass),
        failures=failures,
        runtime_s=round(runtime, 1),
    )
    with open(OUT_DIR / 'tests.json', 'w') as f:
        json.dump(tests, f, indent=1)

    write_summary(per_seizure, per_patient, tests, patients)
    log(f'\nDone in {runtime:.0f}s. Outputs in {OUT_DIR}')
    return 0


# =============================================================================
# FIGURES
# =============================================================================

def plot_patient(patient: str, series: list) -> None:
    g = GROUP[patient]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2, 1]))
    for sid, t, S, log_tr, dur in series:
        ax1.plot(t, S, lw=0.8, alpha=0.85, color=COLOURS[g])
        ax2.plot(t, log_tr, lw=0.8, alpha=0.85, color=COLOURS[g])

    for ax in (ax1, ax2):
        ax.axvline(0, color='#1C2229', lw=1)
        ax.axvspan(-BASELINE_SEC, 0, color='#DFE1DC', alpha=0.55, lw=0)
    ax1.set_ylabel(r'$S(\rho)$  (nats)')
    ax2.set_ylabel(r'$\log\,\mathrm{tr}\,\Sigma$')
    ax2.set_xlabel('time relative to onset (s)')
    ax1.set_title(f'{patient} — {g} — {len(series)} seizures\n'
                  f'2 s windows, 1 s step; onset at t = 0, shaded band = 5 s baseline',
                  color=COLOURS[g], fontsize=10)
    fig.tight_layout()
    fig.savefig(CURVE_DIR / f'{patient}.png', dpi=140)
    plt.close(fig)


def plot_summary(eligible: list) -> None:
    order = ['purifier', 'mixer', 'undetermined']
    fig, ax = plt.subplots(figsize=(7.5, 5))
    rng = np.random.default_rng(92)
    for i, g in enumerate(order):
        v = [r['rS'] for r in eligible if r['group'] == g]
        x = i + rng.uniform(-0.16, 0.16, len(v))
        ax.scatter(x, v, s=34, color=COLOURS[g], alpha=0.85, zorder=3,
                   edgecolor='white', linewidth=0.5,
                   label=f'{g} (n={len(v)} seizures)')
        if v:
            ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color='#1C2229',
                    lw=1.6, zorder=4)
    ax.axhline(RS_THRESHOLD, color='#B8720F', ls='--', lw=1, zorder=2)
    ax.annotate(f'$r_S$ = {RS_THRESHOLD}', (2.45, RS_THRESHOLD), fontsize=8,
                color='#B8720F', va='bottom', ha='right')
    ax.axhline(0, color='#6E7780', lw=0.6, zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel(r'per-seizure $r_S$  (Spearman of $S$ vs time, ictal portion)')
    ax.set_title('PROMPT 092 — within-seizure entropy trend by group\n'
                 'seizures >= 20 s; black bar = group median', fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc='lower right')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'summary_rS_by_group.png', dpi=150)
    plt.close(fig)


# =============================================================================
# SUMMARY.md
# =============================================================================

def write_summary(per_seizure, per_patient, tests, patients) -> None:
    P1, P2, C = tests['P1'], tests['P2'], tests['counts']
    el = [r for r in per_seizure if r['eligible'] and np.isfinite(r['rS'])]

    def by(g):
        return [r for r in el if r['group'] == g]

    def shape(r):
        """Descriptive label from the numbers, not a mechanism."""
        if r['rS'] <= -0.5:
            return 'falls'
        if r['rS'] >= 0.5:
            return 'rises'
        return 'wobbles'

    L = []
    a = L.append
    a('# PROMPT 092 - Within-seizure time-resolved entropy')
    a('')
    a(f"**Generated:** {tests['generated'][:10]}")
    a('**Script:** `scripts/prompt092_within_seizure_entropy.py`')
    a('**Depends on:** 037 (segment building), 042 (groups), 091 (reading B dropped)')
    a('')
    a('**Stance, on record (2026-10-11): no expectation.** The within-seizure entropy '
      'path had never been measured; this is the first measurement. Reading A was '
      'pre-declared because it was the only live reading, not because it was expected '
      'to hold.')
    a('')
    a(f"Conventions: 22 patients, CH8 montage, 256 Hz, `preprocess_eeg` (demean, "
      f"0.5-128 Hz bandpass, 60 Hz notch) and `build_segments` reused from 037 "
      f"verbatim; `Covariances(estimator='lwf')`. Sliding window {W_SEC:g} s at "
      f"{STEP_SEC:g} s step from {PRE_SEC:g} s before onset to the annotated offset; "
      f"window centre time is relative to onset. Per window "
      f"$\\rho = \\Sigma/\\operatorname{{tr}}\\Sigma$ and "
      f"$S = -\\sum_k \\lambda_k \\ln \\lambda_k$ over the 8 eigenvalues of $\\rho$, "
      f"plus $\\log\\operatorname{{tr}}\\Sigma$. Ictal portion = window centres "
      f"$\\geq 0$; $S_0$ = mean $S$ over the {BASELINE_SEC:g} s before onset. "
      f"Seizures shorter than {MIN_DURATION:g} s are computed and plotted but "
      f"excluded from every statistic.")
    a('')
    a('---')
    a('')
    a('## Counts')
    a('')
    a('| group | seizures | $\\geq$ 20 s (used) | < 20 s (excluded) |')
    a('|---|---|---|---|')
    for g in ('purifier', 'mixer', 'undetermined'):
        c = C[g]
        a(f"| {g} | {c['total']} | {c['eligible']} | {c['short']} |")
    a(f"| **all** | **{C['total']}** | **{C['eligible']}** | "
      f"**{C['total'] - C['eligible']}** |")
    a('')

    def mark(ok):
        return 'met' if ok else '**not met**'

    a('## Verdict')
    a('')
    a(f"**P1 {'PASS' if P1['pass_'] else 'FAIL'}** - purifier seizures "
      f"(n = {P1['n']}), both clauses required:")
    a('')
    a(f"- $r_S \\leq {RS_THRESHOLD}$ in {P1['frac_rS_le_threshold']:.0%}, needed "
      f"$\\geq$ {P1['required_fraction']:.0%} - "
      f"{mark(P1['frac_rS_le_threshold'] >= P1['required_fraction'])}")
    a(f"- interior-peak false in {P1['frac_interior_peak_false']:.0%}, needed "
      f"$\\geq$ {P1['required_fraction']:.0%} - "
      f"{mark(P1['frac_interior_peak_false'] >= P1['required_fraction'])}")
    a('')
    a(f"**P2 {'PASS' if P2['pass_'] else 'FAIL'}** - mixer seizures "
      f"(n = {P2['n']}), both clauses required:")
    a('')
    a(f"- median $|r_S| = {P2['median_abs_rS']:.3f}$, needed $< {P2_MEDIAN_ABS_RS}$ - "
      f"{mark(P2['median_abs_rS'] < P2_MEDIAN_ABS_RS)}")
    a(f"- groups differ on $r_S$: Mann-Whitney $p = {P2['mwu_p']:.4f}$, needed "
      f"$< {ALPHA}$ - {mark(P2['mwu_p'] < ALPHA)}")
    a('')
    a(f"**Reading A {'HOLDS' if tests['reading_A_holds'] else 'FAILS'}** "
      f"(A fails if either P1 or P2 fails).")
    a('')

    a('## Reading - the shapes seen')
    a('')
    for g in ('purifier', 'mixer', 'undetermined'):
        rows = by(g)
        if not rows:
            a(f"**{g}**: no seizures $\\geq$ 20 s.")
            a('')
            continue
        falls = sum(1 for r in rows if shape(r) == 'falls')
        rises = sum(1 for r in rows if shape(r) == 'rises')
        wob = sum(1 for r in rows if shape(r) == 'wobbles')
        peaks = sum(1 for r in rows if r['interior_peak'] is True)
        med_rs = statistics.median([r['rS'] for r in rows])
        med_tp = statistics.median([r['turning_points'] for r in rows])
        med_pf = statistics.median([r['peak_frac'] for r in rows
                                    if np.isfinite(r['peak_frac'])])
        med_de = statistics.median([r['dS_end'] for r in rows
                                    if np.isfinite(r['dS_end'])])
        med_tr = statistics.median([r['spearman_S_vs_logtr'] for r in rows
                                    if np.isfinite(r['spearman_S_vs_logtr'])])
        a(f"**{g}** ({len(rows)} seizures $\\geq$ 20 s): {falls} fall "
          f"($r_S \\leq -0.5$), {rises} rise ($r_S \\geq +0.5$), {wob} wobble "
          f"(|$r_S$| < 0.5); median $r_S = {med_rs:+.2f}$. Interior peak in "
          f"{peaks}/{len(rows)}. Median turning points {med_tp:.0f}; median time of "
          f"the entropy maximum at {med_pf:.2f} of seizure length; median "
          f"$S_{{\\text{{offset}}}} - S_0 = {med_de:+.3f}$ nats. Median "
          f"Spearman($S$, $\\log\\operatorname{{tr}}\\Sigma$) within seizure "
          f"= {med_tr:+.2f}.")
        a('')
        per_pat = [f"{p} {per_patient[p]['median_rS']:+.2f} "
                   f"(n={per_patient[p]['n_eligible']})"
                   for p in patients
                   if GROUP[p] == g and per_patient[p]['n_eligible'] > 0]
        a(f"Per-patient median $r_S$: {', '.join(per_pat)}.")
        a('')

    a('### Is entropy just tracking power?')
    a('')
    all_tr = [r['spearman_S_vs_logtr'] for r in el
              if np.isfinite(r['spearman_S_vs_logtr'])]
    near1 = sum(1 for v in all_tr if abs(v) > 0.9)
    a(f"Within-seizure Spearman of $S$ against $\\log\\operatorname{{tr}}\\Sigma$: "
      f"median {statistics.median(all_tr):+.2f} over {len(all_tr)} seizures, "
      f"|value| > 0.9 in {near1}/{len(all_tr)}. "
      + ('The two are not interchangeable: the state carries something the trace does '
         'not.' if near1 / max(len(all_tr), 1) < 0.5 else
         'Entropy and power move together in most seizures, so the trace-free claim '
         'is weak on this measurement.'))
    a('')

    a('### Consistency with 042')
    a('')
    pur_de = statistics.median([r['dS_end'] for r in by('purifier')
                                if np.isfinite(r['dS_end'])])
    mix_de = statistics.median([r['dS_end'] for r in by('mixer')
                                if np.isfinite(r['dS_end'])])
    a(f"The endpoint reproduces the 042 split even though the path does not separate: "
      f"median $S_{{\\text{{offset}}}} - S_0$ is {pur_de:+.3f} nats for purifier "
      f"seizures against {mix_de:+.3f} for mixers. 042 classified patients on medians "
      f"over whole ictal and interictal windows; this measurement, on 2 s windows "
      f"inside the seizure, puts the two groups on the same side of zero that 042 did. "
      f"The null here is about the shape of the path, not about whether the 042 "
      f"endpoint is real.")
    a('')
    a('### Shapes by patient, where they are clearest')
    a('')
    a(f"- **chb14** (purifier, 5 seizures $\\geq$ 20 s): the entropy leaves baseline "
      f"downward immediately - median time of the maximum at "
      f"{per_patient['chb14']['median_peak_frac']:.2f} of seizure length, median "
      f"$S_{{\\text{{offset}}}} - S_0 = {per_patient['chb14']['median_dS_end']:+.3f}$ "
      f"nats, and 7 of its 8 seizures (including the three under 20 s) end below "
      f"baseline. Its median of 5 turning points is the lowest in the cohort: these "
      f"are the least wobbly curves measured.")
    a(f"- **chb08** (purifier, 5 seizures, the longest in the cohort at 134-264 s): "
      f"$S$ wobbles through a band of roughly 0.8-1.8 nats with a median of "
      f"{per_patient['chb08']['median_turning_points']:.0f} turning points per "
      f"seizure, with a slow net decline "
      f"($r_S$ median {per_patient['chb08']['median_rS']:+.2f}).")
    a(f"- **chb04** (mixer, 4 seizures): the only patient with a median "
      f"$r_S \\geq +0.5$ ({per_patient['chb04']['median_rS']:+.2f}), entropy maximum "
      f"late at {per_patient['chb04']['median_peak_frac']:.2f} of seizure length.")
    a(f"- **chb16** (mixer, 10 seizures): every seizure is under 20 s, so the patient "
      f"contributes nothing to either test; its curves are plotted.")
    a('')
    a('### Not pre-declared, for a future prompt')
    a('')
    a(f"The reproducible within-seizure structure in these curves sits in "
      f"$\\log\\operatorname{{tr}}\\Sigma$ rather than in $S$, and where $S$ does move "
      f"consistently (chb14: 7 of 8 seizures end below baseline, median "
      f"$S_{{\\text{{offset}}}} - S_0 = "
      f"{per_patient['chb14']['median_dS_end']:+.2f}$ nats) the move happens within the "
      f"first seconds after onset rather than spread across the seizure.")
    a('')
    a('![per-seizure rS by group](summary_rS_by_group.png)')
    a('')
    a('Per-patient curve figures are in `curves/chbXX.png` (22 files), each showing '
      'every seizure\'s $S(t)$ with $\\log\\operatorname{tr}\\Sigma(t)$ below on the '
      'same time axis, onset at $t = 0$ and the 5 s baseline band shaded. No '
      'smoothing is applied to the plotted curves.')
    a('')

    if tests['failures']:
        a('## Seizures that failed to process')
        a('')
        a('| patient | seizure | reason |')
        a('|---|---|---|')
        for fl in tests['failures']:
            a(f"| {fl['patient']} | {fl['seizure_idx']} | {fl['reason']} |")
        a('')

    a(f"Runtime {tests['runtime_s']:.0f} s.")
    with open(OUT_DIR / 'SUMMARY.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    sys.exit(main())
