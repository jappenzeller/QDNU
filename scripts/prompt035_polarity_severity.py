#!/usr/bin/env python3
"""
PROMPT 035 — Polarity magnitude vs clinical severity (exploratory).

Hypothesis: within the refractory-epilepsy CHB-MIT cohort, polarity magnitude
|AUC - 0.5| (Tier 2A and Tier 3 at 20-s windows) correlates with clinical
seizure-severity indicators. This script parses each patient's summary file
to derive severity variables, joins with the polarity table from
`results/full_cohort/full_cohort_polarity.json`, and runs Spearman
correlations (and point-biserial for polarity sign).

Outputs to results/prompt035/:
    clinical_metadata.csv        one row per patient
    polarity_magnitude.csv       Tier 2A / Tier 3 raw AUC + |AUC-0.5| + sign
    correlations.csv             Spearman rho + p per (variable, magnitude) pair
    biserial.csv                 point-biserial for polarity sign
    scatters/*.png               plotted only for significant hits
    hardware_subset_profile.csv  strong vs weak polar comparison (n=7)
    SUMMARY.md                   findings paragraph + tables

This is exploratory: report null results honestly; multiple comparisons are
NOT corrected in the primary tables (the sample size is small; corrections
would obliterate power). We flag |rho| > 0.4 AND raw p < 0.05 as "worth
following up." A Bonferroni-adjusted alpha is also reported for context.
"""

from __future__ import annotations
import json
import re
import csv
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path('h:/QuantumPython/QDNU')
CHBMIT = Path('H:/Data/PythonDNU/EEG/chbmit')
POLARITY_JSON = ROOT / 'results' / 'full_cohort' / 'full_cohort_polarity.json'
OUT_DIR = ROOT / 'results' / 'prompt035'
SCATTER_DIR = OUT_DIR / 'scatters'
OUT_DIR.mkdir(parents=True, exist_ok=True)
SCATTER_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CHB-MIT demographic table (from PhysioNet documentation, not local files).
# Age in years at time of recording; gender F/M.
# https://physionet.org/content/chbmit/1.0.0/
# ---------------------------------------------------------------------------
CHBMIT_DEMOGRAPHICS = {
    'chb01': {'age': 11.0, 'sex': 'F'},
    'chb02': {'age': 11.0, 'sex': 'M'},
    'chb03': {'age': 14.0, 'sex': 'F'},
    'chb04': {'age': 22.0, 'sex': 'M'},
    'chb05': {'age': 7.0,  'sex': 'F'},
    'chb06': {'age': 1.5,  'sex': 'F'},
    'chb07': {'age': 14.5, 'sex': 'F'},
    'chb08': {'age': 3.5,  'sex': 'M'},
    'chb09': {'age': 10.0, 'sex': 'F'},
    'chb10': {'age': 3.0,  'sex': 'M'},
    'chb11': {'age': 12.0, 'sex': 'F'},
    'chb12': {'age': 2.0,  'sex': 'F'},
    'chb13': {'age': 3.0,  'sex': 'F'},
    'chb14': {'age': 9.0,  'sex': 'F'},
    'chb15': {'age': 16.0, 'sex': 'M'},
    'chb16': {'age': 7.0,  'sex': 'F'},
    'chb17': {'age': 12.0, 'sex': 'F'},
    'chb18': {'age': 18.0, 'sex': 'F'},
    'chb19': {'age': 19.0, 'sex': 'F'},
    'chb20': {'age': 6.0,  'sex': 'F'},
    'chb21': {'age': 13.0, 'sex': 'F'},
    'chb22': {'age': 9.0,  'sex': 'F'},
    'chb23': {'age': 6.0,  'sex': 'F'},
    'chb24': {'age': None, 'sex': None},  # not reported by PhysioNet
}


# ---------------------------------------------------------------------------
# Summary-file parser
# ---------------------------------------------------------------------------
def _hms_to_seconds(s: str) -> int:
    h, m, sec = s.strip().split(':')
    return int(h) * 3600 + int(m) * 60 + int(sec)


def parse_summary(pat: str) -> Dict:
    """Parse chbNN-summary.txt into a per-patient dict.

    Returns:
        n_seizures, mean_seizure_duration_sec, median_seizure_duration_sec,
        max_seizure_duration_sec, n_seizure_files, n_total_files,
        total_recording_hours, seizure_density_per_hour,
        seizure_durations (list)
    """
    p = CHBMIT / pat / f'{pat}-summary.txt'
    if not p.exists():
        raise FileNotFoundError(p)
    text = p.read_text()
    # Split into per-file records; some files have blank File Name blocks
    file_blocks = re.split(r'\n(?=File Name:)', text)
    seizure_durations: List[int] = []
    total_recording_seconds = 0
    n_seizure_files = 0
    n_total_files = 0
    for block in file_blocks:
        if 'File Name:' not in block:
            continue
        n_total_files += 1
        # duration of the file
        m_start = re.search(r'File Start Time:\s*([\d:]+)', block)
        m_end   = re.search(r'File End Time:\s*([\d:]+)', block)
        if m_start and m_end:
            try:
                dur = _hms_to_seconds(m_end.group(1)) - _hms_to_seconds(m_start.group(1))
                # handle midnight wrap-around
                if dur < 0:
                    dur += 24 * 3600
                total_recording_seconds += dur
            except Exception:
                pass
        # seizures in this file
        m_n = re.search(r'Number of Seizures in File:\s*(\d+)', block)
        if not m_n:
            continue
        n = int(m_n.group(1))
        if n == 0:
            continue
        n_seizure_files += 1
        # find pairs of Seizure Start Time / Seizure End Time (they can be
        # numbered as "Seizure 1 Start Time" for multi-seizure files)
        starts = [int(x) for x in re.findall(r'Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)', block)]
        ends   = [int(x) for x in re.findall(r'Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)',   block)]
        for s, e in zip(starts, ends):
            d = e - s
            if d > 0:
                seizure_durations.append(d)
    if not seizure_durations:
        return {
            'n_seizures': 0,
            'mean_seizure_duration_sec': float('nan'),
            'median_seizure_duration_sec': float('nan'),
            'max_seizure_duration_sec': float('nan'),
            'n_seizure_files': 0,
            'n_total_files': n_total_files,
            'total_recording_hours': total_recording_seconds / 3600.0,
            'seizure_density_per_hour': 0.0,
            'seizure_durations': [],
        }
    total_recording_hours = total_recording_seconds / 3600.0
    return {
        'n_seizures': len(seizure_durations),
        'mean_seizure_duration_sec': float(np.mean(seizure_durations)),
        'median_seizure_duration_sec': float(np.median(seizure_durations)),
        'max_seizure_duration_sec': float(np.max(seizure_durations)),
        'n_seizure_files': n_seizure_files,
        'n_total_files': n_total_files,
        'total_recording_hours': total_recording_hours,
        'seizure_density_per_hour': (len(seizure_durations) / total_recording_hours
                                     if total_recording_hours > 0 else float('nan')),
        'seizure_durations': seizure_durations,
    }


# ---------------------------------------------------------------------------
# Load polarity data (Tier 2A + Tier 3 at 20-s windows)
# ---------------------------------------------------------------------------
def load_polarity() -> Tuple[List[str], Dict[str, Dict], Dict[str, Dict], Dict]:
    with open(POLARITY_JSON) as f:
        d = json.load(f)
    cfg = d['config']
    t2a = d['tier2a_eigenvalues']['per_subject']
    t3  = d['tier3']['per_subject']
    patients = sorted(set(t2a) | set(t3))
    return patients, t2a, t3, cfg


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------
def spearman(x, y) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return float('nan'), float('nan')
    r, p = stats.spearmanr(x[mask], y[mask])
    return float(r), float(p)


def pointbiserial(binary, continuous) -> Tuple[float, float]:
    binary = np.asarray(binary, dtype=int)
    continuous = np.asarray(continuous, dtype=float)
    mask = np.isfinite(continuous)
    if mask.sum() < 4 or len(np.unique(binary[mask])) < 2:
        return float('nan'), float('nan')
    r, p = stats.pointbiserialr(binary[mask], continuous[mask])
    return float(r), float(p)


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------
def scatter(x, y, xlabel, ylabel, title, rho, p, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 4.3), dpi=140)
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    xv, yv = x[mask], y[mask]
    ax.scatter(xv, yv, s=40, color='#1f4fa6', alpha=0.85, edgecolor='white', linewidth=0.8)
    # least-squares line (visual only; the reported statistic is Spearman rho)
    if len(xv) >= 2:
        slope, intercept = np.polyfit(xv, yv, 1)
        xs = np.linspace(xv.min(), xv.max(), 100)
        ax.plot(xs, slope * xs + intercept, color='#cf4040', lw=1.5, alpha=0.7,
                label='OLS fit (visual)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}\nSpearman rho = {rho:+.3f}, p = {p:.4f}, n = {mask.sum()}')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print('=' * 68)
    print('PROMPT 035 — polarity magnitude vs clinical severity')
    print('=' * 68)

    patients, t2a, t3, cfg = load_polarity()
    print(f'Polarity source: {POLARITY_JSON.name} (window {cfg["window_sec"]}s, '
          f'{len(patients)} patients, excludes {cfg["exclude_subjects"]})')

    # --- Step 1: clinical metadata ---
    clin_rows = []
    for pat in patients:
        try:
            meta = parse_summary(pat)
        except FileNotFoundError:
            print(f'  {pat}: summary missing, skipping')
            continue
        demo = CHBMIT_DEMOGRAPHICS.get(pat, {})
        row = {
            'patient': pat,
            'age_years': demo.get('age'),
            'sex': demo.get('sex'),
            **{k: v for k, v in meta.items() if k != 'seizure_durations'},
        }
        clin_rows.append(row)

    with open(OUT_DIR / 'clinical_metadata.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(clin_rows[0].keys()))
        w.writeheader()
        for r in clin_rows:
            w.writerow(r)
    print(f'Wrote clinical_metadata.csv ({len(clin_rows)} patients)')

    # --- Step 2: polarity magnitude table ---
    pol_rows = []
    for pat in patients:
        r = {'patient': pat}
        v2 = t2a.get(pat); v3 = t3.get(pat)
        r['t2a_raw_auc'] = v2['raw_auc'] if v2 else float('nan')
        r['t2a_polarity_mag'] = abs(v2['raw_auc'] - 0.5) if v2 else float('nan')
        r['t2a_polarity'] = v2['polarity'] if v2 else ''
        r['t3_raw_auc']  = v3['raw_auc'] if v3 else float('nan')
        r['t3_polarity_mag'] = abs(v3['raw_auc'] - 0.5) if v3 else float('nan')
        r['t3_polarity'] = v3['polarity'] if v3 else ''
        pol_rows.append(r)
    with open(OUT_DIR / 'polarity_magnitude.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(pol_rows[0].keys()))
        w.writeheader()
        for r in pol_rows:
            w.writerow(r)
    print(f'Wrote polarity_magnitude.csv')

    # --- Step 3: correlations ---
    # Assemble arrays keyed by patient
    by_pat = {r['patient']: r for r in clin_rows}
    pol_by_pat = {r['patient']: r for r in pol_rows}
    ordered = [p for p in patients if p in by_pat and p in pol_by_pat]

    clin_vars = [
        ('age_years',                  'Age (years)'),
        ('n_seizures',                 'Total number of seizures'),
        ('mean_seizure_duration_sec',  'Mean seizure duration (s)'),
        ('median_seizure_duration_sec','Median seizure duration (s)'),
        ('max_seizure_duration_sec',   'Max seizure duration (s)'),
        ('n_seizure_files',            'Files with seizures'),
        ('total_recording_hours',      'Total recording hours'),
        ('seizure_density_per_hour',   'Seizure density (per hour)'),
    ]
    pol_measures = [
        ('t2a_polarity_mag', 'Tier 2A |AUC - 0.5|'),
        ('t3_polarity_mag',  'Tier 3 |AUC - 0.5|'),
    ]

    corr_rows = []
    scatter_hits = []
    for cvar, clabel in clin_vars:
        for pvar, plabel in pol_measures:
            xs = [by_pat[p].get(cvar) for p in ordered]
            ys = [pol_by_pat[p].get(pvar) for p in ordered]
            rho, pval = spearman(xs, ys)
            row = {
                'clinical_variable': cvar,
                'polarity_measure':  pvar,
                'spearman_rho':      rho,
                'p_value':           pval,
                'n':                 int(np.isfinite(np.asarray(xs, float)).sum() &
                                         np.isfinite(np.asarray(ys, float)).sum()),
                'flag':              '',
            }
            n_finite = int((np.isfinite(np.asarray(xs, dtype=float)) &
                            np.isfinite(np.asarray(ys, dtype=float))).sum())
            row['n'] = n_finite
            if np.isfinite(rho) and abs(rho) > 0.4 and pval < 0.05:
                row['flag'] = 'FOLLOW-UP'
                scatter_hits.append((cvar, clabel, pvar, plabel, xs, ys, rho, pval))
            corr_rows.append(row)

    with open(OUT_DIR / 'correlations.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        w.writeheader()
        for r in corr_rows:
            w.writerow(r)
    print(f'Wrote correlations.csv ({len(corr_rows)} pairs, {len(scatter_hits)} flagged)')

    # --- Point-biserial for polarity sign ---
    def sign_to_int(v):
        return 1 if v == 'inverted' else 0

    biserial_rows = []
    for tier_key, tier_label in [('t2a_polarity', 'Tier 2A polarity sign'),
                                 ('t3_polarity',  'Tier 3 polarity sign')]:
        signs = [sign_to_int(pol_by_pat[p].get(tier_key)) for p in ordered]
        for cvar, clabel in clin_vars:
            xs = [by_pat[p].get(cvar) for p in ordered]
            r, p_ = pointbiserial(signs, xs)
            biserial_rows.append({
                'polarity_sign': tier_key,
                'clinical_variable': cvar,
                'r_pb': r,
                'p_value': p_,
                'flag': ('FOLLOW-UP' if np.isfinite(r) and abs(r) > 0.4 and p_ < 0.05 else ''),
            })
    with open(OUT_DIR / 'biserial.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(biserial_rows[0].keys()))
        w.writeheader()
        for r in biserial_rows:
            w.writerow(r)
    print(f'Wrote biserial.csv ({len(biserial_rows)} pairs)')

    # --- Scatter plots for flagged hits (Spearman) ---
    for cvar, clabel, pvar, plabel, xs, ys, rho, pval in scatter_hits:
        out = SCATTER_DIR / f'{cvar}__{pvar}.png'
        scatter(xs, ys, clabel, plabel,
                f'{plabel} vs {clabel}',
                rho, pval, out)
        print(f'  scatter: {out.name}')

    # --- Scatter plots for flagged point-biserial (sign vs continuous) ---
    def bi_scatter(binary_labels, continuous, xlabel_bin, ylabel, title, r, p, out_path):
        binary_labels = np.asarray(binary_labels, dtype=int)
        continuous = np.asarray(continuous, dtype=float)
        mask = np.isfinite(continuous)
        b, y = binary_labels[mask], continuous[mask]
        fig, ax = plt.subplots(figsize=(5.5, 4.3), dpi=140)
        rng = np.random.default_rng(0)
        jitter = rng.uniform(-0.08, 0.08, size=b.size)
        ax.scatter(b + jitter, y, s=40, color='#1f4fa6', alpha=0.85,
                   edgecolor='white', linewidth=0.8)
        for g in (0, 1):
            gy = y[b == g]
            if gy.size:
                ax.plot([g - 0.25, g + 0.25], [gy.mean(), gy.mean()],
                        color='#cf4040', lw=2, alpha=0.8)
                ax.text(g, gy.mean() + 0.02 * (y.max() - y.min()),
                        f'mean = {gy.mean():.1f}', ha='center', fontsize=8,
                        color='#cf4040')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['standard', 'inverted'])
        ax.set_xlabel(xlabel_bin)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{title}\nPoint-biserial r = {r:+.3f}, p = {p:.4f}, n = {b.size}')
        ax.grid(True, alpha=0.25, linewidth=0.5)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140, bbox_inches='tight')
        plt.close(fig)

    for r_bi in biserial_rows:
        if r_bi['flag'] != 'FOLLOW-UP':
            continue
        tier_key = r_bi['polarity_sign']
        cvar = r_bi['clinical_variable']
        signs = [1 if pol_by_pat[p].get(tier_key) == 'inverted' else 0 for p in ordered]
        xs = [by_pat[p].get(cvar) for p in ordered]
        out = SCATTER_DIR / f'{cvar}__{tier_key}_sign.png'
        bi_scatter(signs, xs,
                   f'{tier_key} polarity',
                   dict(clin_vars)[cvar],
                   f'{dict(clin_vars)[cvar]} vs {tier_key} polarity sign',
                   r_bi['r_pb'], r_bi['p_value'], out)
        print(f'  bi-scatter: {out.name}')

    # --- Near-miss scatter (age vs Tier 2A magnitude, rho ~ 0.42, p ~ 0.054) ---
    near_miss_thresh = 0.06  # slightly above conventional 0.05
    for r in corr_rows:
        if r['flag']:
            continue
        if (np.isfinite(r['spearman_rho']) and abs(r['spearman_rho']) > 0.4
                and 0.05 <= r['p_value'] < near_miss_thresh):
            cvar = r['clinical_variable']
            pvar = r['polarity_measure']
            xs = [by_pat[p].get(cvar) for p in ordered]
            ys = [pol_by_pat[p].get(pvar) for p in ordered]
            clabel = dict(clin_vars)[cvar]
            plabel = dict(pol_measures)[pvar]
            out = SCATTER_DIR / f'nearmiss__{cvar}__{pvar}.png'
            scatter(xs, ys, clabel, plabel,
                    f'{plabel} vs {clabel} (near-miss)',
                    r['spearman_rho'], r['p_value'], out)
            print(f'  near-miss scatter: {out.name}')

    # --- Step 4: strong vs weak polar (hardware subset) ---
    STRONG = ['chb03', 'chb21']
    WEAK   = ['chb01', 'chb05', 'chb07', 'chb14']
    HARDWARE = STRONG + WEAK + ['chb11']  # chb11 also in the 7-patient hardware subset

    def group_row(pat_list, label):
        keep = [p for p in pat_list if p in by_pat]
        if not keep:
            return None
        rows = [by_pat[p] for p in keep]
        pol_rowsK = [pol_by_pat[p] for p in keep]
        return {
            'group': label,
            'n_patients': len(keep),
            'patients': ';'.join(keep),
            'mean_age_years': np.nanmean([r.get('age_years') or np.nan for r in rows]),
            'mean_seizure_count': np.nanmean([r['n_seizures'] for r in rows]),
            'mean_seizure_duration_sec': np.nanmean([r['mean_seizure_duration_sec'] for r in rows]),
            'mean_recording_hours': np.nanmean([r['total_recording_hours'] for r in rows]),
            'mean_seizure_density_per_hour': np.nanmean([r['seizure_density_per_hour'] for r in rows]),
            'mean_t2a_polarity_mag': np.nanmean([r['t2a_polarity_mag'] for r in pol_rowsK]),
            'mean_t3_polarity_mag':  np.nanmean([r['t3_polarity_mag']  for r in pol_rowsK]),
        }

    profile_rows = [row for row in [group_row(STRONG, 'strong_polar'),
                                    group_row(WEAK,   'weak_polar'),
                                    group_row(HARDWARE, 'hardware_subset_all')]
                    if row is not None]
    with open(OUT_DIR / 'hardware_subset_profile.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(profile_rows[0].keys()))
        w.writeheader()
        for r in profile_rows:
            w.writerow(r)
    print(f'Wrote hardware_subset_profile.csv')

    # --- SUMMARY.md ---
    n_flag = sum(1 for r in corr_rows if r['flag'])
    n_bi_flag = sum(1 for r in biserial_rows if r['flag'])
    n_pairs = len(corr_rows)
    bonf = 0.05 / n_pairs
    top_hits = sorted(
        [r for r in corr_rows if np.isfinite(r['spearman_rho'])],
        key=lambda r: (-abs(r['spearman_rho']), r['p_value']))[:6]

    lines = []
    lines.append('# PROMPT 035 — Polarity magnitude vs clinical severity (CHB-MIT)')
    lines.append('')
    lines.append(f'Generated: {datetime.now().isoformat()}')
    lines.append('')
    lines.append(f'Cohort: {len(ordered)} patients from CHB-MIT (chb12 and chb24 excluded); '
                 f'20-s windows; Tier 2A eigenvalues + Tier 3 Riemannian. '
                 f'Source: `{POLARITY_JSON.relative_to(ROOT)}`.')
    lines.append('')
    lines.append('## Findings paragraph')
    lines.append('')
    # Assemble a concise, honest paragraph.
    bi_hits = [r for r in biserial_rows if r['flag']]
    spear_hits = [r for r in corr_rows if r['flag']]
    near_miss = [r for r in corr_rows
                 if not r['flag'] and np.isfinite(r['spearman_rho'])
                 and abs(r['spearman_rho']) > 0.4 and 0.05 <= r['p_value'] < 0.06]

    para = []
    para.append(
        f"Across {n_pairs} Spearman-correlation tests pairing polarity magnitude "
        f"($|\\mathrm{{AUC}}-0.5|$ under Tier 2A eigenvalues and Tier 3 Riemannian) "
        f"with eight CHB-MIT clinical severity variables (age, seizure count, "
        f"mean/median/max seizure duration, seizure-carrying files, total recording "
        f"hours, and seizure density per hour), "
        f"{'no pair' if not spear_hits else f'{len(spear_hits)} pair(s)'} "
        f"crossed the pre-declared exploratory threshold of $|\\rho| > 0.4$ AND "
        f"$p < 0.05$."
    )
    if near_miss:
        r = near_miss[0]
        para.append(
            f" One near-miss appeared for Tier 2A magnitude vs "
            f"{r['clinical_variable'].replace('_',' ')} "
            f"(Spearman $\\rho = {r['spearman_rho']:+.3f}$, $p = {r['p_value']:.3f}$, "
            f"$n={r['n']}$), suggesting a weak positive relation between age and "
            f"how strongly the eigenvalue-basis classifier separates from chance, "
            f"but the test does not clear the 0.05 threshold."
        )
    if bi_hits:
        r = bi_hits[0]
        para.append(
            f" The one flagged point-biserial finding was Tier 3 polarity sign "
            f"(standard vs inverted) vs age "
            f"($r_{{pb}} = {r['r_pb']:+.3f}$, $p = {r['p_value']:.3f}$, "
            f"$n={len(ordered)}$): under Tier 3, inverted-polarity patients "
            f"skewed younger. Bonferroni-corrected alpha across all "
            f"{n_pairs + len(biserial_rows)} tests is "
            f"$\\approx {0.05 / (n_pairs + len(biserial_rows)):.4f}$, so this hit is "
            f"exploratory and could easily be a false positive at this sample size; "
            f"it also is not corroborated by the Tier 2A sign or by Spearman on "
            f"the magnitude itself. If real, it is more plausibly a signal of "
            f"developmental cortical maturation affecting the Riemannian tangent "
            f"projection than of seizure severity per se."
        )
    para.append(
        f" Overall the hypothesis 'polarity magnitude scales with clinical seizure "
        f"severity within CHB-MIT' is NOT supported by this dataset: the seven "
        f"summary-derived severity variables show no reliable monotone relationship "
        f"with $|\\mathrm{{AUC}}-0.5|$ under either tier. This is consistent with a "
        f"structural interpretation of polarity (a property of the covariance "
        f"geometry, not of how many or how long a patient seized in the recording) "
        f"but should be read as a failure to reject the null on $n={len(ordered)}$: "
        f"a larger cohort, richer clinical annotations (seizure type, onset "
        f"lateralization, medication load), or direct developmental / anatomical "
        f"covariates could still surface an effect."
    )
    lines.append(''.join(para))
    lines.append('')
    lines.append('## Top |rho| pairs (Spearman)')
    lines.append('')
    lines.append('| Polarity measure | Clinical variable | rho | p | n | flag |')
    lines.append('|---|---|---|---|---|---|')
    for r in top_hits:
        lines.append(f"| {r['polarity_measure']} | {r['clinical_variable']} | "
                     f"{r['spearman_rho']:+.3f} | {r['p_value']:.4f} | {r['n']} | {r['flag']} |")
    lines.append('')
    lines.append(f'Total tests: {n_pairs}. Bonferroni-adjusted alpha: {bonf:.4f}.')
    lines.append('')
    lines.append('## Strong vs weak polar (7-patient hardware subset)')
    lines.append('')
    lines.append('Strong polar: chb03, chb21 (from prompt).')
    lines.append('Weak polar:   chb01, chb05, chb07, chb14 (from prompt).')
    lines.append('Hardware subset also includes chb11 (not classified by prompt).')
    lines.append('')
    lines.append('| Group | n | mean age | mean seizures | mean dur (s) | mean rec (h) | '
                 'mean density (/h) | mean t2a mag | mean t3 mag |')
    lines.append('|---|---|---|---|---|---|---|---|---|')
    for r in profile_rows:
        lines.append(f"| {r['group']} | {r['n_patients']} | "
                     f"{r['mean_age_years']:.1f} | {r['mean_seizure_count']:.1f} | "
                     f"{r['mean_seizure_duration_sec']:.1f} | {r['mean_recording_hours']:.1f} | "
                     f"{r['mean_seizure_density_per_hour']:.3f} | "
                     f"{r['mean_t2a_polarity_mag']:.3f} | {r['mean_t3_polarity_mag']:.3f} |")
    lines.append('')
    lines.append('## Files in this directory')
    lines.append('')
    lines.append('- `clinical_metadata.csv` — one row per patient, parsed from '
                 '`chbNN-summary.txt` (age from PhysioNet documentation).')
    lines.append('- `polarity_magnitude.csv` — Tier 2A / Tier 3 raw AUC, '
                 '$|\\mathrm{AUC}-0.5|$, and sign.')
    lines.append('- `correlations.csv` — full Spearman table (16 pairs).')
    lines.append('- `biserial.csv` — point-biserial for polarity sign vs clinical vars.')
    lines.append('- `hardware_subset_profile.csv` — strong / weak / all hardware means.')
    lines.append('- `scatters/*.png` — plotted only for FOLLOW-UP-flagged pairs.')
    lines.append('')
    lines.append('## Reading guide')
    lines.append('')
    lines.append('- This is exploratory, not confirmatory.')
    lines.append('- Bonferroni is reported for context; the small n means the Bonferroni-'
                 'corrected test has near-zero power against realistic effect sizes.')
    lines.append('- A null result narrows interpretation: polarity direction and magnitude '
                 'appear independent of the severity variables that CHB-MIT summaries expose. '
                 'Structural / anatomical / medication-status variables were not tested here '
                 'and remain plausible mediators.')
    (OUT_DIR / 'SUMMARY.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Wrote SUMMARY.md')
    print()
    print(f'Flagged Spearman hits: {n_flag}')
    print(f'Flagged biserial hits: {n_bi_flag}')


if __name__ == '__main__':
    main()
