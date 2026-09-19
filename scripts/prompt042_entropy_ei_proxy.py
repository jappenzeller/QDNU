#!/usr/bin/env python3
"""
================================================================================
PROMPT 042 - The ictal entropy split and the E/I proxy
================================================================================

Part A formalises the 2026-09-06 check: per patient, does the 8-channel state
rho = Sigma / tr Sigma get more mixed or purer during seizures?
    dS_p = median_ictal S(rho) - median_interictal S(rho),  S = -sum lambda log lambda
    sign determined if Mann-Whitney p < 0.05 and bootstrap sign stability >= 0.9.

Part B tests the E/I two-level note's reading of the split. The note says
purification = mode separation (coupling weak relative to rate asymmetry),
mixing = oscillatory regime. A classical proxy for cortical E/I balance is the
aperiodic (1/f) exponent of the power spectrum: steeper (more negative) slope
= more inhibition-dominated, flatter = more excitation-dominated (Gao et al.
2017). Prediction, pre-declared:

    P1  purifiers and mixers differ in their ictal change of aperiodic slope
        (Mann-Whitney on d_slope between the two determined groups, p < 0.05),
        with the same sign of difference at both slope-fit ranges below.
    P2  (weaker) across all 22 patients, Spearman(d_slope, dS) is significant.

Slope is fit per window as the linear regression of log10 PSD on log10 f over
two ranges: 30-45 Hz (above the rhythms; the standard "aperiodic" range for
scalp EEG) and 2-45 Hz with 6-15 Hz excluded (alpha/theta peak removed).
Both ranges must agree in sign for P1 to pass. PSD: Welch, 1 s segments,
averaged over the 8 channels, on the same preprocessed W = 8 windows as 037.

Also reported (not a pass criterion): whether dS or d_slope tracks the power
change alpha_p from 038, to expose a power confound.

Outputs: results/prompt042/per_patient.csv, tests.json
Author: Claude Code
Date: 2026-09-07
================================================================================
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import mannwhitneyu, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import prompt037_per_seizure_decomposition as p37     # noqa: E402
import prompt038_agate_balance as p38                  # noqa: E402  (load_windows_like_037)

OUT = PROJECT_ROOT / "results" / "prompt042"; OUT.mkdir(parents=True, exist_ok=True)
P038 = PROJECT_ROOT / "results" / "prompt038"
W = 8.0; SEED = 42; N_BOOT = 2000
RANGES = {"hi": (30.0, 45.0), "broad_nopeak": (2.0, 45.0)}
PEAK_EXCL = (6.0, 15.0)


def log(m=""):
    print(m, flush=True)


def vn(C):
    l = np.linalg.eigvalsh(C); l = l / l.sum(); l = l[l > 1e-15]
    return float(-(l * np.log(l)).sum())


def slopes(win, fs):
    f, p = welch(win, fs=fs, nperseg=int(fs), axis=1)
    lp = np.log10(p.mean(axis=0) + 1e-30); lf = np.log10(f + 1e-12)
    out = {}
    for name, (lo, hi) in RANGES.items():
        m = (f >= lo) & (f <= hi)
        if name == "broad_nopeak":
            m &= ~((f >= PEAK_EXCL[0]) & (f <= PEAK_EXCL[1]))
        out[name] = float(np.polyfit(lf[m], lp[m], 1)[0])
    return out


def main():
    data_root = Path(sys.argv[1]) if len(sys.argv) > 1 else p37.DEFAULT_DATA_ROOT
    q1 = {r["patient"]: r for r in csv.DictReader(open(P038 / "q1_scale_shape.csv"))}
    subjects = sorted(q1)
    rng = np.random.default_rng(SEED)
    rows = []
    for s in subjects:
        z = p37.load_cache(s, W)
        win, phase, sids, widx, fs = p38.load_windows_like_037(s, data_root, W)
        mask = phase != "interictal"; mask[z["interictal_kept"]] = True
        win, phase = win[mask], phase[mask]
        assert (phase == z["phase"]).all()
        covs = z["covs"]
        S = np.array([vn(c) for c in covs])
        sl = [slopes(w, fs) for w in win]
        i, n = phase == "ictal", phase == "interictal"
        dS = float(np.median(S[i]) - np.median(S[n])); pS = float(mannwhitneyu(S[i], S[n]).pvalue)
        bs = np.array([np.median(rng.choice(S[i], i.sum())) - np.median(rng.choice(S[n], n.sum())) for _ in range(N_BOOT)])
        stab = float(np.mean(np.sign(bs) == np.sign(dS)))
        det = pS < 0.05 and stab >= 0.9
        row = dict(patient=s, n_ictal=int(i.sum()), n_interictal=int(n.sum()), S_int=float(np.median(S[n])), S_ict=float(np.median(S[i])),
                   dS=dS, p_S=pS, stab_S=stab, group=("purifier" if det and dS < 0 else ("mixer" if det and dS > 0 else "undetermined")),
                   alpha_038=float(q1[s]["alpha"]), f_scale_038=float(q1[s]["f_scale"]))
        for name in RANGES:
            a = np.array([x[name] for x in sl])
            row[f"slope_int_{name}"] = float(np.median(a[n])); row[f"slope_ict_{name}"] = float(np.median(a[i]))
            row[f"d_slope_{name}"] = float(np.median(a[i]) - np.median(a[n])); row[f"p_slope_{name}"] = float(mannwhitneyu(a[i], a[n]).pvalue)
        rows.append(row)
        log(f"  {s}: dS={dS:+.3f} (p={pS:.1e}, stab {stab:.2f}) {row['group']:12s} "
            f"slope hi {row['slope_int_hi']:+.2f}->{row['slope_ict_hi']:+.2f} (d {row['d_slope_hi']:+.2f})  "
            f"broad {row['slope_int_broad_nopeak']:+.2f}->{row['slope_ict_broad_nopeak']:+.2f} (d {row['d_slope_broad_nopeak']:+.2f})  alpha038 {row['alpha_038']:+.1f}")
    keys = list(rows[0].keys())
    with open(OUT / "per_patient.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in rows:
            w.writerow([f"{r[k]:.6g}" if isinstance(r[k], float) else r[k] for k in keys])

    pur = [r for r in rows if r["group"] == "purifier"]; mix = [r for r in rows if r["group"] == "mixer"]
    tests = dict(n_purifier=len(pur), n_mixer=len(mix), n_undetermined=len(rows) - len(pur) - len(mix),
                 purifiers=[r["patient"] for r in pur], mixers=[r["patient"] for r in mix])
    dS_all = np.array([r["dS"] for r in rows]); alpha = np.array([r["alpha_038"] for r in rows])
    for name in RANGES:
        dp = np.array([r[f"d_slope_{name}"] for r in pur]); dm = np.array([r[f"d_slope_{name}"] for r in mix])
        d_all = np.array([r[f"d_slope_{name}"] for r in rows])
        mw = float(mannwhitneyu(dp, dm).pvalue) if len(pur) >= 2 and len(mix) >= 2 else np.nan
        tests[name] = dict(median_dslope_purifiers=float(np.median(dp)) if len(dp) else None,
                           median_dslope_mixers=float(np.median(dm)) if len(dm) else None,
                           mannwhitney_p=mw, direction=("purifiers steeper" if np.median(dp) < np.median(dm) else "purifiers flatter") if len(dp) and len(dm) else None,
                           spearman_dslope_vs_dS_all22=[float(v) for v in spearmanr(d_all, dS_all)],
                           spearman_dslope_vs_alpha038=[float(v) for v in spearmanr(d_all, alpha)],
                           n_patients_slope_flattens=int((d_all > 0).sum()))
    tests["spearman_dS_vs_alpha038"] = [float(v) for v in spearmanr(dS_all, alpha)]
    same_dir = tests["hi"]["direction"] == tests["broad_nopeak"]["direction"]
    tests["P1_PASS"] = bool(same_dir and tests["hi"]["mannwhitney_p"] < 0.05 and tests["broad_nopeak"]["mannwhitney_p"] < 0.05)
    tests["P2_PASS"] = bool(tests["hi"]["spearman_dslope_vs_dS_all22"][1] < 0.05 and tests["broad_nopeak"]["spearman_dslope_vs_dS_all22"][1] < 0.05)
    json.dump(tests, open(OUT / "tests.json", "w"), indent=2)
    log(f"\ngroups: {len(pur)} purifiers {tests['purifiers']}, {len(mix)} mixers, {tests['n_undetermined']} undetermined")
    for name in RANGES:
        t = tests[name]
        log(f"  {name:13s}: d_slope purifiers {t['median_dslope_purifiers']:+.3f} vs mixers {t['median_dslope_mixers']:+.3f} -> {t['direction']}, MWU p={t['mannwhitney_p']:.3f}; "
            f"spearman(d_slope, dS) rho={t['spearman_dslope_vs_dS_all22'][0]:+.2f} p={t['spearman_dslope_vs_dS_all22'][1]:.3f}; "
            f"vs alpha038 rho={t['spearman_dslope_vs_alpha038'][0]:+.2f}; slope flattens ictally in {t['n_patients_slope_flattens']}/22")
    log(f"  dS vs alpha038 rho={tests['spearman_dS_vs_alpha038'][0]:+.2f}  P1 {'PASS' if tests['P1_PASS'] else 'FAIL'}  P2 {'PASS' if tests['P2_PASS'] else 'FAIL'}")


if __name__ == "__main__":
    main()
