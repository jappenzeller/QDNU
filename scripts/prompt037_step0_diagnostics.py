#!/usr/bin/env python3
"""
================================================================================
PROMPT 037 Step 0 - Closed-form diagnostics on the PROMPT 035 geometric addendum
================================================================================

No EDF reads, no pyriemann. Reads only:
    results/prompt035/geometric_measures.csv
    results/prompt035/clinical_metadata.csv

and tests two mechanical explanations for the addendum's flagged correlations.

DIAGNOSTIC 1 - averaging cancellation.
    If the per-seizure shift vectors carry no common direction then
    ||s_bar|| = L_bar / sqrt(K) purely mechanically, where K = n_seizures.
    Multiplying by sqrt(K) should therefore annihilate the correlation, and
    the log-log slope of ||s_p|| on K should sit near -0.5.

DIAGNOSTIC 2 - ictal-window dropout.
    The addendum loader keeps one window per segment and only if the segment
    is at least window_sec long, so patients whose seizures are shorter than
    20 s contribute few or no ictal windows and their "seizure class" is
    mostly preictal. Implied ictal retention = n_seizure_windows - n_seizures
    (each seizure supplies at most one preictal + one ictal window).

Also reports rank partial correlations, because n_seizures and
mean_seizure_duration are themselves collinear in CHB-MIT.

Exit code 0 if every assertion holds, 1 otherwise.

Usage:  python scripts/prompt037_step0_diagnostics.py
Output: results/prompt037/step0_diagnostics.csv

Author: Claude (Cowork)
Date: 2026-09-03
================================================================================
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, linregress, rankdata, t as tdist

PROJECT_ROOT = Path(__file__).resolve().parent.parent
IN_DIR = PROJECT_ROOT / "results" / "prompt035"
OUT_DIR = PROJECT_ROOT / "results" / "prompt037"

# Expected values, from the run of 2026-09-02. Assertions are to 3 decimals.
EXPECTED = {
    "s_norm~K": (-0.581, 0.0046),
    "s_norm_corr~K": (-0.524, 0.0122),
    "s_norm*sqrtK~K": (+0.108, 0.6332),
    "bw_dist~mean_dur": (+0.506, 0.0162),
    "retention~mean_dur": (+0.723, 0.0001),
    "K~mean_dur": (-0.558, 0.0069),
}
TOL = 0.0015


def read_keyed(path: Path) -> dict:
    with open(path, newline="") as f:
        return {r["patient"]: r for r in csv.DictReader(f)}


def partial_spearman(x, y, z):
    """Rank partial correlation of x,y controlling z, with a t-test p."""
    n = len(x)
    X, Y, Z = (rankdata(v) for v in (x, y, z))
    rxy = np.corrcoef(X, Y)[0, 1]
    rxz = np.corrcoef(X, Z)[0, 1]
    ryz = np.corrcoef(Y, Z)[0, 1]
    r = (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    df = n - 3
    ts = r * np.sqrt(df / (1 - r ** 2))
    return float(r), float(2 * (1 - tdist.cdf(abs(ts), df)))


def main() -> int:
    geo = read_keyed(IN_DIR / "geometric_measures.csv")
    clin = read_keyed(IN_DIR / "clinical_metadata.csv")
    pts = sorted(geo)

    K = np.array([float(clin[p]["n_seizures"]) for p in pts])
    dur = np.array([float(clin[p]["mean_seizure_duration_sec"]) for p in pts])
    s = np.array([float(geo[p]["s_norm"]) for p in pts])
    sc = np.array([float(geo[p]["s_norm_corr"]) for p in pts])
    bw = np.array([float(geo[p]["bw_dist"]) for p in pts])
    nsz_win = np.array([float(geo[p]["n_seizure_windows"]) for p in pts])
    n_win = np.array([float(geo[p]["n_windows"]) for p in pts])

    # each seizure supplies at most one preictal + one ictal window
    ictal_kept = nsz_win - K
    retention = ictal_kept / K

    rows, failures = [], []

    def record(name, rho, p, expect_key=None):
        rows.append(dict(test=name, statistic=round(float(rho), 4),
                         p_value=round(float(p), 4), n=len(pts)))
        print(f"  {name:46} {rho:+.3f}  p={p:.4f}")
        if expect_key:
            erho, ep = EXPECTED[expect_key]
            if abs(rho - erho) > TOL:
                failures.append(f"{name}: rho {rho:+.4f} != expected {erho:+.4f}")

    print(f"\nn = {len(pts)} patients, {int(K.sum())} seizures total")
    print(f"interictal window counts present: "
          f"{sorted(set((n_win - nsz_win).astype(int)))}")

    print("\nDIAGNOSTIC 1 - averaging cancellation (Spearman)")
    record("s_norm vs n_seizures", *spearmanr(K, s), "s_norm~K")
    record("s_norm_corr vs n_seizures", *spearmanr(K, sc), "s_norm_corr~K")
    record("s_norm*sqrt(K) vs n_seizures", *spearmanr(K, s * np.sqrt(K)),
           "s_norm*sqrtK~K")
    record("s_norm*sqrt(K) vs mean_seizure_duration",
           *spearmanr(dur, s * np.sqrt(K)))
    record("bw_dist*sqrt(K) vs mean_seizure_duration",
           *spearmanr(dur, bw * np.sqrt(K)))

    print("\n  log-log slopes on log(K)  [pure cancellation predicts -0.5]")
    for name, y in [("s_norm", s), ("s_norm_corr", sc), ("bw_dist", bw)]:
        lr = linregress(np.log(K), np.log(y))
        lo, hi = lr.slope - 1.96 * lr.stderr, lr.slope + 1.96 * lr.stderr
        rows.append(dict(test=f"loglog_slope[{name}]",
                         statistic=round(float(lr.slope), 4),
                         p_value=round(float(lr.pvalue), 4), n=len(pts)))
        holds_cancel = lo <= -0.5 <= hi
        holds_null = lo <= 0.0 <= hi
        if holds_cancel and not holds_null:
            verdict = "consistent with cancellation, excludes 0"
        elif holds_cancel and holds_null:
            verdict = "uninformative: CI spans both -0.5 and 0"
        elif holds_null:
            verdict = "consistent with no K-dependence"
        else:
            verdict = "CI excludes both -0.5 and 0"
        print(f"    {name:14} slope={lr.slope:+.3f}  "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]  R2={lr.rvalue ** 2:.3f}  "
              f"-> {verdict}")

    print("\nDIAGNOSTIC 2 - ictal-window dropout")
    print(f"    {'patient':9}{'K':>4}{'szwin':>7}{'ictal':>7}{'frac':>7}"
          f"{'meandur':>9}{'bw':>9}")
    for i, p in enumerate(pts):
        mark = "  <-- degraded" if retention[i] < 0.9 else ""
        print(f"    {p:9}{K[i]:>4.0f}{nsz_win[i]:>7.0f}{ictal_kept[i]:>7.0f}"
              f"{retention[i]:>7.2f}{dur[i]:>9.1f}{bw[i]:>9.1f}{mark}")
    print()
    record("ictal_retention vs mean_seizure_duration",
           *spearmanr(retention, dur), "retention~mean_dur")
    record("ictal_retention vs max_seizure_duration",
           *spearmanr(retention,
                      [float(clin[p]["max_seizure_duration_sec"]) for p in pts]))
    record("ictal_retention vs bw_dist", *spearmanr(retention, bw))
    record("ictal_retention vs s_norm", *spearmanr(retention, s))

    print("\nCOLLINEARITY AND PARTIALS")
    record("n_seizures vs mean_seizure_duration", *spearmanr(K, dur),
           "K~mean_dur")
    record("bw_dist vs mean_seizure_duration", *spearmanr(dur, bw),
           "bw_dist~mean_dur")
    for nm, a, b, c in [
        ("bw_dist vs mean_dur | n_seizures", bw, dur, K),
        ("bw_dist vs n_seizures | mean_dur", bw, K, dur),
        ("s_norm vs n_seizures | mean_dur", s, K, dur),
        ("s_norm vs mean_dur | n_seizures", s, dur, K),
    ]:
        record(f"partial: {nm}", *partial_spearman(a, b, c))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "step0_diagnostics.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["test", "statistic", "p_value", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {out}")

    if failures:
        print("\nASSERTION FAILURES - do not proceed to Step 1:")
        for msg in failures:
            print("  " + msg)
        return 1
    print(f"\nAll {len(EXPECTED)} anchored values reproduce within {TOL}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
