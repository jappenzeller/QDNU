#!/usr/bin/env python3
"""
================================================================================
PROMPT 041 - Eye-closure predictions of the E/I two-level note on public data
================================================================================

Data: PhysioNet EEG Motor Movement/Imagery (eegmmidb), runs R01 (eyes open,
60 s) and R02 (eyes closed, 60 s), 109 subjects, 64 ch, 160 Hz. Downloaded via
mne.datasets.eegbci into data/eegmmidb/.

Channel subset: the DSP-000 "P" subset  Fz Cz Pz Oz O1 O2 P3 P4  (8-dim state).
Preprocessing: 0.5-40 Hz zero-phase Butterworth, 60 Hz notch, tiled W-second
windows (W = 4 primary, 8 replication), Ledoit-Wolf covariances (pyriemann 'lwf').

Pre-declared predictions (docs/EI_TWO_LEVEL_NOTES.md section 6), tested on the
per-subject difference closed - open (medians over windows):

  P1  trace:   log tr(Sigma) rises on eye closure.       pass: > 0 in >= 75 % of subjects, sign test p < 1e-3
  P2  entropy: S(rho) = -sum lambda log lambda falls.     pass: < 0 in >= 75 % of subjects, sign test p < 1e-3
  P3a gauge:   the absolute alpha-band phase of a channel (circular mean over a
               window) is uniform across windows in BOTH conditions.
                                                          pass: pooled resultant length < 0.10 in each condition
  P3b PLV:     phase locking of each channel to the global alpha phase rises.
                                                          pass: > 0 in >= 75 % of subjects
  Sanity:      Berger effect present: closed/open alpha power on O1 or O2 > 2 in most subjects (reported, not a pass criterion)

Outputs: results/prompt041/per_subject_W{4,8}.csv, cohort.json, SUMMARY.md (hand-written)
Author: Claude Code
Date: 2026-09-07
================================================================================
"""

from __future__ import annotations

import csv
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import mne
from scipy.signal import butter, filtfilt, iirnotch, hilbert, welch
from scipy.stats import mannwhitneyu, wilcoxon, binomtest, spearmanr
from pyriemann.estimation import Covariances

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data" / "eegmmidb"
OUT = PROJECT_ROOT / "results" / "prompt041"; OUT.mkdir(parents=True, exist_ok=True)
CH = ["Fz", "Cz", "Pz", "Oz", "O1", "O2", "P3", "P4"]
ALPHA = (8.0, 13.0)
mne.set_log_level("ERROR")


def log(m=""):
    print(m, flush=True)


def preprocess(x, fs):
    x = x - x.mean(axis=1, keepdims=True)
    b, a = butter(4, [0.5 / (fs / 2), 40.0 / (fs / 2)], btype="band"); x = filtfilt(b, a, x, axis=1)
    b, a = iirnotch(60.0, Q=30, fs=fs); x = filtfilt(b, a, x, axis=1)
    return x


def load_run(path):
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    mne.datasets.eegbci.standardize(raw)
    raw.pick(CH)
    x = raw.get_data() * 1e6                      # microvolts
    return preprocess(x, raw.info["sfreq"]), raw.info["sfreq"]


def vn_entropy(C):
    l = np.linalg.eigvalsh(C); l = l / l.sum(); l = l[l > 1e-15]
    return float(-(l * np.log(l)).sum())


def window_features(x, fs, W):
    n = int(W * fs); nw = x.shape[1] // n
    wins = np.array([x[:, i * n:(i + 1) * n] for i in range(nw)])
    covs = Covariances(estimator="lwf").transform(wins)
    logtr = np.log(np.trace(covs, axis1=1, axis2=2))
    S = np.array([vn_entropy(c) for c in covs])
    # alpha-band analytic signal per window: absolute phase (gauge) and PLV to global phase
    b, a = butter(4, [ALPHA[0] / (fs / 2), ALPHA[1] / (fs / 2)], btype="band")
    xa = filtfilt(b, a, x, axis=1); an = hilbert(xa, axis=1)
    ph = np.angle(an); env = np.abs(an)
    absphase = np.zeros((nw, len(CH))); plv = np.zeros((nw, len(CH)))
    for i in range(nw):
        sl = slice(i * n, (i + 1) * n)
        g = np.angle(np.mean(env[:, sl] * np.exp(1j * ph[:, sl]), axis=0))   # global phase, as in extract_plv_params
        for c in range(len(CH)):
            absphase[i, c] = np.angle(np.mean(np.exp(1j * ph[c, sl])))
            plv[i, c] = np.abs(np.mean(np.exp(1j * (ph[c, sl] - g))))
    return logtr, S, absphase, plv


def alpha_power(x, fs, chans=("O1", "O2")):
    out = {}
    for c in chans:
        f, p = welch(x[CH.index(c)], fs=fs, nperseg=int(2 * fs))
        out[c] = float(p[(f >= ALPHA[0]) & (f <= ALPHA[1])].mean())
    return out


def main():
    subjects = sorted(p.parent.name for p in Path(DATA).rglob("S???R01.edf"))
    log(f"{len(subjects)} subjects")
    for W in [4.0, 8.0]:
        t0 = time.time(); rows = []; abs_open, abs_closed = [], []
        for s in subjects:
            f1 = next(Path(DATA).rglob(f"{s}R01.edf")); f2 = next(Path(DATA).rglob(f"{s}R02.edf"))
            xo, fs = load_run(f1); xc, _ = load_run(f2)
            lo, So, ao, po = window_features(xo, fs, W); lc, Sc, ac, pc = window_features(xc, fs, W)
            abs_open.append(ao.ravel()); abs_closed.append(ac.ravel())
            apo, apc = alpha_power(xo, fs), alpha_power(xc, fs)
            ratio = max(apc["O1"] / apo["O1"], apc["O2"] / apo["O2"])
            rows.append(dict(subject=s, n_open=len(lo), n_closed=len(lc),
                             d_logtr=float(np.median(lc) - np.median(lo)), p_logtr=float(mannwhitneyu(lc, lo).pvalue),
                             d_S=float(np.median(Sc) - np.median(So)), p_S=float(mannwhitneyu(Sc, So).pvalue),
                             S_open=float(np.median(So)), S_closed=float(np.median(Sc)),
                             d_plv=float(np.median(pc.mean(1)) - np.median(po.mean(1))),
                             plv_open=float(np.median(po.mean(1))), plv_closed=float(np.median(pc.mean(1))),
                             alpha_ratio_best=float(ratio)))
        keys = list(rows[0].keys())
        with open(OUT / f"per_subject_W{W:g}.csv", "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(keys)
            for r in rows:
                w.writerow([f"{r[k]:.6g}" if isinstance(r[k], float) else r[k] for k in keys])
        dl = np.array([r["d_logtr"] for r in rows]); dS = np.array([r["d_S"] for r in rows]); dp = np.array([r["d_plv"] for r in rows])
        ar = np.array([r["alpha_ratio_best"] for r in rows]); n = len(rows)
        Ro = float(np.abs(np.mean(np.exp(1j * np.concatenate(abs_open))))); Rc = float(np.abs(np.mean(np.exp(1j * np.concatenate(abs_closed)))))
        n_abs = len(np.concatenate(abs_open))
        res = dict(
            W=W, n_subjects=n,
            P1_trace=dict(n_positive=int((dl > 0).sum()), frac=float((dl > 0).mean()), sign_p=float(binomtest(int((dl > 0).sum()), n, 0.5).pvalue),
                          wilcoxon_p=float(wilcoxon(dl).pvalue), median_delta=float(np.median(dl)),
                          n_sig_positive=int(((dl > 0) & (np.array([r["p_logtr"] for r in rows]) < 0.05)).sum()),
                          PASS=bool((dl > 0).mean() >= 0.75 and binomtest(int((dl > 0).sum()), n, 0.5).pvalue < 1e-3)),
            P2_entropy=dict(n_negative=int((dS < 0).sum()), frac=float((dS < 0).mean()), sign_p=float(binomtest(int((dS < 0).sum()), n, 0.5).pvalue),
                            wilcoxon_p=float(wilcoxon(dS).pvalue), median_delta=float(np.median(dS)),
                            n_sig_negative=int(((dS < 0) & (np.array([r["p_S"] for r in rows]) < 0.05)).sum()),
                            n_sig_positive=int(((dS > 0) & (np.array([r["p_S"] for r in rows]) < 0.05)).sum()),
                            PASS=bool((dS < 0).mean() >= 0.75 and binomtest(int((dS < 0).sum()), n, 0.5).pvalue < 1e-3)),
            P3a_gauge=dict(resultant_open=Ro, resultant_closed=Rc, n_channel_windows=n_abs, uniform_expectation=float(1 / np.sqrt(n_abs)),
                           PASS=bool(Ro < 0.10 and Rc < 0.10)),
            P3b_plv=dict(n_positive=int((dp > 0).sum()), frac=float((dp > 0).mean()), sign_p=float(binomtest(int((dp > 0).sum()), n, 0.5).pvalue),
                         median_delta=float(np.median(dp)), PASS=bool((dp > 0).mean() >= 0.75)),
            berger=dict(n_ratio_gt_2=int((ar > 2).sum()), median_ratio=float(np.median(ar))),
            relations=dict(spearman_dS_vs_alpha_ratio=float(spearmanr(dS, ar)[0]), spearman_dS_vs_dlogtr=float(spearmanr(dS, dl)[0]),
                           spearman_dplv_vs_alpha_ratio=float(spearmanr(dp, ar)[0]),
                           entropy_drop_frac_in_berger_subjects=float((dS[ar > 2] < 0).mean()) if (ar > 2).any() else None,
                           entropy_drop_frac_in_nonberger=float((dS[ar <= 2] < 0).mean()) if (ar <= 2).any() else None))
        json.dump(res, open(OUT / f"cohort_W{W:g}.json", "w"), indent=2)
        log(f"W={W:g}s ({time.time()-t0:.0f}s): P1 trace up {res['P1_trace']['n_positive']}/{n} ({'PASS' if res['P1_trace']['PASS'] else 'FAIL'}) med {res['P1_trace']['median_delta']:+.3f} | "
            f"P2 entropy down {res['P2_entropy']['n_negative']}/{n} ({'PASS' if res['P2_entropy']['PASS'] else 'FAIL'}) med {res['P2_entropy']['median_delta']:+.3f} "
            f"[sig down {res['P2_entropy']['n_sig_negative']}, sig up {res['P2_entropy']['n_sig_positive']}] | "
            f"P3a R_open={Ro:.3f} R_closed={Rc:.3f} ({'PASS' if res['P3a_gauge']['PASS'] else 'FAIL'}) | "
            f"P3b PLV up {res['P3b_plv']['n_positive']}/{n} ({'PASS' if res['P3b_plv']['PASS'] else 'FAIL'}) | "
            f"Berger ratio>2 {res['berger']['n_ratio_gt_2']}/{n} med {res['berger']['median_ratio']:.2f} | "
            f"dS~alpha {res['relations']['spearman_dS_vs_alpha_ratio']:+.2f}, drop-frac Berger {res['relations']['entropy_drop_frac_in_berger_subjects']:.2f} vs non {res['relations']['entropy_drop_frac_in_nonberger']}")


if __name__ == "__main__":
    main()
