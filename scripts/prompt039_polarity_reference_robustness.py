#!/usr/bin/env python3
"""
================================================================================
PROMPT 039 - Is per-patient polarity robust to the reference it is measured
against?
================================================================================

Polarity (Paper 2) is the sign of AUC - 0.5 when a direction learned from the
OTHER patients is applied to a held-out patient's windows. PROMPT 038 showed
the sign is not carried by the scale axis nor by the cohort's leading shape
axis, so it is a property of the pair (patient, reference direction). This
prompt asks how much of it is the patient and how much is the reference.

For each held-out patient p and each window length W in {8, 20}:
  1. Full LOSO: tangent space at the Frechet mean of the other 21 patients'
     covariances; direction w from shrinkage LDA on their tangent vectors
     (ictal = 1, interictal = 0); score = w . x on p's windows; AUC_p.
  2. Reference resampling: N_DRAWS random subsets of n_sub of the 21 training
     patients; recompute reference + w + AUC_p each time. Stability =
     fraction of draws whose sign(AUC - 0.5) matches the full-LOSO sign.
  3. Same with the un-whitened mean-difference direction (ictal mean minus
     interictal mean in the training tangent space) as a control for "is the
     sign a whitening artifact".

Pre-declared reading (per patient):
  robust     : |AUC_full - 0.5| >= 0.1 at W=8, stability >= 0.90 at W=8, and
               the same sign at W=20 (full LOSO)
  reference-dependent : |AUC_full - 0.5| >= 0.1 at W=8 but stability < 0.90
                        or opposite sign at W=20
  orthogonal : |AUC_full - 0.5| < 0.1 at W=8

Outputs: results/prompt039/per_patient_W{8,20}.csv, robustness.json, SUMMARY.md
(SUMMARY written by hand from the CSVs).

Author: Claude Code
Date: 2026-09-06
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from pyriemann.tangentspace import TangentSpace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import prompt037_per_seizure_decomposition as p37   # noqa: E402

OUT = PROJECT_ROOT / "results" / "prompt039"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 39
N_DRAWS = 60
N_SUB = 15


def log(m=""):
    print(m, flush=True)


def load_all(subjects, W, seizure_class):
    """seizure_class 'ictal' (strict) or 'ictal+preictal' (Paper 2 / 035 class definition)."""
    pos = ["ictal"] if seizure_class == "ictal" else ["ictal", "preictal"]
    data = {}
    for s in subjects:
        z = p37.load_cache(s, W)
        m = np.isin(z["phase"], pos + ["interictal"])
        data[s] = (z["covs"][m], np.isin(z["phase"][m], pos).astype(int))
    return data


def fit_direction(train, method):
    covs = np.concatenate([train[s][0] for s in train]); y = np.concatenate([train[s][1] for s in train])
    ts = TangentSpace(metric="riemann").fit(covs)
    X = ts.transform(covs)
    if method == "lda":
        lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X, y)
        w = lda.coef_[0]
    else:
        w = X[y == 1].mean(0) - X[y == 0].mean(0)
    return ts, w / np.linalg.norm(w)


def auc_for(ts, w, covs, y):
    return float(roc_auc_score(y, ts.transform(covs) @ w))


def run_patient(p, data, W, rng):
    if len(np.unique(data[p][1])) < 2:
        return None                      # no ictal windows at this W (chb16 at W=20)
    others = [s for s in data if s != p and len(np.unique(data[s][1])) == 2]
    train = {s: data[s] for s in others}
    out = {}
    for method in ["lda", "meandiff"]:
        ts, w = fit_direction(train, method)
        auc_full = auc_for(ts, w, *data[p])
        sign_full = np.sign(auc_full - 0.5)
        aucs = []
        for _ in range(N_DRAWS):
            sub = rng.choice(others, N_SUB, replace=False)
            ts_b, w_b = fit_direction({s: data[s] for s in sub}, method)
            aucs.append(auc_for(ts_b, w_b, *data[p]))
        aucs = np.array(aucs)
        out[method] = dict(auc_full=auc_full, sign_full=int(sign_full),
                           stability=float(np.mean(np.sign(aucs - 0.5) == sign_full)),
                           auc_draw_min=float(aucs.min()), auc_draw_median=float(np.median(aucs)),
                           auc_draw_max=float(aucs.max()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="")
    ap.add_argument("--windows", default="8,20")
    ap.add_argument("--seizure-class", default="ictal", choices=["ictal", "ictal+preictal"])
    args = ap.parse_args()
    tag = "" if args.seizure_class == "ictal" else "_pre"
    all_subjects = sorted(d.name for d in (p37.OUT_DIR / "cache" / "W8").glob("chb*.npz"))
    all_subjects = [s.replace(".npz", "") for s in all_subjects]
    todo = args.subjects.split(",") if args.subjects else all_subjects
    for W in [float(w) for w in args.windows.split(",")]:
        data = load_all(all_subjects, W, args.seizure_class)
        for p in todo:
            f = OUT / f"patient_{p}_W{W:g}{tag}.json"
            if f.exists():
                continue
            t0 = time.time()
            rng = np.random.default_rng(SEED + int(p[3:]) + int(W))
            res = run_patient(p, data, W, rng)
            if res is None:
                log(f"  W={W:g} {p}: skipped (single class)"); continue
            json.dump(res, open(f, "w"), indent=2)
            r = res["lda"]; m = res["meandiff"]
            log(f"  W={W:g} {p}: LDA auc={r['auc_full']:.3f} stab={r['stability']:.2f} "
                f"[{r['auc_draw_min']:.2f},{r['auc_draw_max']:.2f}] | meandiff auc={m['auc_full']:.3f} "
                f"stab={m['stability']:.2f}  ({time.time()-t0:.0f}s)")
    # assemble
    for W in [8, 20]:
        rows = []
        for p in all_subjects:
            f = OUT / f"patient_{p}_W{W}{tag}.json"
            if not f.exists():
                continue
            res = json.load(open(f)); r, m = res["lda"], res["meandiff"]
            rows.append([p, r["auc_full"], r["sign_full"], r["stability"], r["auc_draw_min"], r["auc_draw_median"], r["auc_draw_max"],
                         m["auc_full"], m["sign_full"], m["stability"], m["auc_draw_min"], m["auc_draw_max"]])
        if rows:
            with open(OUT / f"per_patient_W{W}{tag}.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["patient", "lda_auc_full", "lda_sign", "lda_stability", "lda_auc_min", "lda_auc_median", "lda_auc_max",
                            "md_auc_full", "md_sign", "md_stability", "md_auc_min", "md_auc_max"])
                for r in rows:
                    w.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in r])


if __name__ == "__main__":
    main()
