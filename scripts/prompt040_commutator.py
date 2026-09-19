#!/usr/bin/env python3
"""
================================================================================
PROMPT 040 - Does non-commutativity between patients' shift generators predict
the reference-dependence of polarity?
================================================================================

Claim under test (from the 039 discussion): polarity is axis-dependent because
patients' ictal-shift generators do not share an eigenbasis. If so, the
commutator norm between two patients' generators should predict how badly one
patient's direction reads the other, beyond what plain alignment (cosine)
explains, and a patient's mean incompatibility with the cohort should predict
the reference-dependence measured in 039.

Objects (W = 8 cache, common frame):
    C_ref      Riemannian mean of ALL 22 patients' windows (ictal + interictal)
    L(C)       log(C_ref^{-1/2} C C_ref^{-1/2})           symmetric 8x8
    G_p        mean_{ictal} L - mean_{interictal} L        shift generator
    G_p^perp   G_p minus its identity component            shape generator
    cos_pq     <G_p, G_q>_F / (|G_p| |G_q|)                alignment
    kappa_pq   |[G_p, G_q]|_F / (sqrt(2) |G_p| |G_q|)      incompatibility, in [0, 1]
    AUC_{p|q}  AUC of patient p's windows scored by <L(C), G_q>   (q's direction)

Tests:
    T1  pairwise: |AUC_{p|q} - 0.5| vs cos and kappa (Spearman; partial of kappa given |cos|)
    T2  per patient: mean kappa_{p,.} and mean |cos_{p,.}| vs 039 LDA stability
        (strict and pooled), Spearman
    T3  same with shape-only generators

Outputs: results/prompt040/{generators.npz, pairwise.csv, per_patient.csv, tests.json}
Author: Claude Code
Date: 2026-09-06
================================================================================
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.metrics import roc_auc_score
from pyriemann.utils.mean import mean_riemann

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
import prompt037_per_seizure_decomposition as p37   # noqa: E402

OUT = PROJECT_ROOT / "results" / "prompt040"; OUT.mkdir(parents=True, exist_ok=True)
P039 = PROJECT_ROOT / "results" / "prompt039"
P038 = PROJECT_ROOT / "results" / "prompt038"
W = 8.0; N = 8


def log(m=""):
    print(m, flush=True)


def sym_fn(C, fn):
    w, V = np.linalg.eigh(C)
    return (V * fn(w)) @ V.T


def logmap(C, Cref_isqrt):
    M = Cref_isqrt @ C @ Cref_isqrt
    return sym_fn(0.5 * (M + M.T), np.log)


def frob(A):
    return float(np.linalg.norm(A, "fro"))


def partial_spearman(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    rxy, rxz, ryz = np.corrcoef(rx, ry)[0, 1], np.corrcoef(rx, rz)[0, 1], np.corrcoef(ry, rz)[0, 1]
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2)))


def main():
    subjects = sorted(p.stem for p in (p37.OUT_DIR / "cache" / "W8").glob("chb*.npz"))
    data = {}
    for s in subjects:
        z = p37.load_cache(s, W)
        m = np.isin(z["phase"], ["ictal", "interictal"])
        data[s] = (z["covs"][m], z["phase"][m] == "ictal")
    allc = np.concatenate([data[s][0] for s in subjects])
    log(f"common reference: Riemannian mean of {len(allc)} windows")
    Cref = mean_riemann(allc)
    Cref_isqrt = sym_fn(Cref, lambda w: 1 / np.sqrt(w))

    Lw, G, Gp = {}, {}, {}
    I_unit = np.eye(N) / np.sqrt(N)
    for s in subjects:
        covs, ict = data[s]
        L = np.array([logmap(c, Cref_isqrt) for c in covs])
        Lw[s] = (L, ict)
        g = L[ict].mean(0) - L[~ict].mean(0)
        G[s] = g
        Gp[s] = g - np.sum(g * I_unit) * I_unit
    np.savez(OUT / "generators.npz", subjects=np.array(subjects),
             G=np.array([G[s] for s in subjects]), G_perp=np.array([Gp[s] for s in subjects]), Cref=Cref)

    # pairwise quantities
    rows = []
    for p in subjects:
        Lp, ictp = Lw[p]
        for q in subjects:
            if p == q:
                continue
            cos = float(np.sum(G[p] * G[q]) / (frob(G[p]) * frob(G[q])))
            cosS = float(np.sum(Gp[p] * Gp[q]) / (frob(Gp[p]) * frob(Gp[q])))
            kap = frob(G[p] @ G[q] - G[q] @ G[p]) / (np.sqrt(2) * frob(G[p]) * frob(G[q]))
            kapS = frob(Gp[p] @ Gp[q] - Gp[q] @ Gp[p]) / (np.sqrt(2) * frob(Gp[p]) * frob(Gp[q]))
            auc = float(roc_auc_score(ictp, np.einsum("nij,ij->n", Lp, G[q])))
            aucS = float(roc_auc_score(ictp, np.einsum("nij,ij->n", Lp, Gp[q])))
            rows.append([p, q, cos, kap, auc, cosS, kapS, aucS])
    with open(OUT / "pairwise.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["p", "q", "cos", "kappa", "auc_p_given_q", "cos_shape", "kappa_shape", "auc_shape"])
        for r in rows:
            w.writerow([r[0], r[1]] + [f"{v:.5f}" for v in r[2:]])
    R = np.array([r[2:] for r in rows], float)
    cos, kap, auc, cosS, kapS, aucS = R.T
    dev, devS = np.abs(auc - 0.5), np.abs(aucS - 0.5)

    tests = {}
    tests["T1_full"] = dict(
        n_pairs=len(rows),
        spearman_dev_vs_abs_cos=spearmanr(dev, np.abs(cos))[0],
        spearman_dev_vs_kappa=spearmanr(dev, kap)[0],
        partial_dev_vs_kappa_given_abs_cos=partial_spearman(dev, kap, np.abs(cos)),
        sign_agreement_auc_vs_cos=float(np.mean(np.sign(auc - 0.5) == np.sign(cos))),
        spearman_abs_cos_vs_kappa=spearmanr(np.abs(cos), kap)[0],
        kappa_median=float(np.median(kap)), kappa_min=float(kap.min()), kappa_max=float(kap.max()),
        cos_median=float(np.median(cos)), frac_cos_negative=float(np.mean(cos < 0)),
        frac_auc_below_half=float(np.mean(auc < 0.5)))
    tests["T1_shape"] = dict(
        spearman_dev_vs_abs_cos=spearmanr(devS, np.abs(cosS))[0],
        spearman_dev_vs_kappa=spearmanr(devS, kapS)[0],
        partial_dev_vs_kappa_given_abs_cos=partial_spearman(devS, kapS, np.abs(cosS)),
        sign_agreement_auc_vs_cos=float(np.mean(np.sign(aucS - 0.5) == np.sign(cosS))),
        kappa_median=float(np.median(kapS)), cos_median=float(np.median(cosS)),
        frac_cos_negative=float(np.mean(cosS < 0)), frac_auc_below_half=float(np.mean(aucS < 0.5)))
    log(f"T1 full : dev~|cos| rho={tests['T1_full']['spearman_dev_vs_abs_cos']:.3f}  dev~kappa rho={tests['T1_full']['spearman_dev_vs_kappa']:.3f}  "
        f"partial(kappa|cos)={tests['T1_full']['partial_dev_vs_kappa_given_abs_cos']:.3f}  sign(auc)=sign(cos) {tests['T1_full']['sign_agreement_auc_vs_cos']:.3f}  "
        f"kappa med {tests['T1_full']['kappa_median']:.3f} cos med {tests['T1_full']['cos_median']:.3f}")
    log(f"T1 shape: dev~|cos| rho={tests['T1_shape']['spearman_dev_vs_abs_cos']:.3f}  dev~kappa rho={tests['T1_shape']['spearman_dev_vs_kappa']:.3f}  "
        f"partial(kappa|cos)={tests['T1_shape']['partial_dev_vs_kappa_given_abs_cos']:.3f}  sign agree {tests['T1_shape']['sign_agreement_auc_vs_cos']:.3f}  "
        f"kappa med {tests['T1_shape']['kappa_median']:.3f} cos med {tests['T1_shape']['cos_median']:.3f} neg {tests['T1_shape']['frac_cos_negative']:.2f}")

    # per patient
    def load039(name):
        p = P039 / name
        return {r["patient"]: r for r in csv.DictReader(open(p))} if p.exists() else {}
    s8, s8p = load039("per_patient_W8.csv"), load039("per_patient_W8_pre.csv")
    q1 = {r["patient"]: r for r in csv.DictReader(open(P038 / "q1_scale_shape.csv"))}
    per = []
    for p in subjects:
        idx = [i for i, r in enumerate(rows) if r[0] == p]
        per.append(dict(patient=p, mean_kappa=float(kap[idx].mean()), mean_abs_cos=float(np.abs(cos[idx]).mean()),
                        mean_cos=float(cos[idx].mean()), mean_kappa_shape=float(kapS[idx].mean()),
                        mean_abs_cos_shape=float(np.abs(cosS[idx]).mean()), mean_cos_shape=float(cosS[idx].mean()),
                        frac_others_read_me_inverted=float(np.mean(auc[idx] < 0.5)),
                        frac_others_read_me_inverted_shape=float(np.mean(aucS[idx] < 0.5)),
                        G_norm=frob(G[p]), G_perp_norm=frob(Gp[p]), f_scale=float(q1[p]["f_scale"]),
                        stab_strict_lda=float(s8[p]["lda_stability"]) if p in s8 else np.nan,
                        stab_pooled_lda=float(s8p[p]["lda_stability"]) if p in s8p else np.nan,
                        auc_pooled_lda=float(s8p[p]["lda_auc_full"]) if p in s8p else np.nan))
    keys = list(per[0].keys())
    with open(OUT / "per_patient.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(keys)
        for r in per:
            w.writerow([f"{r[k]:.5f}" if isinstance(r[k], float) else r[k] for k in keys])
    A = {k: np.array([r[k] for r in per]) for k in keys if k != "patient"}
    t2 = {}
    for target in ["stab_strict_lda", "stab_pooled_lda", "frac_others_read_me_inverted", "frac_others_read_me_inverted_shape"]:
        for pred in ["mean_kappa", "mean_abs_cos", "mean_kappa_shape", "mean_abs_cos_shape", "f_scale", "G_perp_norm"]:
            ok = np.isfinite(A[target]) & np.isfinite(A[pred])
            r, pv = spearmanr(A[pred][ok], A[target][ok])
            t2[f"{target} ~ {pred}"] = dict(rho=float(r), p=float(pv), n=int(ok.sum()))
    tests["T2_T3_per_patient"] = t2
    log("T2/T3 per patient (Spearman):")
    for k, v in t2.items():
        log(f"  {k:60s} rho={v['rho']:+.3f} p={v['p']:.3f}")
    json.dump(tests, open(OUT / "tests.json", "w"), indent=2, default=float)
    log("Done.")


if __name__ == "__main__":
    main()
