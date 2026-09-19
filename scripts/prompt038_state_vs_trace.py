#!/usr/bin/env python3
"""
================================================================================
PROMPT 038 / Q1 - Is polarity in the state (rho) or in the trace?
================================================================================

Cache-only. Reads results/prompt037/cache/W8/chbNN.npz and splits each
patient's mean ictal shift vector s_bar (tangent space at the patient's own
Frechet mean, d = 36) into the component along the pure-scaling direction
u = vec(I_8)/sqrt(8) and the remainder:

    alpha_p   = <s_bar, u>                 (signed; + = ictal power up)
    s_perp    = s_bar - alpha_p u          (shape change)
    f_scale   = alpha_p^2 / ||s_bar||^2    (fraction of the shift that is scale)

Under pyriemann's log map at reference C, scaling Sigma -> e^t Sigma adds
t * vec(I) to the tangent vector, so u is exactly the scale direction and
sqrt(8) * alpha_p equals the ictal-minus-interictal difference of mean
log det Sigma. Both facts are asserted before anything else is computed.

Outputs (results/prompt038/):
    q1_scale_shape.csv, q1_per_seizure.csv, q1_polarity_crosstab.json,
    q1_shape_only_bw.csv, q1_calibration.json

Author: Claude Code
Date: 2026-09-05
================================================================================
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, fisher_exact

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.mean import mean_riemann, mean_wasserstein
from pyriemann.utils.distance import distance_wasserstein

import prompt037_per_seizure_decomposition as p37   # noqa: E402  (decompose, load_cache)

OUT = PROJECT_ROOT / "results" / "prompt038"
OUT.mkdir(parents=True, exist_ok=True)
P035 = PROJECT_ROOT / "results" / "prompt035"
P037 = PROJECT_ROOT / "results" / "prompt037"
W = 8.0
N = 8
STRONG_POLAR_REF = "chb03"      # sign anchor for the cohort shape axis


def log(m=""):
    print(m, flush=True)


def read_csv_dict(path):
    with open(path, newline="") as f:
        return {r["patient"]: r for r in csv.DictReader(f)}


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in r])


# ---------------------------------------------------------------- calibration
def scale_direction(ts: TangentSpace, d=N):
    """Unit tangent vector along uniform scaling, in pyriemann's vectorization.
    Verified by round-tripping through the fitted TangentSpace."""
    T = ts.transform(np.array([ts.reference_]))[0]        # log map of Cref = 0
    assert np.abs(T).max() < 1e-9, "reference does not map to the origin"
    u_mat = np.eye(d)
    # vectorize the identity the way pyriemann does: transform(Cref^{1/2} exp(I) Cref^{1/2})
    from scipy.linalg import sqrtm, expm
    Cs = np.real(sqrtm(ts.reference_))
    u_vec = ts.transform(np.array([Cs @ expm(u_mat) @ Cs]))[0]
    u = u_vec / np.linalg.norm(u_vec)
    assert abs(np.linalg.norm(u_vec) - np.sqrt(d)) < 1e-6, f"|vec(I)| = {np.linalg.norm(u_vec)}"
    # round trip: t*u must invert to e^t * Cref
    t = 0.3
    back = ts.inverse_transform(np.array([t * u * np.sqrt(d)]))[0]
    err = np.abs(back - np.exp(t) * ts.reference_).max() / np.abs(ts.reference_).max()
    assert err < 1e-8, f"scale round-trip error {err}"
    return u, float(err)


# ------------------------------------------------------------------- per patient
def analyze(sid, pol):
    z = p37.load_cache(sid, W)
    covs, phase, szid = z["covs"], z["phase"], z["seizure_id"]
    ts = TangentSpace(metric="riemann").fit(covs)
    tv = ts.transform(covs)
    u, cal_err = scale_direction(ts)
    m_int, m_ict = phase == "interictal", phase == "ictal"
    ks_all = [int(k) for k in z["seizure_ids"]]
    dec = p37.decompose(tv[m_ict], szid[m_ict], tv[m_int], ks_all)
    S = dec["s_vecs"]                                  # (K, 36) per-seizure shifts
    s_bar = S.mean(axis=0)

    alpha = float(s_bar @ u)
    s_perp = s_bar - alpha * u
    f_scale = alpha ** 2 / float(s_bar @ s_bar)

    # exact identity: sqrt(8)*alpha = mean_ictal logdet - mean_int logdet
    ld = np.linalg.slogdet(covs)[1]
    ld_ict = np.mean([ld[m_ict & (szid == k)].mean() for k in dec["ks"]])   # equal-weight over seizures
    ld_int = ld[m_int].mean()
    ident_err = abs(np.sqrt(N) * alpha - (ld_ict - ld_int))

    # per-seizure split
    a_k = S @ u
    f_k = a_k ** 2 / np.sum(S ** 2, axis=1)

    # log-trace ratio of class Frechet (Riemannian) means
    C_ict = mean_riemann(covs[m_ict]); C_int = mean_riemann(covs[m_int])
    logtr = float(np.log(np.trace(C_ict)) - np.log(np.trace(C_int)))

    # shape-only BW per seizure on trace-normalised covs
    rho = covs / np.trace(covs, axis1=1, axis2=2)[:, None, None]
    R_int = mean_wasserstein(rho[m_int])
    d_rho = []
    for k in dec["ks"]:
        x = rho[m_ict & (szid == k)]
        R_k = mean_wasserstein(x) if len(x) > 1 else x[0]
        d_rho.append(float(distance_wasserstein(R_k, R_int)))
    return dict(patient=sid, K=dec["K"], alpha=alpha, s_bar_norm=float(np.linalg.norm(s_bar)),
                s_perp_norm=float(np.linalg.norm(s_perp)), f_scale=f_scale, logtr_ratio=logtr,
                ident_err=ident_err, cal_err=cal_err, s_perp=s_perp, a_k=a_k, f_k=f_k,
                ks=dec["ks"], durations=[float(z["durations"][list(z["seizure_ids"]).index(k)]) for k in dec["ks"]],
                d_rho=np.array(d_rho), D_bar_rho=float(np.mean(d_rho)), t3=pol[sid]["t3_polarity"])


def main():
    pol = read_csv_dict(P035 / "polarity_magnitude.csv")
    clin = read_csv_dict(P035 / "clinical_metadata.csv")
    dec37 = read_csv_dict(P037 / "patient_decomposition.csv")
    subjects = sorted(p for p in dec37)
    log(f"Q1: {len(subjects)} patients, W={W:g}s cache")

    R = {}
    for s in subjects:
        r = analyze(s, pol); R[s] = r
        log(f"  {s}: alpha={r['alpha']:+.3f} |s|={r['s_bar_norm']:.3f} f_scale={r['f_scale']:.3f} "
            f"logtr={r['logtr_ratio']:+.3f} ident_err={r['ident_err']:.1e} t3={r['t3']}")

    # ---- assertions
    max_cal = max(r["cal_err"] for r in R.values()); max_id = max(r["ident_err"] for r in R.values())
    alphas = np.array([R[s]["alpha"] for s in subjects]); logtrs = np.array([R[s]["logtr_ratio"] for s in subjects])
    sign_ok = int(np.sum(np.sign(alphas) == np.sign(logtrs)))
    rho_al, p_al = spearmanr(alphas, logtrs)
    log(f"  calibration: max round-trip err {max_cal:.1e}; max |sqrt8*alpha - dlogdet| {max_id:.1e}; "
        f"sign(alpha)==sign(logtr) {sign_ok}/22; spearman {rho_al:.3f}")
    cal = dict(max_roundtrip_err=max_cal, max_identity_err=max_id, sign_agreement=sign_ok,
               spearman_alpha_logtr=float(rho_al), unit_test_1=bool(max_cal < 1e-8 and max_id < 1e-8),
               unit_test_2=bool(sign_ok == 22 and rho_al > 0.9))
    json.dump(cal, open(OUT / "q1_calibration.json", "w"), indent=2)

    # ---- cohort shape axis (uncentred PCA of s_perp), sign anchored on chb03
    P = np.array([R[s]["s_perp"] for s in subjects])
    _, sv, Vt = np.linalg.svd(P, full_matrices=False)
    v1 = Vt[0]
    if (R[STRONG_POLAR_REF]["s_perp"] @ v1) < 0:
        v1 = -v1
    shape_proj = P @ v1
    explained = float(sv[0] ** 2 / np.sum(sv ** 2))

    # ---- crosstabs vs t3_polarity
    t3 = np.array([1 if R[s]["t3"] == "standard" else -1 for s in subjects])

    def crosstab(sig, name):
        sig = np.sign(sig).astype(int)
        agree = int(np.sum(sig == t3))
        table = [[int(np.sum((sig > 0) & (t3 > 0))), int(np.sum((sig > 0) & (t3 < 0)))],
                 [int(np.sum((sig < 0) & (t3 > 0))), int(np.sum((sig < 0) & (t3 < 0)))]]
        _, pf = fisher_exact(table)
        best = max(agree, 22 - agree)
        return dict(name=name, agree=agree, agree_best_flip=best, fisher_p=float(pf), table=table)

    ct_scale = crosstab(alphas, "sign(alpha) vs t3_polarity")
    ct_shape = crosstab(shape_proj, "sign(shape projection on cohort axis) vs t3_polarity")
    json.dump(dict(scale=ct_scale, shape=ct_shape, shape_axis_explained_var=explained,
                   t3_polarity={s: R[s]["t3"] for s in subjects}),
              open(OUT / "q1_polarity_crosstab.json", "w"), indent=2)
    log(f"  scale sign vs t3: {ct_scale['agree']}/22 (best flip {ct_scale['agree_best_flip']}), fisher p={ct_scale['fisher_p']:.3f}")
    log(f"  shape sign vs t3: {ct_shape['agree']}/22 (best flip {ct_shape['agree_best_flip']}), fisher p={ct_shape['fisher_p']:.3f}; axis explains {explained:.2f}")

    # ---- shape-only BW vs 037 D_bar and duration
    Dr = np.array([R[s]["D_bar_rho"] for s in subjects])
    D37 = np.array([float(dec37[s]["D_bar"]) for s in subjects])
    dur = np.array([float(clin[s]["mean_seizure_duration_sec"]) for s in subjects])
    K = np.array([float(clin[s]["n_seizures"]) for s in subjects])
    r1, p1 = spearmanr(Dr, D37); r2, p2 = spearmanr(Dr, dur); r3, p3 = spearmanr(Dr, K)
    log(f"  shape-only D_bar: vs 037 D_bar rho={r1:.3f}; vs mean_dur rho={r2:.3f} p={p2:.4f}; vs K rho={r3:.3f} p={p3:.4f}")
    write_csv(OUT / "q1_shape_only_bw.csv", ["patient", "D_bar_rho", "D_bar_037", "mean_seizure_duration_sec", "n_seizures"],
              [[s, float(Dr[i]), float(D37[i]), float(dur[i]), float(K[i])] for i, s in enumerate(subjects)])
    with open(OUT / "q1_shape_only_bw.csv", "a") as f:
        f.write(f"# spearman D_bar_rho vs D_bar_037 = {r1:.4f} (p={p1:.2e}); vs mean_dur = {r2:.4f} (p={p2:.4f}); vs n_seizures = {r3:.4f} (p={p3:.4f})\n")

    # ---- per-patient and per-seizure tables
    write_csv(OUT / "q1_scale_shape.csv",
              ["patient", "K", "alpha", "s_bar_norm", "s_perp_norm", "f_scale", "logtr_ratio",
               "shape_proj_cohort_axis", "t3_polarity", "f_scale_seizure_min", "f_scale_seizure_median",
               "f_scale_seizure_max", "identity_err"],
              [[s, R[s]["K"], R[s]["alpha"], R[s]["s_bar_norm"], R[s]["s_perp_norm"], R[s]["f_scale"],
                R[s]["logtr_ratio"], float(shape_proj[i]), R[s]["t3"], float(R[s]["f_k"].min()),
                float(np.median(R[s]["f_k"])), float(R[s]["f_k"].max()), R[s]["ident_err"]]
               for i, s in enumerate(subjects)])
    rows = []
    for s in subjects:
        r = R[s]
        for k, ak, fk, dk, dur_k in zip(r["ks"], r["a_k"], r["f_k"], r["d_rho"], r["durations"]):
            rows.append([s, k, dur_k, float(ak), float(fk), float(dk)])
    write_csv(OUT / "q1_per_seizure.csv", ["patient", "seizure_id", "duration_sec", "alpha_k", "f_scale_k", "d_rho_k"], rows)

    med_f = float(np.median([R[s]["f_scale"] for s in subjects]))
    log(f"\n  median f_scale = {med_f:.3f}; per-patient range "
        f"{min(R[s]['f_scale'] for s in subjects):.3f}-{max(R[s]['f_scale'] for s in subjects):.3f}")
    log("Done.")


if __name__ == "__main__":
    main()
