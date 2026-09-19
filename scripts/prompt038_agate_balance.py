#!/usr/bin/env python3
"""
================================================================================
PROMPT 038 / Q2 - Does the A-Gate's model-derived E/I balance observable read
manifold polarity?
================================================================================

One EDF pass through the PROMPT 037 loader (W = 8 s, identical windows to
results/prompt037/cache/W8 - asserted), (a, b, c) per channel-window via the
PROMPT 006 encoding (`extract_plv_params`, unchanged), then three observables
per window from the single-channel A-Gate statevector (coupling included):

    Pi_bal   = <U X_E U^dag>/sin b + <U X_I U^dag>/cos b   (= sin 2a - sin 2c)
    Pi_naive = <Z_E - Z_I>                                  (un-derived control)
    Pi_anc   = <Z_ancilla> of the full 8-channel circuit    (Paper 1's readout)

Per patient: sign of median(ictal) - median(interictal), bootstrap stability,
agreement with t3_polarity (035) and with the Paper 2 hardware assignments.

Stages: --stage abc (EDF pass -> abc_cache), --stage test (unit tests),
--stage analyze (everything else), default all.

Author: Claude Code
Date: 2026-09-05
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
from scipy.stats import binomtest, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import prompt037_per_seizure_decomposition as p37          # noqa: E402
from braket_ch16_simulation import extract_plv_params        # noqa: E402  (PROMPT 006 encoding, verbatim)

OUT = PROJECT_ROOT / "results" / "prompt038"
ABC = OUT / "abc_cache"
OUT.mkdir(parents=True, exist_ok=True); ABC.mkdir(exist_ok=True)
P035 = PROJECT_ROOT / "results" / "prompt035"
P037 = PROJECT_ROOT / "results" / "prompt037"
W = 8.0
BAND = (4, 13)
DEAD = 0.3                      # |sin b| or |cos b| below this -> term unreadable
AMAX = np.pi / 4                # monotone range of sin 2a
N_BOOT = 2000
SEED = 38
HW_INVERTED = {"chb03", "chb11", "chb21"}                       # Paper 2, CH8 1.95 s
HW_SUBSET = ["chb01", "chb03", "chb05", "chb07", "chb11", "chb14", "chb21"]


def log(m=""):
    print(m, flush=True)


# =============================================================================
# 2-qubit A-Gate in numpy (q0 = E is the FIRST tensor factor here; the qiskit
# ordering is handled - and asserted - in the unit tests)
# =============================================================================
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1.]).astype(complex)
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
P0 = np.diag([1., 0.]); P1 = np.diag([0., 1.])
Pg = lambda b: np.diag([1, np.exp(1j * b)])
Rx = lambda t: np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]])
Ry = lambda t: np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
Rz = lambda t: np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])


def ctrl(Ug, control):
    """controlled-U on 2 qubits, tensor order (E, I)."""
    return np.kron(P0, I2) + np.kron(P1, Ug) if control == 0 else np.kron(I2, P0) + np.kron(Ug, P1)


U_COUPLE = ctrl(Rz(np.pi / 4), 1) @ ctrl(Ry(np.pi / 4), 0)      # CRy(E->I) then CRz(I->E)
XE, XI = np.kron(X, I2), np.kron(I2, X)
ZE, ZI = np.kron(Z, I2), np.kron(I2, Z)
OXE = U_COUPLE @ XE @ U_COUPLE.conj().T                        # transformed observables
OXI = U_COUPLE @ XI @ U_COUPLE.conj().T


def agate_state(a, b, c):
    UE = H @ Pg(b) @ Rx(2 * a) @ Pg(b) @ H
    UI = H @ Pg(b) @ Ry(2 * c) @ Pg(b) @ H
    return U_COUPLE @ np.kron(UE, UI) @ np.array([1, 0, 0, 0], complex)


def expect(O, psi):
    return float(np.real(psi.conj() @ O @ psi))


def channel_observables(a, b, c):
    """returns (Pi_bal or nan, Pi_naive, <Z_E>) for one channel."""
    psi = agate_state(a, b, c)
    sb, cb = np.sin(b), np.cos(b)
    bal = np.nan if (abs(sb) < DEAD or abs(cb) < DEAD) else expect(OXE, psi) / sb + expect(OXI, psi) / cb
    return bal, expect(ZE, psi) - expect(ZI, psi), expect(ZE, psi)


# =============================================================================
# Ancilla observable of the full circuit = product of single-channel <Z_E>
# over a subset S fixed by the closed CNOT ring (Heisenberg propagation of the
# E-parity string). Derived once by Pauli-string propagation, verified against
# the 17-qubit qiskit statevector in the unit tests.
# =============================================================================
def ancilla_subset(M=8):
    # ancilla: H, CZ from every E, H  ->  <Z_anc> = < prod_i Z_{E_i} >  (before layer 3)
    # propagate prod Z_E backwards through the E ring CNOT(E_i -> E_{(i+1)%M}), i = 0..M-1,
    # applied in that order; Heisenberg conjugation goes in reverse gate order.
    S = set(range(M))
    for i in reversed(range(M)):
        c, t = i, (i + 1) % M
        if t in S:                    # Z_t -> Z_c Z_t
            S ^= {c}
    return sorted(S)


ANC_S = ancilla_subset(8)


def ancilla_from_channels(zE):
    return float(np.prod([zE[i] for i in ANC_S]))


# =============================================================================
# Stage: abc  (EDF pass, same windows as the 037 cache)
# =============================================================================
def load_windows_like_037(sid, data_root, window_sec):
    """Replicates p37.load_patient_windows but keeps the preprocessed windows."""
    from sagemaker.train_chbmit import preprocess_eeg, read_edf_segment
    from pyriemann.estimation import Covariances
    subject_dir = data_root / sid
    segs, durations = p37.build_segments(subject_dir)
    windows, phase, sids, widx = [], [], [], []
    for seg, s_id in segs:
        edf_path = data_root / seg.subject / seg.source_file
        if not edf_path.exists():
            continue
        data, fs = read_edf_segment(str(edf_path), seg.start_sec, seg.end_sec, target_channels=p37.LOSO_CHANNELS)
        if data is None or data.size == 0:
            continue
        expected = int(window_sec * fs)
        data = preprocess_eeg(data, fs=int(fs))
        for i in range(data.shape[1] // expected):
            windows.append(data[:, i * expected:(i + 1) * expected]); phase.append(seg.label); sids.append(s_id); widx.append(i)
    windows = np.array(windows); covs = Covariances(estimator="lwf").transform(windows)
    valid = np.array([bool(np.all(np.linalg.eigvalsh(cv) > 0) and not np.isnan(cv).any()) for cv in covs])
    return windows[valid], np.array(phase)[valid], np.array(sids)[valid], np.array(widx)[valid], fs


def stage_abc(subjects, data_root):
    for sid in subjects:
        f = ABC / f"{sid}.npz"
        if f.exists():
            log(f"  {sid}: cached"); continue
        t0 = time.time()
        win, phase, sids, widx, fs = load_windows_like_037(sid, data_root, W)
        z = p37.load_cache(sid, W)
        mask = phase != "interictal"; mask[z["interictal_kept"]] = True
        win, phase, sids, widx = win[mask], phase[mask], sids[mask], widx[mask]
        assert (phase == z["phase"]).all() and (sids == z["seizure_id"]).all() and (widx == z["window_idx"]).all(), \
            f"{sid}: window set differs from the 037 cache"
        abc = np.array([extract_plv_params(w, fs=float(fs), band=BAND) for w in win])   # (n, 8, 3)
        np.savez(f, abc=abc, phase=phase, seizure_id=sids, window_idx=widx)
        log(f"  {sid}: {len(win)} windows -> (a,b,c) in {time.time()-t0:.1f}s")


# =============================================================================
# Stage: test
# =============================================================================
def stage_test():
    from qiskit.quantum_info import Statevector, SparsePauliOp
    from qdnu.quantum_agate import create_single_channel_agate
    from qdnu.multichannel_circuit import create_multichannel_circuit
    rng = np.random.default_rng(SEED); res = {}

    # closed forms, uncoupled
    err = 0.0
    for _ in range(500):
        a, b, c = rng.uniform(0, 1), rng.uniform(0, 2 * np.pi), rng.uniform(0, 1)
        pe = H @ Pg(b) @ Rx(2 * a) @ Pg(b) @ H @ np.array([1, 0], complex)
        pi_ = H @ Pg(b) @ Ry(2 * c) @ Pg(b) @ H @ np.array([1, 0], complex)
        e = lambda O, p: float(np.real(p.conj() @ O @ p))
        err = max(err, abs(e(X, pe) - np.sin(2 * a) * np.sin(b)), abs(e(X, pi_) + np.sin(2 * c) * np.cos(b)),
                  abs(e(Z, pe) - (np.sin(a) ** 2 + np.cos(a) ** 2 * np.cos(2 * b))),
                  abs(e(Z, pi_) - (-np.sin(c) ** 2 + np.cos(c) ** 2 * np.cos(2 * b))))
    res["closed_forms_max_err"] = err

    # numpy 2-qubit state == qiskit create_single_channel_agate (q0 = E), and balance exactness
    err_sv, err_bal = 0.0, 0.0
    for _ in range(500):
        a, b, c = rng.uniform(0, 1), rng.uniform(0, 2 * np.pi), rng.uniform(0, 1)
        sv = Statevector(create_single_channel_agate(a, b, c)).data
        psi = agate_state(a, b, c)
        # qiskit little-endian: amplitude index = q1*2 + q0 ; ours = E*2 + I with E=q0 -> swap
        sv_ours = sv.reshape(2, 2).T.reshape(4)          # [q1,q0] -> [q0,q1]
        err_sv = max(err_sv, float(np.abs(np.abs(sv_ours) - np.abs(psi)).max()),
                     float(abs(abs(np.vdot(sv_ours, psi)) - 1)))
        if min(abs(np.sin(b)), abs(np.cos(b))) > 1e-3:
            bal = expect(OXE, psi) / np.sin(b) + expect(OXI, psi) / np.cos(b)
            err_bal = max(err_bal, abs(bal - (np.sin(2 * a) - np.sin(2 * c))))
    res["qiskit_vs_numpy_state_max_err"] = err_sv
    res["balance_exactness_max_err"] = err_bal

    # Pauli expansion of U X_E U^dag: leading coefficient must be ~0.789 on X_E
    paulis = {"I": I2, "X": X, "Y": Y, "Z": Z}; coefs = {}
    for n1, p1 in paulis.items():
        for n2, p2 in paulis.items():
            cf = np.trace(np.kron(p1, p2).conj().T @ OXE) / 4
            if abs(cf) > 1e-9:
                coefs[f"{n1}_E {n2}_I"] = round(float(cf.real), 4)
    res["U_XE_Udag_pauli"] = coefs
    res["leading_on_XE"] = bool(abs(coefs.get("X_E I_I", 0) - 0.7886) < 5e-4)

    # ancilla shortcut vs the 17-qubit circuit
    err_anc = 0.0
    for _ in range(20):
        params = [(rng.uniform(0.05, 0.95), rng.uniform(0, 2 * np.pi), rng.uniform(0.05, 0.95)) for _ in range(8)]
        qc = create_multichannel_circuit(params)
        z_anc = float(np.real(Statevector(qc).expectation_value(SparsePauliOp("I" * 16 + "Z"))))
        zE = [channel_observables(*p)[2] for p in params]
        err_anc = max(err_anc, abs(z_anc - ancilla_from_channels(zE)))
    res["ancilla_subset_E_indices"] = ANC_S
    res["ancilla_shortcut_max_err"] = err_anc

    res["pass"] = bool(err < 1e-12 and err_sv < 1e-9 and err_bal < 1e-12 and res["leading_on_XE"] and err_anc < 1e-9)
    json.dump(res, open(OUT / "q2_unit_tests.json", "w"), indent=2)
    for k, v in res.items():
        log(f"  {k}: {v}")
    return res["pass"]


# =============================================================================
# Stage: analyze
# =============================================================================
def read_csv_dict(path):
    with open(path, newline="") as f:
        return {r["patient"]: r for r in csv.DictReader(f)}


def stage_analyze(subjects):
    pol = read_csv_dict(P035 / "polarity_magnitude.csv")
    dec37 = read_csv_dict(P037 / "patient_decomposition.csv")
    q1 = read_csv_dict(OUT / "q1_scale_shape.csv") if (OUT / "q1_scale_shape.csv").exists() else {}
    rng = np.random.default_rng(SEED)

    # ---- Step 0: coverage
    cov_rows, pooled = [], dict(n=0, a_hi=0, c_hi=0, b_dead_E=0, b_dead_I=0, readable=0)
    per_win = {}
    for sid in subjects:
        z = np.load(ABC / f"{sid}.npz", allow_pickle=False)
        abc, phase = z["abc"], z["phase"]
        a, b, c = abc[..., 0], abc[..., 1], abc[..., 2]
        dE, dI = np.abs(np.sin(b)) < DEAD, np.abs(np.cos(b)) < DEAD
        ok = (a <= AMAX) & (c <= AMAX) & ~dE & ~dI
        row = [sid, a.size, float((a > AMAX).mean()), float((c > AMAX).mean()), float(dE.mean()), float(dI.mean()), float(ok.mean()),
               float(np.median(a)), float(np.median(c))]
        cov_rows.append(row)
        for k, v in zip(["n", "a_hi", "c_hi", "b_dead_E", "b_dead_I", "readable"],
                        [a.size, (a > AMAX).sum(), (c > AMAX).sum(), dE.sum(), dI.sum(), ok.sum()]):
            pooled[k] += int(v)
        # per-window observables
        bal = np.full(abc.shape[:2], np.nan); naive = np.zeros(abc.shape[:2]); zE = np.zeros(abc.shape[:2])
        for i in range(abc.shape[0]):
            for ch in range(8):
                bal[i, ch], naive[i, ch], zE[i, ch] = channel_observables(*abc[i, ch])
        with np.errstate(all="ignore"):
            bal_w = np.nanmean(bal, axis=1)
        n_contrib = np.sum(~np.isnan(bal), axis=1)
        naive_w = naive.mean(axis=1)
        anc_w = np.array([ancilla_from_channels(zE[i]) for i in range(len(zE))])
        per_win[sid] = dict(phase=phase, bal=bal_w, n_contrib=n_contrib, naive=naive_w, anc=anc_w,
                            a=a, c=c, b=b)
        np.savez(OUT / f"q2_per_window_{sid}.npz", phase=phase, seizure_id=z["seizure_id"],
                 pi_bal=bal_w, n_contrib=n_contrib, pi_naive=naive_w, pi_anc=anc_w)
    pooled_frac = {k: pooled[k] / pooled["n"] for k in ["a_hi", "c_hi", "b_dead_E", "b_dead_I", "readable"]}
    with open(OUT / "q2_step0_encoding_coverage.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "n_channel_windows", "frac_a_gt_pi4", "frac_c_gt_pi4", "frac_b_deadzone_E",
                    "frac_b_deadzone_I", "frac_fully_readable", "median_a", "median_c"])
        for r in cov_rows:
            w.writerow([r[0], r[1]] + [f"{v:.4f}" for v in r[2:]])
        w.writerow(["POOLED", pooled["n"]] + [f"{pooled_frac[k]:.4f}" for k in ["a_hi", "c_hi", "b_dead_E", "b_dead_I", "readable"]] + ["", ""])
    log(f"Step 0 pooled: a>pi/4 {pooled_frac['a_hi']:.3f}, c>pi/4 {pooled_frac['c_hi']:.3f}, "
        f"E dead {pooled_frac['b_dead_E']:.3f}, I dead {pooled_frac['b_dead_I']:.3f}, fully readable {pooled_frac['readable']:.3f}")

    # ---- Step 3: per-patient signs
    t3 = {s: (1 if pol[s]["t3_polarity"] == "standard" else -1) for s in subjects}
    t3_auc = {s: float(pol[s]["t3_raw_auc"]) for s in subjects}
    hw = {s: (-1 if s in HW_INVERTED else 1) for s in HW_SUBSET}
    rows, signs = [], {k: {} for k in ["bal", "naive", "anc"]}
    for sid in subjects:
        d = per_win[sid]; m_i, m_n = d["phase"] == "ictal", d["phase"] == "interictal"
        row = [sid, int(m_i.sum()), int(m_n.sum()), t3[sid], t3_auc[sid], hw.get(sid, "")]
        for key in ["bal", "naive", "anc"]:
            x = d[key]; xi, xn = x[m_i], x[m_n]
            xi, xn = xi[np.isfinite(xi)], xn[np.isfinite(xn)]
            delta = float(np.median(xi) - np.median(xn))
            bs = np.array([np.median(rng.choice(xi, len(xi))) - np.median(rng.choice(xn, len(xn))) for _ in range(N_BOOT)])
            stab = float(np.mean(np.sign(bs) == np.sign(delta)))
            determined = stab >= 0.7
            signs[key][sid] = (int(np.sign(delta)), determined, delta, stab)
            row += [delta, stab, int(np.sign(delta)) if determined else 0]
        rows.append(row)
        log(f"  {sid}: t3={'+' if t3[sid]>0 else '-'}({t3_auc[sid]:.2f}) hw={hw.get(sid,'.')}  "
            + "  ".join(f"{k}={signs[k][sid][2]:+.3f}({'det' if signs[k][sid][1] else 'und'} {signs[k][sid][3]:.2f})" for k in signs))
    with open(OUT / "q2_per_patient_signs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient", "n_ictal", "n_interictal", "t3_polarity_sign", "t3_raw_auc", "hw_polarity_sign",
                    "delta_bal", "stab_bal", "sign_bal", "delta_naive", "stab_naive", "sign_naive",
                    "delta_anc", "stab_anc", "sign_anc"])
        for r in rows:
            w.writerow([f"{v:.6g}" if isinstance(v, float) else v for v in r])

    # ---- agreement
    def agreement(key, ref, subset, label):
        det = [s for s in subset if signs[key][s][1]]
        agree = sum(signs[key][s][0] == ref[s] for s in det)
        n = len(det); p = float(binomtest(agree, n, 0.5).pvalue) if n else np.nan
        best = max(agree, n - agree); p_best = float(min(1.0, 2 * binomtest(best, n, 0.5, alternative="greater").pvalue)) if n else np.nan
        return dict(label=label, n_determined=n, n_total=len(subset), agree=agree, p_two_sided=p,
                    agree_best_flip=best, p_best_flip_adjusted=p_best, undetermined=[s for s in subset if not signs[key][s][1]])
    strong = [s for s in subjects if abs(t3_auc[s] - 0.5) >= 0.1]          # post-hoc, reported separately
    result = {}
    for key in ["bal", "naive", "anc"]:
        result[key] = dict(
            vs_t3_all22=agreement(key, t3, subjects, "vs t3_polarity, all 22"),
            vs_t3_strong=agreement(key, t3, strong, f"vs t3_polarity, |AUC-0.5|>=0.1 (n={len(strong)}; post hoc)"),
            vs_hardware_paper2=agreement(key, hw, HW_SUBSET, "vs Paper 2 hardware polarity, 7-patient subset"))
        for k2, v in result[key].items():
            log(f"  {key:5s} {v['label']}: {v['agree']}/{v['n_determined']} determined (of {v['n_total']}), "
                f"p={v['p_two_sided']:.3f}; best flip {v['agree_best_flip']} p_adj={v['p_best_flip_adjusted']:.3f}")

    # ---- cross-check with Q1 alpha and 037 L_corr
    if q1:
        alpha = np.array([float(q1[s]["alpha"]) for s in subjects])
        for key in ["bal", "naive", "anc"]:
            d = np.array([signs[key][s][2] for s in subjects])
            r, p = spearmanr(d, alpha)
            result[key]["spearman_delta_vs_alpha_Q1"] = dict(rho=float(r), p=float(p))
            L = np.array([float(dec37[s]["L_corr"]) for s in subjects]) * np.array([t3[s] for s in subjects])
            r2, p2 = spearmanr(d, L)
            result[key]["spearman_delta_vs_signed_Lcorr_037"] = dict(rho=float(r2), p=float(p2))
            log(f"  {key}: delta vs Q1 alpha rho={r:+.3f} (p={p:.3f}); vs signed L_corr rho={r2:+.3f} (p={p2:.3f})")
    result["step0_pooled"] = pooled_frac
    result["dead_zone"] = DEAD; result["a_max"] = AMAX; result["n_boot"] = N_BOOT; result["strong_subset"] = strong
    json.dump(result, open(OUT / "q2_agreement.json", "w"), indent=2)

    # ---- scatters
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    sc = OUT / "scatters"; sc.mkdir(exist_ok=True)
    if q1:
        for key in ["bal", "anc"]:
            d = np.array([signs[key][s][2] for s in subjects]); col = ["tab:blue" if t3[s] > 0 else "tab:red" for s in subjects]
            fig, ax = plt.subplots(figsize=(5, 4)); ax.scatter(alpha, d, c=col, s=40)
            for s, x_, y_ in zip(subjects, alpha, d):
                ax.annotate(s[3:], (x_, y_), fontsize=7, xytext=(3, 3), textcoords="offset points")
            ax.axhline(0, color="k", lw=0.5); ax.set_xlabel("Q1 alpha (scale component of ictal shift)")
            ax.set_ylabel(f"delta Pi_{key} (ictal - interictal)"); ax.set_title(f"Pi_{key} vs scale; blue=t3 standard, red=inverted", fontsize=9)
            ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(sc / f"q2_delta_{key}_vs_alpha.png", dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", "abc", "test", "analyze"])
    ap.add_argument("--data-root", default=str(p37.DEFAULT_DATA_ROOT))
    ap.add_argument("--subjects", default="")
    args = ap.parse_args()
    subjects = args.subjects.split(",") if args.subjects else sorted(read_csv_dict(P037 / "patient_decomposition.csv"))
    t0 = time.time()
    if args.stage in ("all", "test"):
        log("Unit tests"); ok = stage_test(); log(f"  unit tests: {'PASS' if ok else 'FAIL'}")
        if not ok: sys.exit(1)
        if args.stage == "test": return
    if args.stage in ("all", "abc"):
        log("Step: (a,b,c) per window"); stage_abc(subjects, Path(args.data_root))
        if args.stage == "abc": log(f"Done in {time.time()-t0:.0f}s"); return
    log("Analyze"); stage_analyze(subjects); log(f"Done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
