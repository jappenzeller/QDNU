#!/usr/bin/env python3
"""
BW-001 — Bures-Wasserstein Polarity Replication Check

Does per-patient manifold polarity replicate under the Bures-Wasserstein metric,
or is it specific to the affine-invariant geometry?

Runs the full LOSO pipeline twice (metric='riemann' vs metric='wasserstein')
on all 22 CHB-MIT patients. Same 8-channel covariances, same LDA, same folds.

Outputs:
  results/bw_polarity/per_patient_polarity_comparison.csv
  results/bw_polarity/per_patient_moments.csv
  results/bw_polarity/projection_scatter.png
  results/bw_polarity/SUMMARY.md
"""

import sys
import json
import logging
import csv
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from sagemaker.train_chbmit import (
    EXCLUDE_SUBJECTS, WINDOW_SEC, FS,
    preprocess_eeg, read_edf_segment, extract_segments_for_subject,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
LOSO_CHANNELS = ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
                 "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]
DATA_ROOT = Path("H:/Data/PythonDNU/EEG/chbmit")
OUTPUT_DIR = Path(__file__).parent.parent / "results" / "bw_polarity"
FALLBACK_WINDOW_SEC = 10.0
SNR_THRESHOLD = 1.0


# ── Data loading (copied from tier3_loso_cv.py, unchanged) ─────────────────
def load_covariances_for_subject(subject_id, channels=LOSO_CHANNELS,
                                  window_sec=WINDOW_SEC):
    subject_dir = DATA_ROOT / subject_id
    if not subject_dir.exists():
        return np.array([]), np.array([]), window_sec

    segments = extract_segments_for_subject(subject_dir)
    if not segments:
        return np.array([]), np.array([]), window_sec

    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    windows_list, labels_list = [], []

    for seg in segments:
        try:
            edf_path = DATA_ROOT / seg.subject / seg.source_file
            if not edf_path.exists():
                continue
            data, fs = read_edf_segment(str(edf_path), seg.start_sec,
                                        seg.end_sec, target_channels=channels)
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
        return np.array([]), np.array([]), window_sec

    windows = np.array(windows_list)
    labels = np.array(labels_list)

    try:
        covs = Covariances(estimator='lwf').transform(windows)
    except Exception:
        return np.array([]), np.array([]), window_sec

    # Filter valid SPD
    valid = []
    for cov in covs:
        try:
            eigs = np.linalg.eigvalsh(cov)
            valid.append(bool(np.all(eigs > 0) and not np.isnan(cov).any()))
        except Exception:
            valid.append(False)
    valid = np.array(valid)
    return covs[valid], labels[valid], window_sec


def load_covariances_with_fallback(subject_id):
    covs, labels, w = load_covariances_for_subject(subject_id)
    if len(covs) > 0 and len(np.unique(labels)) == 2:
        return covs, labels, w
    logger.info(f"  {subject_id}: fallback to {FALLBACK_WINDOW_SEC}s")
    covs, labels, _ = load_covariances_for_subject(
        subject_id, window_sec=FALLBACK_WINDOW_SEC)
    return covs, labels, FALLBACK_WINDOW_SEC


# ── LOSO with score collection ─────────────────────────────────────────────
def run_loso(subject_data, metric):
    """
    Run TS+LDA LOSO for a given metric.
    Returns dict: patient -> {auc, y_true, y_prob, scores, polarity}
    """
    subjects = sorted(subject_data.keys())
    results = {}

    for test_subj in subjects:
        X_train = np.vstack([subject_data[s][0] for s in subjects if s != test_subj])
        y_train = np.concatenate([subject_data[s][1] for s in subjects if s != test_subj])
        X_test, y_test = subject_data[test_subj][0], subject_data[test_subj][1]

        try:
            clf = Pipeline([
                ('ts', TangentSpace(metric=metric)),
                ('lda', LinearDiscriminantAnalysis()),
            ])
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]

            # LDA decision function scores (for moment analysis)
            ts_step = clf.named_steps['ts']
            lda_step = clf.named_steps['lda']
            X_test_ts = ts_step.transform(X_test)
            scores = lda_step.decision_function(X_test_ts)

            auc = roc_auc_score(y_test, y_prob)
        except Exception as e:
            logger.warning(f"  {test_subj} [{metric}]: failed — {e}")
            y_prob = np.full(len(y_test), 0.5)
            scores = np.zeros(len(y_test))
            auc = 0.5

        # Polarity from training fold: sign(median(score_ictal) - median(score_interictal))
        # Use training patients' within-fold scores for polarity determination
        # (matching Paper 2 definition: sign of median score gap on training data)
        try:
            ts_train = clf.named_steps['ts']
            lda_train = clf.named_steps['lda']
            X_train_ts = ts_train.transform(X_train)
            train_scores = lda_train.decision_function(X_train_ts)
            train_ictal = train_scores[y_train == 1]
            train_inter = train_scores[y_train == 0]
            polarity_sign = np.sign(np.median(train_ictal) - np.median(train_inter))
        except Exception:
            polarity_sign = 1.0

        # But also determine polarity from test AUC (oracle, for comparison)
        oracle_polarity = 'standard' if auc >= 0.5 else 'inverted'

        results[test_subj] = {
            'auc': float(auc),
            'oracle_polarity': oracle_polarity,
            'train_polarity_sign': float(polarity_sign),
            'y_true': y_test.tolist(),
            'y_prob': y_prob.tolist(),
            'scores': scores.tolist(),
        }

        logger.info(f"  {test_subj} [{metric}]: AUC={auc:.3f} pol={oracle_polarity}")

    return results


# ── Moment analysis ────────────────────────────────────────────────────────
def compute_moments(results):
    """Compute M1, M2, SNR per patient."""
    moments = {}
    for patient, r in results.items():
        scores = np.array(r['scores'])
        m1 = float(np.mean(scores))
        m2 = float(np.var(scores))
        snr = (m1 ** 2) / m2 if m2 > 1e-12 else 0.0
        moments[patient] = {'M1': m1, 'M2': m2, 'SNR': snr}
    return moments


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BW-001: Bures-Wasserstein Polarity Replication Check")
    logger.info("=" * 60)

    # Discover subjects
    all_subjects = sorted([
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith('chb')
        and d.name not in EXCLUDE_SUBJECTS
    ])
    logger.info(f"Found {len(all_subjects)} subjects")

    # Load covariances once (shared between both metrics)
    logger.info("\nLoading covariance matrices...")
    subject_data = {}
    for sid in all_subjects:
        covs, labels, w = load_covariances_with_fallback(sid)
        if len(covs) > 0 and len(np.unique(labels)) == 2:
            subject_data[sid] = (covs, labels, w)
            n1 = int(labels.sum())
            logger.info(f"  {sid}: {len(covs)} matrices ({n1} seizure, "
                        f"{len(covs)-n1} non-seizure, window={w}s)")
        else:
            logger.warning(f"  {sid}: skipped")

    logger.info(f"\nValid subjects: {len(subject_data)}")

    # Run LOSO with both metrics
    logger.info("\n--- Affine-invariant (riemann) ---")
    res_ai = run_loso(subject_data, 'riemann')

    logger.info("\n--- Bures-Wasserstein (wasserstein) ---")
    res_bw = run_loso(subject_data, 'wasserstein')

    # Compute moments
    mom_ai = compute_moments(res_ai)
    mom_bw = compute_moments(res_bw)

    # ── CSV 1: polarity comparison ──────────────────────────────────────
    patients = sorted(res_ai.keys())
    csv1_path = OUTPUT_DIR / "per_patient_polarity_comparison.csv"
    with open(csv1_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['patient', 'polarity_affine', 'polarity_bw',
                     'auc_affine', 'auc_bw', 'concordant'])
        n_concordant = 0
        for p in patients:
            pol_ai = res_ai[p]['oracle_polarity']
            pol_bw = res_bw[p]['oracle_polarity']
            conc = 1 if pol_ai == pol_bw else 0
            n_concordant += conc
            w.writerow([p, pol_ai, pol_bw,
                        f"{res_ai[p]['auc']:.4f}", f"{res_bw[p]['auc']:.4f}",
                        conc])
    logger.info(f"\nSaved {csv1_path}")

    concordance_rate = n_concordant / len(patients)

    # ── CSV 2: moments ──────────────────────────────────────────────────
    csv2_path = OUTPUT_DIR / "per_patient_moments.csv"
    with open(csv2_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['patient', 'M1_affine', 'M2_affine', 'SNR_affine',
                     'M1_bw', 'M2_bw', 'SNR_bw'])
        for p in patients:
            a = mom_ai[p]
            b = mom_bw[p]
            w.writerow([p,
                        f"{a['M1']:.4f}", f"{a['M2']:.4f}", f"{a['SNR']:.4f}",
                        f"{b['M1']:.4f}", f"{b['M2']:.4f}", f"{b['SNR']:.4f}"])
    logger.info(f"Saved {csv2_path}")

    # ── Figure: projection scatter ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(patients) * 0.4)),
                             sharey=True)

    for ax, (metric_name, res) in zip(axes,
                                       [('Affine-Invariant', res_ai),
                                        ('Bures-Wasserstein', res_bw)]):
        for i, p in enumerate(patients):
            scores = np.array(res[p]['scores'])
            y_true = np.array(res[p]['y_true'])
            jitter = np.random.RandomState(42).uniform(-0.2, 0.2, len(scores))

            ictal_mask = y_true == 1
            ax.scatter(scores[~ictal_mask], i + jitter[~ictal_mask],
                       c='#2196F3', alpha=0.4, s=12, label='interictal' if i == 0 else '')
            ax.scatter(scores[ictal_mask], i + jitter[ictal_mask],
                       c='#F44336', alpha=0.6, s=18, label='ictal' if i == 0 else '')

        ax.set_yticks(range(len(patients)))
        ax.set_yticklabels(patients, fontsize=7)
        ax.set_xlabel('LDA Score', fontsize=10)
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    fig_path = OUTPUT_DIR / "projection_scatter.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {fig_path}")

    # ── SUMMARY.md ──────────────────────────────────────────────────────
    n_strong_ai = sum(1 for p in patients if mom_ai[p]['SNR'] > SNR_THRESHOLD)
    n_strong_bw = sum(1 for p in patients if mom_bw[p]['SNR'] > SNR_THRESHOLD)
    mean_auc_ai = np.mean([res_ai[p]['auc'] for p in patients])
    mean_auc_bw = np.mean([res_bw[p]['auc'] for p in patients])

    if concordance_rate >= 0.85:
        verdict = "Polarity replicates under Bures-Wasserstein."
    elif concordance_rate >= 0.60:
        verdict = "Polarity partially replicates under Bures-Wasserstein."
    else:
        verdict = "Polarity does not replicate under Bures-Wasserstein."

    summary_lines = [
        "# BW-001: Bures-Wasserstein Polarity Replication Check",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Subjects:** {len(patients)} CHB-MIT patients",
        f"**Channels:** 8 ({', '.join(LOSO_CHANNELS)})",
        f"**Window:** 30s (10s fallback)",
        "",
        "## Four Numbers",
        "",
        f"- **Concordance rate:** {concordance_rate:.3f} ({n_concordant}/{len(patients)} patients match)",
        f"- **Strongly-polar patients (SNR > {SNR_THRESHOLD}):** affine={n_strong_ai}, BW={n_strong_bw}",
        f"- **Mean AUC (affine-invariant):** {mean_auc_ai:.4f}",
        f"- **Mean AUC (Bures-Wasserstein):** {mean_auc_bw:.4f}",
        "",
        f"## Verdict",
        "",
        f"**{verdict}**",
        "",
        "## Polarity Comparison",
        "",
        "| Patient | Pol (AI) | Pol (BW) | AUC (AI) | AUC (BW) | Match |",
        "|---------|----------|----------|----------|----------|-------|",
    ]

    for p in patients:
        pol_ai = res_ai[p]['oracle_polarity']
        pol_bw = res_bw[p]['oracle_polarity']
        match = "Y" if pol_ai == pol_bw else "**N**"
        summary_lines.append(
            f"| {p} | {pol_ai} | {pol_bw} | "
            f"{res_ai[p]['auc']:.3f} | {res_bw[p]['auc']:.3f} | {match} |"
        )

    summary_lines.extend([
        "",
        "## Moments (M1, M2, SNR)",
        "",
        "| Patient | M1(AI) | M2(AI) | SNR(AI) | M1(BW) | M2(BW) | SNR(BW) |",
        "|---------|--------|--------|---------|--------|--------|---------|",
    ])

    for p in patients:
        a, b = mom_ai[p], mom_bw[p]
        summary_lines.append(
            f"| {p} | {a['M1']:.3f} | {a['M2']:.3f} | {a['SNR']:.2f} | "
            f"{b['M1']:.3f} | {b['M2']:.3f} | {b['SNR']:.2f} |"
        )

    # Discordant patients detail
    discordant = [p for p in patients
                  if res_ai[p]['oracle_polarity'] != res_bw[p]['oracle_polarity']]
    if discordant:
        summary_lines.extend([
            "",
            "## Discordant Patients",
            "",
        ])
        for p in discordant:
            summary_lines.append(
                f"- **{p}**: AI={res_ai[p]['oracle_polarity']} (AUC={res_ai[p]['auc']:.3f}), "
                f"BW={res_bw[p]['oracle_polarity']} (AUC={res_bw[p]['auc']:.3f}), "
                f"delta={abs(res_ai[p]['auc'] - res_bw[p]['auc']):.3f}"
            )

    with open(OUTPUT_DIR / "SUMMARY.md", 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    logger.info(f"Saved SUMMARY.md")

    # ── Console summary ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BW-001 RESULTS")
    print("=" * 60)
    print(f"Concordance: {concordance_rate:.3f} ({n_concordant}/{len(patients)})")
    print(f"Strong polar (AI): {n_strong_ai}  (BW): {n_strong_bw}")
    print(f"Mean AUC (AI): {mean_auc_ai:.4f}  (BW): {mean_auc_bw:.4f}")
    print(f"\n>>> {verdict}")

    if discordant:
        print(f"\nDiscordant patients ({len(discordant)}):")
        for p in discordant:
            print(f"  {p}: AI={res_ai[p]['oracle_polarity']}({res_ai[p]['auc']:.3f}) "
                  f"BW={res_bw[p]['oracle_polarity']}({res_bw[p]['auc']:.3f})")


if __name__ == '__main__':
    main()
