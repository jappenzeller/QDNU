#!/usr/bin/env python3
"""
Tier 3 Riemannian LOSO Cross-Patient CV Benchmark

This script evaluates Riemannian geometry-based classifiers using Leave-One-Subject-Out
cross-validation to compare against Tier 1, Tier 2, and paper baselines.

Target to beat: Tier 2 MAX pooling at 0.7234 AUC
Paper baseline: XGBoost-8ch-V2 at 0.6253 AUC

Variants:
- MDM (Minimum Distance to Mean): Classifies by nearest Riemannian mean
- TS+LDA: Tangent space projection + Linear Discriminant Analysis
- TS+XGBoost: Tangent space projection + XGBoost 5-bag ensemble
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Pyriemann imports
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sagemaker.train_chbmit import (
    EXCLUDE_SUBJECTS,
    WINDOW_SEC,
    FS,
    EEGSegment,
    preprocess_eeg,
    read_edf_segment,
    extract_segments_for_subject,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration - matches tier1_tier2_loso_cv.py exactly
# ============================================================================

# 8-channel subset used in the paper
LOSO_CHANNELS = [
    "FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
    "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"
]

# XGBoost hyperparameters (same as paper)
N_BAGS = 5

# Paper and Tier 2 baselines for comparison
BASELINES = {
    'XGBoost-8ch-V1': 0.5989,
    'XGBoost-8ch-V2': 0.6253,
    'Tier1_FFT': 0.5869,
    'Tier2_MEAN': 0.5674,
    'Tier2_MAX': 0.7234,
}

# Short-seizure fallback window
FALLBACK_WINDOW_SEC = 10.0

# CHB-MIT data path
DATA_ROOT = Path("H:/Data/PythonDNU/EEG/chbmit")


# ============================================================================
# XGBoost Model (exact hyperparameters from paper)
# ============================================================================

def make_xgb_model(seed: int = 42) -> XGBClassifier:
    """Create XGBoost classifier with paper's exact hyperparameters."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        eval_metric='logloss',
        n_jobs=-1,
    )


# ============================================================================
# Data Loading
# ============================================================================

def load_segments_for_subject(subject_id: str) -> List[EEGSegment]:
    """Load segment metadata for a subject."""
    subject_dir = DATA_ROOT / subject_id
    if not subject_dir.exists():
        logger.warning(f"Subject directory not found: {subject_dir}")
        return []
    return extract_segments_for_subject(subject_dir)


def load_segment_data(segment: EEGSegment, channels: List[str],
                      window_sec: float = WINDOW_SEC) -> Optional[np.ndarray]:
    """Load and preprocess EEG data for a segment."""
    edf_path = DATA_ROOT / segment.subject / segment.source_file
    if not edf_path.exists():
        return None

    data, fs = read_edf_segment(
        str(edf_path),
        segment.start_sec,
        segment.end_sec,
        target_channels=channels
    )

    if data is None or data.size == 0:
        return None

    # Minimum duration check
    min_samples = int(window_sec * fs * 0.5)
    if data.shape[1] < min_samples:
        return None

    # Preprocess - but NO per-channel z-score normalization
    # (normalization is implicit in Riemannian metric)
    data = preprocess_eeg(data, fs=int(fs))

    return data


def load_covariances_for_subject(
    subject_id: str,
    channels: List[str] = LOSO_CHANNELS,
    window_sec: float = WINDOW_SEC,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load subject data and compute covariance matrices.

    Args:
        subject_id: Subject ID
        channels: List of channel names to use
        window_sec: Window size in seconds

    Returns:
        Tuple of (covariances, labels, actual_window_sec)
    """
    segments = load_segments_for_subject(subject_id)

    if len(segments) == 0:
        return np.array([]), np.array([]), window_sec

    # Label mapping
    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}

    # Collect raw windows
    windows_list = []
    labels_list = []

    for segment in segments:
        try:
            data = load_segment_data(segment, channels, window_sec)
            if data is None:
                continue

            # Check we have full window
            expected_samples = int(window_sec * FS)
            if data.shape[1] >= expected_samples:
                windows_list.append(data[:, :expected_samples])
                labels_list.append(label_map.get(segment.label, 0))
        except Exception as e:
            logger.debug(f"Error loading segment: {e}")
            continue

    if len(windows_list) == 0:
        return np.array([]), np.array([]), window_sec

    # Stack windows: (n_windows, n_channels, n_samples)
    windows = np.array(windows_list)
    labels = np.array(labels_list)

    # Compute regularized covariance matrices using Ledoit-Wolf
    try:
        cov_estimator = Covariances(estimator='lwf')
        covariances = cov_estimator.transform(windows)
    except Exception as e:
        logger.warning(f"Covariance estimation failed for {subject_id}: {e}")
        return np.array([]), np.array([]), window_sec

    # Check for valid SPD matrices
    valid_mask = []
    for i, cov in enumerate(covariances):
        try:
            # Check positive definite
            eigvals = np.linalg.eigvalsh(cov)
            if np.all(eigvals > 0) and not np.isnan(cov).any():
                valid_mask.append(True)
            else:
                valid_mask.append(False)
        except:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    covariances = covariances[valid_mask]
    labels = labels[valid_mask]

    return covariances, labels, window_sec


def load_covariances_with_fallback(
    subject_id: str,
    channels: List[str] = LOSO_CHANNELS,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Load covariances with fallback to shorter windows if needed.

    Returns:
        Tuple of (covariances, labels, actual_window_sec_used)
    """
    # Try 30-second windows first
    covs, labels, window_used = load_covariances_for_subject(
        subject_id, channels, WINDOW_SEC
    )

    # Check if we have both classes
    if len(covs) > 0 and len(np.unique(labels)) == 2:
        return covs, labels, window_used

    # If no ictal samples or single class, try 10-second windows
    logger.info(f"  {subject_id}: Trying {FALLBACK_WINDOW_SEC}s fallback...")
    covs, labels, window_used = load_covariances_for_subject(
        subject_id, channels, FALLBACK_WINDOW_SEC
    )

    return covs, labels, FALLBACK_WINDOW_SEC


# ============================================================================
# Classifiers
# ============================================================================

def run_mdm_loso(subject_data: Dict) -> Dict:
    """Run MDM classifier with LOSO CV."""
    results_per_subject = {}
    all_y_true = []
    all_y_prob = []

    valid_subjects = list(subject_data.keys())

    for test_subject in valid_subjects:
        logger.info(f"  MDM testing on {test_subject}...")

        # Train data
        X_train_list = []
        y_train_list = []
        for train_subject in valid_subjects:
            if train_subject != test_subject:
                covs, labels, _ = subject_data[train_subject]
                X_train_list.append(covs)
                y_train_list.append(labels)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Test data
        X_test, y_test, _ = subject_data[test_subject]

        # Check MDM requirements
        unique_train = np.unique(y_train)
        if len(unique_train) < 2:
            logger.warning(f"    {test_subject}: Skipped (single training class)")
            continue

        # Check minimum samples per class for mean computation
        min_per_class = min(np.sum(y_train == c) for c in unique_train)
        if min_per_class < 2:
            logger.warning(f"    {test_subject}: Skipped (< 2 samples per class)")
            continue

        try:
            clf = MDM(metric='riemann')
            clf.fit(X_train, y_train)

            # Get prediction probabilities
            # MDM uses distance-based prediction, convert to probability
            distances = clf.transform(X_test)
            # Probability based on relative distance (closer = higher prob for that class)
            probs = np.exp(-distances) / np.exp(-distances).sum(axis=1, keepdims=True)
            y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

            auc = roc_auc_score(y_test, y_prob)
        except Exception as e:
            logger.warning(f"    {test_subject}: MDM failed - {e}")
            auc = 0.5
            y_prob = np.full(len(y_test), 0.5)

        results_per_subject[test_subject] = {
            'auc': auc,
            'n_samples': len(y_test),
            'n_seizure': int(sum(y_test)),
            'n_non_seizure': int(len(y_test) - sum(y_test)),
        }

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

        logger.info(f"    {test_subject}: AUC = {auc:.4f}")

    # Overall AUC
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except:
        overall_auc = 0.5

    aucs = [r['auc'] for r in results_per_subject.values()]
    return {
        'overall_auc': overall_auc,
        'mean_auc': np.mean(aucs) if aucs else 0.5,
        'std_auc': np.std(aucs) if aucs else 0.0,
        'per_subject': results_per_subject,
    }


def run_ts_lda_loso(subject_data: Dict) -> Dict:
    """Run Tangent Space + LDA classifier with LOSO CV."""
    results_per_subject = {}
    all_y_true = []
    all_y_prob = []

    valid_subjects = list(subject_data.keys())

    for test_subject in valid_subjects:
        logger.info(f"  TS+LDA testing on {test_subject}...")

        # Train data
        X_train_list = []
        y_train_list = []
        for train_subject in valid_subjects:
            if train_subject != test_subject:
                covs, labels, _ = subject_data[train_subject]
                X_train_list.append(covs)
                y_train_list.append(labels)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Test data
        X_test, y_test, _ = subject_data[test_subject]

        try:
            clf = Pipeline([
                ('ts', TangentSpace(metric='riemann')),
                ('lda', LinearDiscriminantAnalysis())
            ])
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        except Exception as e:
            logger.warning(f"    {test_subject}: TS+LDA failed - {e}")
            auc = 0.5
            y_prob = np.full(len(y_test), 0.5)

        results_per_subject[test_subject] = {
            'auc': auc,
            'n_samples': len(y_test),
            'n_seizure': int(sum(y_test)),
            'n_non_seizure': int(len(y_test) - sum(y_test)),
        }

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

        logger.info(f"    {test_subject}: AUC = {auc:.4f}")

    # Overall AUC
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except:
        overall_auc = 0.5

    aucs = [r['auc'] for r in results_per_subject.values()]
    return {
        'overall_auc': overall_auc,
        'mean_auc': np.mean(aucs) if aucs else 0.5,
        'std_auc': np.std(aucs) if aucs else 0.0,
        'per_subject': results_per_subject,
    }


def run_ts_xgb_loso(subject_data: Dict) -> Dict:
    """Run Tangent Space + XGBoost (5-bag ensemble) with LOSO CV."""
    results_per_subject = {}
    all_y_true = []
    all_y_prob = []

    valid_subjects = list(subject_data.keys())

    for test_subject in valid_subjects:
        logger.info(f"  TS+XGB testing on {test_subject}...")

        # Train data
        X_train_list = []
        y_train_list = []
        for train_subject in valid_subjects:
            if train_subject != test_subject:
                covs, labels, _ = subject_data[train_subject]
                X_train_list.append(covs)
                y_train_list.append(labels)

        X_train_cov = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Test data
        X_test_cov, y_test, _ = subject_data[test_subject]

        try:
            # Project to tangent space
            ts = TangentSpace(metric='riemann')
            ts.fit(X_train_cov, y_train)
            X_train = ts.transform(X_train_cov)
            X_test = ts.transform(X_test_cov)

            # Handle NaN
            X_train = np.nan_to_num(X_train)
            X_test = np.nan_to_num(X_test)

            # Train 5-bag ensemble with bootstrap sampling
            y_probs = []
            for bag_idx in range(N_BAGS):
                rng = np.random.RandomState(42 + bag_idx)
                idx = rng.choice(len(X_train), size=len(X_train), replace=True)
                model = make_xgb_model(seed=42 + bag_idx)
                model.fit(X_train[idx], y_train[idx])
                y_prob = model.predict_proba(X_test)[:, 1]
                y_probs.append(y_prob)

            y_prob_mean = np.mean(y_probs, axis=0)
            auc = roc_auc_score(y_test, y_prob_mean)
        except Exception as e:
            logger.warning(f"    {test_subject}: TS+XGB failed - {e}")
            auc = 0.5
            y_prob_mean = np.full(len(y_test), 0.5)

        results_per_subject[test_subject] = {
            'auc': auc,
            'n_samples': len(y_test),
            'n_seizure': int(sum(y_test)),
            'n_non_seizure': int(len(y_test) - sum(y_test)),
        }

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob_mean)

        logger.info(f"    {test_subject}: AUC = {auc:.4f}")

    # Overall AUC
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except:
        overall_auc = 0.5

    aucs = [r['auc'] for r in results_per_subject.values()]
    return {
        'overall_auc': overall_auc,
        'mean_auc': np.mean(aucs) if aucs else 0.5,
        'std_auc': np.std(aucs) if aucs else 0.0,
        'per_subject': results_per_subject,
    }


# ============================================================================
# Main
# ============================================================================

def generate_final_comparison(mdm_results: Dict, ts_lda_results: Dict,
                               ts_xgb_results: Dict) -> str:
    """Generate final comparison markdown with all tiers."""

    md = """# LOSO Cross-Patient CV Results — Final Comparison

## All Tiers Comparison

| Classifier                | Channels | Window | Overall AUC | Delta vs Paper |
|---------------------------|----------|--------|-------------|----------------|
| XGBoost-8ch (paper)       | 8        | 30s    | 0.5989      | baseline       |
| XGBoost-8ch-V2 (paper)    | 8        | 30s    | 0.6253      | baseline       |
| Tier 1 FFT Bands          | 8        | 30s    | 0.5869      | -0.038         |
| Tier 2 Corr-Eig MEAN      | 8        | 30s    | 0.5674      | -0.058         |
| Tier 2 Corr-Eig MAX       | 8        | 30s    | 0.7234      | +0.098         |
"""

    # Add Tier 3 results
    for name, results in [
        ('Tier 3 MDM', mdm_results),
        ('Tier 3 TS+LDA', ts_lda_results),
        ('Tier 3 TS+XGBoost', ts_xgb_results),
    ]:
        auc = results['overall_auc']
        delta = auc - BASELINES['XGBoost-8ch-V2']
        md += f"| {name:<25} | 8        | 30s    | {auc:.4f}      | {delta:+.3f}          |\n"

    # Determine best method
    all_results = {
        'Tier 2 MAX': BASELINES['Tier2_MAX'],
        'Tier 3 MDM': mdm_results['overall_auc'],
        'Tier 3 TS+LDA': ts_lda_results['overall_auc'],
        'Tier 3 TS+XGBoost': ts_xgb_results['overall_auc'],
    }
    best_method = max(all_results, key=all_results.get)
    best_auc = all_results[best_method]

    # Tier 3 vs Tier 2 MAX comparison
    tier3_best = max(mdm_results['overall_auc'], ts_lda_results['overall_auc'],
                     ts_xgb_results['overall_auc'])
    tier3_best_name = 'MDM' if tier3_best == mdm_results['overall_auc'] else \
                      'TS+LDA' if tier3_best == ts_lda_results['overall_auc'] else 'TS+XGBoost'

    tier2_max = BASELINES['Tier2_MAX']
    tier3_vs_tier2 = tier3_best - tier2_max

    md += f"""
## Conclusion

**Best Tier 3 variant:** {tier3_best_name} with AUC = {tier3_best:.4f}

**Tier 3 vs Tier 2 MAX (0.7234):**
"""

    if tier3_vs_tier2 > 0:
        md += f"- Tier 3 {tier3_best_name} BEATS Tier 2 MAX by +{tier3_vs_tier2:.4f}\n"
    elif tier3_vs_tier2 < 0:
        md += f"- Tier 3 {tier3_best_name} is BELOW Tier 2 MAX by {tier3_vs_tier2:.4f}\n"
    else:
        md += f"- Tier 3 {tier3_best_name} EQUALS Tier 2 MAX\n"

    md += f"""
**Strongest classical baseline overall:** {best_method} at {best_auc:.4f} AUC

**Riemannian SPD geometry assessment:**
"""

    if tier3_best > tier2_max:
        md += """- Riemannian geometry DOES provide measurable cross-patient lift over correlation eigenvalues
- The SPD manifold structure captures additional discriminative information
- This supports the geometric argument for quantum approaches operating on this manifold
"""
    else:
        md += """- Riemannian geometry does NOT outperform correlation eigenvalue features cross-patient
- The eigenvalue summary (Tier 2 MAX) already captures the discriminative information
- The geometric argument for quantum approaches needs reexamination — simpler features may suffice
"""

    md += f"""
## Configuration

- **Channels**: 8 (matching paper)
- **Window**: 30 seconds (with 10s fallback for short seizures)
- **CV Method**: Leave-One-Subject-Out (LOSO)
- **Covariance Estimator**: Ledoit-Wolf regularization
- **XGBoost**: 5-bag ensemble, same hyperparameters as paper

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return md


def main():
    """Run all Tier 3 benchmarks and save results."""

    output_dir = Path(__file__).parent.parent / 'results' / 'loso_cv'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Tier 3 Riemannian LOSO Cross-Patient CV Benchmark")
    logger.info("=" * 60)

    # Get all subjects
    if not DATA_ROOT.exists():
        logger.error(f"Data directory not found: {DATA_ROOT}")
        return

    all_subjects = sorted([
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith('chb')
        and d.name not in EXCLUDE_SUBJECTS
    ])

    logger.info(f"Found {len(all_subjects)} subjects")

    # Load covariances for all subjects
    logger.info("\nLoading and computing covariance matrices...")
    subject_data = {}
    window_sec_used = {}

    for subject_id in all_subjects:
        logger.info(f"  Processing {subject_id}...")
        covs, labels, window_used = load_covariances_with_fallback(subject_id)

        if len(covs) > 0 and len(np.unique(labels)) == 2:
            subject_data[subject_id] = (covs, labels, window_used)
            window_sec_used[subject_id] = window_used
            n_pos = int(sum(labels))
            n_neg = len(labels) - n_pos
            logger.info(f"    {subject_id}: {len(covs)} matrices, {n_pos} seizure, "
                       f"{n_neg} non-seizure (window={window_used}s)")
        else:
            if len(covs) > 0:
                logger.warning(f"    {subject_id}: Skipped (single class)")
            else:
                logger.warning(f"    {subject_id}: Skipped (no valid covariances)")

    logger.info(f"\nValid subjects: {len(subject_data)}")

    # Run MDM
    logger.info("\n" + "=" * 60)
    logger.info("TIER 3: MDM (Minimum Distance to Mean)")
    logger.info("=" * 60)
    mdm_results = run_mdm_loso(subject_data)
    mdm_results['method'] = 'MDM'
    mdm_results['n_channels'] = len(LOSO_CHANNELS)
    mdm_results['channels'] = LOSO_CHANNELS
    mdm_results['window_sec_per_subject'] = window_sec_used
    mdm_results['timestamp'] = datetime.now().isoformat()

    with open(output_dir / 'tier3_mdm_loso_results.json', 'w', encoding='utf-8') as f:
        json.dump(mdm_results, f, indent=2)
    logger.info(f"MDM Overall AUC: {mdm_results['overall_auc']:.4f}")

    # Run TS+LDA
    logger.info("\n" + "=" * 60)
    logger.info("TIER 3: Tangent Space + LDA")
    logger.info("=" * 60)
    ts_lda_results = run_ts_lda_loso(subject_data)
    ts_lda_results['method'] = 'TS+LDA'
    ts_lda_results['n_channels'] = len(LOSO_CHANNELS)
    ts_lda_results['channels'] = LOSO_CHANNELS
    ts_lda_results['window_sec_per_subject'] = window_sec_used
    ts_lda_results['timestamp'] = datetime.now().isoformat()

    with open(output_dir / 'tier3_ts_lda_loso_results.json', 'w', encoding='utf-8') as f:
        json.dump(ts_lda_results, f, indent=2)
    logger.info(f"TS+LDA Overall AUC: {ts_lda_results['overall_auc']:.4f}")

    # Run TS+XGBoost
    logger.info("\n" + "=" * 60)
    logger.info("TIER 3: Tangent Space + XGBoost (5-bag)")
    logger.info("=" * 60)
    ts_xgb_results = run_ts_xgb_loso(subject_data)
    ts_xgb_results['method'] = 'TS+XGBoost'
    ts_xgb_results['n_channels'] = len(LOSO_CHANNELS)
    ts_xgb_results['channels'] = LOSO_CHANNELS
    ts_xgb_results['window_sec_per_subject'] = window_sec_used
    ts_xgb_results['timestamp'] = datetime.now().isoformat()

    with open(output_dir / 'tier3_ts_xgb_loso_results.json', 'w', encoding='utf-8') as f:
        json.dump(ts_xgb_results, f, indent=2)
    logger.info(f"TS+XGBoost Overall AUC: {ts_xgb_results['overall_auc']:.4f}")

    # Generate final comparison
    comparison_md = generate_final_comparison(mdm_results, ts_lda_results, ts_xgb_results)
    with open(output_dir / 'loso_comparison_final.md', 'w', encoding='utf-8') as f:
        f.write(comparison_md)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Paper XGBoost-8ch-V2:  {BASELINES['XGBoost-8ch-V2']:.4f}")
    logger.info(f"Tier 2 MAX (target):   {BASELINES['Tier2_MAX']:.4f}")
    logger.info(f"Tier 3 MDM:            {mdm_results['overall_auc']:.4f} "
                f"(Delta = {mdm_results['overall_auc'] - BASELINES['Tier2_MAX']:+.4f})")
    logger.info(f"Tier 3 TS+LDA:         {ts_lda_results['overall_auc']:.4f} "
                f"(Delta = {ts_lda_results['overall_auc'] - BASELINES['Tier2_MAX']:+.4f})")
    logger.info(f"Tier 3 TS+XGBoost:     {ts_xgb_results['overall_auc']:.4f} "
                f"(Delta = {ts_xgb_results['overall_auc'] - BASELINES['Tier2_MAX']:+.4f})")
    logger.info(f"\nResults saved to: {output_dir}")

    return {
        'mdm': mdm_results,
        'ts_lda': ts_lda_results,
        'ts_xgb': ts_xgb_results,
    }


if __name__ == '__main__':
    main()
