#!/usr/bin/env python3
"""
Tier 3 Riemannian with MAX Sub-Window Aggregation LOSO CV

Computes Riemannian covariance at 1-second sub-window level (matching Tier 2's
temporal structure) then aggregates via MAX pooling. This isolates the feature
type variable from the temporal resolution variable.

Variants:
- MDM with MAX/MEAN aggregation
- TS+XGBoost with MAX/MEAN aggregation
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
from xgboost import XGBClassifier

from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace

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
# Configuration
# ============================================================================

LOSO_CHANNELS = [
    "FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
    "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"
]

N_BAGS = 5
SUB_WINDOW_SEC = 1.0
SUB_WINDOW_SAMPLES = int(SUB_WINDOW_SEC * FS)  # 256
N_SUB_WINDOWS = int(WINDOW_SEC / SUB_WINDOW_SEC)  # 30
MIN_VALID_SUB_WINDOWS = 15  # Minimum valid sub-windows to include segment

BASELINES = {
    'Tier2_MAX': 0.7234,
    'Tier3_MDM_30s': 0.5664,
    'Tier3_TS_XGB_30s': 0.5658,
}

DATA_ROOT = Path("H:/Data/PythonDNU/EEG/chbmit")


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
        return []
    return extract_segments_for_subject(subject_dir)


def load_segment_data(segment: EEGSegment, channels: List[str]) -> Optional[np.ndarray]:
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

    min_samples = int(WINDOW_SEC * fs * 0.5)
    if data.shape[1] < min_samples:
        return None

    data = preprocess_eeg(data, fs=int(fs))
    return data


def extract_subwindow_covariances(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract covariance matrices from 1-second sub-windows.

    Args:
        data: EEG data of shape (n_channels, n_samples) - 30 seconds

    Returns:
        Tuple of (covariances, valid_mask) where covariances has shape (n_valid, 8, 8)
    """
    n_channels, n_samples = data.shape
    expected_samples = int(WINDOW_SEC * FS)

    if n_samples < expected_samples:
        # Pad if needed
        padded = np.zeros((n_channels, expected_samples))
        padded[:, :n_samples] = data
        data = padded

    # Split into sub-windows
    sub_windows = []
    for i in range(N_SUB_WINDOWS):
        start = i * SUB_WINDOW_SAMPLES
        end = start + SUB_WINDOW_SAMPLES
        sub_windows.append(data[:, start:end])

    sub_windows = np.array(sub_windows)  # (30, 8, 256)

    # Compute covariances with lwf regularization
    try:
        cov_estimator = Covariances(estimator='lwf')
        covariances = cov_estimator.transform(sub_windows)  # (30, 8, 8)
    except Exception as e:
        logger.debug(f"Covariance estimation failed: {e}")
        return np.array([]), np.array([])

    # Validate each covariance matrix
    valid_mask = []
    valid_covs = []
    for cov in covariances:
        try:
            eigvals = np.linalg.eigvalsh(cov)
            if np.all(eigvals > 0) and not np.isnan(cov).any():
                valid_mask.append(True)
                valid_covs.append(cov)
            else:
                valid_mask.append(False)
        except:
            valid_mask.append(False)

    return np.array(valid_covs), np.array(valid_mask)


def load_subwindow_data_for_subject(
    subject_id: str,
    channels: List[str] = LOSO_CHANNELS,
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """
    Load all sub-window covariances for a subject.

    Returns:
        Tuple of (segment_covs, segment_labels, segment_n_valid)
        where segment_covs[i] has shape (n_valid_subwindows, 8, 8)
    """
    segments = load_segments_for_subject(subject_id)
    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}

    segment_covs = []
    segment_labels = []
    segment_n_valid = []

    for segment in segments:
        data = load_segment_data(segment, channels)
        if data is None:
            continue

        covs, valid_mask = extract_subwindow_covariances(data)

        if len(covs) >= MIN_VALID_SUB_WINDOWS:
            segment_covs.append(covs)
            segment_labels.append(label_map.get(segment.label, 0))
            segment_n_valid.append(len(covs))

    return segment_covs, segment_labels, segment_n_valid


# ============================================================================
# MDM with Aggregation
# ============================================================================

def run_mdm_max_loso(subject_data: Dict) -> Dict:
    """Run MDM classifier with MAX/MEAN sub-window aggregation."""
    results_per_subject = {}
    all_y_true_max = []
    all_y_prob_max = []
    all_y_true_mean = []
    all_y_prob_mean = []

    valid_subjects = list(subject_data.keys())

    for test_subject in valid_subjects:
        logger.info(f"  MDM testing on {test_subject}...")

        # Build training data: explode segments into sub-windows
        X_train_list = []
        y_train_list = []
        for train_subject in valid_subjects:
            if train_subject != test_subject:
                segment_covs, segment_labels, _ = subject_data[train_subject]
                for covs, label in zip(segment_covs, segment_labels):
                    X_train_list.append(covs)
                    y_train_list.extend([label] * len(covs))

        if len(X_train_list) == 0:
            continue

        X_train = np.vstack(X_train_list)
        y_train = np.array(y_train_list)

        # Check class balance
        unique_train = np.unique(y_train)
        if len(unique_train) < 2:
            logger.warning(f"    {test_subject}: Skipped (single training class)")
            continue

        min_per_class = min(np.sum(y_train == c) for c in unique_train)
        if min_per_class < 2:
            logger.warning(f"    {test_subject}: Skipped (< 2 samples per class)")
            continue

        # Test data
        test_segment_covs, test_segment_labels, _ = subject_data[test_subject]

        try:
            clf = MDM(metric='riemann')
            clf.fit(X_train, y_train)

            # Get probabilities for each sub-window, then aggregate per segment
            segment_probs_max = []
            segment_probs_mean = []
            segment_true = []

            for covs, label in zip(test_segment_covs, test_segment_labels):
                distances = clf.transform(covs)
                probs = np.exp(-distances) / np.exp(-distances).sum(axis=1, keepdims=True)
                sub_probs = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

                segment_probs_max.append(np.max(sub_probs))
                segment_probs_mean.append(np.mean(sub_probs))
                segment_true.append(label)

            y_true = np.array(segment_true)
            y_prob_max = np.array(segment_probs_max)
            y_prob_mean = np.array(segment_probs_mean)

            auc_max = roc_auc_score(y_true, y_prob_max) if len(np.unique(y_true)) > 1 else 0.5
            auc_mean = roc_auc_score(y_true, y_prob_mean) if len(np.unique(y_true)) > 1 else 0.5

        except Exception as e:
            logger.warning(f"    {test_subject}: MDM failed - {e}")
            auc_max = 0.5
            auc_mean = 0.5
            y_true = np.array(test_segment_labels)
            y_prob_max = np.full(len(y_true), 0.5)
            y_prob_mean = np.full(len(y_true), 0.5)

        results_per_subject[test_subject] = {
            'auc_max': auc_max,
            'auc_mean': auc_mean,
            'n_segments': len(test_segment_labels),
            'n_seizure': int(sum(test_segment_labels)),
            'n_non_seizure': int(len(test_segment_labels) - sum(test_segment_labels)),
        }

        all_y_true_max.extend(y_true)
        all_y_prob_max.extend(y_prob_max)
        all_y_true_mean.extend(y_true)
        all_y_prob_mean.extend(y_prob_mean)

        logger.info(f"    {test_subject}: AUC MAX={auc_max:.4f}, MEAN={auc_mean:.4f}")

    # Overall AUC
    try:
        overall_auc_max = roc_auc_score(all_y_true_max, all_y_prob_max)
    except:
        overall_auc_max = 0.5
    try:
        overall_auc_mean = roc_auc_score(all_y_true_mean, all_y_prob_mean)
    except:
        overall_auc_mean = 0.5

    aucs_max = [r['auc_max'] for r in results_per_subject.values()]
    aucs_mean = [r['auc_mean'] for r in results_per_subject.values()]

    return {
        'overall_auc_max': overall_auc_max,
        'overall_auc_mean': overall_auc_mean,
        'mean_auc_max': np.mean(aucs_max) if aucs_max else 0.5,
        'mean_auc_mean': np.mean(aucs_mean) if aucs_mean else 0.5,
        'std_auc_max': np.std(aucs_max) if aucs_max else 0.0,
        'std_auc_mean': np.std(aucs_mean) if aucs_mean else 0.0,
        'per_subject': results_per_subject,
    }


# ============================================================================
# TS+XGBoost with Aggregation
# ============================================================================

def run_ts_xgb_max_loso(subject_data: Dict) -> Dict:
    """Run Tangent Space + XGBoost with MAX/MEAN sub-window aggregation."""
    results_per_subject = {}
    all_y_true_max = []
    all_y_prob_max = []
    all_y_true_mean = []
    all_y_prob_mean = []

    valid_subjects = list(subject_data.keys())

    for test_subject in valid_subjects:
        logger.info(f"  TS+XGB testing on {test_subject}...")

        # Build training data
        X_train_list = []
        y_train_list = []
        for train_subject in valid_subjects:
            if train_subject != test_subject:
                segment_covs, segment_labels, _ = subject_data[train_subject]
                for covs, label in zip(segment_covs, segment_labels):
                    X_train_list.append(covs)
                    y_train_list.extend([label] * len(covs))

        if len(X_train_list) == 0:
            continue

        X_train_cov = np.vstack(X_train_list)
        y_train = np.array(y_train_list)

        test_segment_covs, test_segment_labels, _ = subject_data[test_subject]

        try:
            # Project to tangent space
            ts = TangentSpace(metric='riemann')
            ts.fit(X_train_cov, y_train)
            X_train = ts.transform(X_train_cov)
            X_train = np.nan_to_num(X_train)

            # Train 5-bag ensemble
            models = []
            for bag_idx in range(N_BAGS):
                rng = np.random.RandomState(42 + bag_idx)
                idx = rng.choice(len(X_train), size=len(X_train), replace=True)
                model = make_xgb_model(seed=42 + bag_idx)
                model.fit(X_train[idx], y_train[idx])
                models.append(model)

            # Test: aggregate sub-window probabilities per segment
            segment_probs_max = []
            segment_probs_mean = []
            segment_true = []

            for covs, label in zip(test_segment_covs, test_segment_labels):
                X_test_cov = covs
                X_test = ts.transform(X_test_cov)
                X_test = np.nan_to_num(X_test)

                # Average across bags
                bag_probs = []
                for model in models:
                    bag_probs.append(model.predict_proba(X_test)[:, 1])
                sub_probs = np.mean(bag_probs, axis=0)

                segment_probs_max.append(np.max(sub_probs))
                segment_probs_mean.append(np.mean(sub_probs))
                segment_true.append(label)

            y_true = np.array(segment_true)
            y_prob_max = np.array(segment_probs_max)
            y_prob_mean = np.array(segment_probs_mean)

            auc_max = roc_auc_score(y_true, y_prob_max) if len(np.unique(y_true)) > 1 else 0.5
            auc_mean = roc_auc_score(y_true, y_prob_mean) if len(np.unique(y_true)) > 1 else 0.5

        except Exception as e:
            logger.warning(f"    {test_subject}: TS+XGB failed - {e}")
            auc_max = 0.5
            auc_mean = 0.5
            y_true = np.array(test_segment_labels)
            y_prob_max = np.full(len(y_true), 0.5)
            y_prob_mean = np.full(len(y_true), 0.5)

        results_per_subject[test_subject] = {
            'auc_max': auc_max,
            'auc_mean': auc_mean,
            'n_segments': len(test_segment_labels),
            'n_seizure': int(sum(test_segment_labels)),
            'n_non_seizure': int(len(test_segment_labels) - sum(test_segment_labels)),
        }

        all_y_true_max.extend(y_true)
        all_y_prob_max.extend(y_prob_max)
        all_y_true_mean.extend(y_true)
        all_y_prob_mean.extend(y_prob_mean)

        logger.info(f"    {test_subject}: AUC MAX={auc_max:.4f}, MEAN={auc_mean:.4f}")

    # Overall AUC
    try:
        overall_auc_max = roc_auc_score(all_y_true_max, all_y_prob_max)
    except:
        overall_auc_max = 0.5
    try:
        overall_auc_mean = roc_auc_score(all_y_true_mean, all_y_prob_mean)
    except:
        overall_auc_mean = 0.5

    aucs_max = [r['auc_max'] for r in results_per_subject.values()]
    aucs_mean = [r['auc_mean'] for r in results_per_subject.values()]

    return {
        'overall_auc_max': overall_auc_max,
        'overall_auc_mean': overall_auc_mean,
        'mean_auc_max': np.mean(aucs_max) if aucs_max else 0.5,
        'mean_auc_mean': np.mean(aucs_mean) if aucs_mean else 0.5,
        'std_auc_max': np.std(aucs_max) if aucs_max else 0.0,
        'std_auc_mean': np.std(aucs_mean) if aucs_mean else 0.0,
        'per_subject': results_per_subject,
    }


# ============================================================================
# Output
# ============================================================================

def update_comparison_markdown(mdm_results: Dict, ts_xgb_results: Dict) -> str:
    """Update comparison markdown with new rows."""

    md = """# LOSO Cross-Patient CV Results — Final Comparison

## All Tiers Comparison

| Classifier                | Channels | Window | Overall AUC | Delta vs Paper |
|---------------------------|----------|--------|-------------|----------------|
| XGBoost-8ch (paper)       | 8        | 30s    | 0.5989      | baseline       |
| XGBoost-8ch-V2 (paper)    | 8        | 30s    | 0.6253      | baseline       |
| Tier 1 FFT Bands          | 8        | 30s    | 0.5869      | -0.038         |
| Tier 2 Corr-Eig MEAN      | 8        | 30s    | 0.5674      | -0.058         |
| Tier 2 Corr-Eig MAX       | 8        | 30s    | 0.7234      | +0.098         |
| Tier 3 MDM (30s)          | 8        | 30s    | 0.5664      | -0.059         |
| Tier 3 TS+LDA (30s)       | 8        | 30s    | 0.5686      | -0.057         |
| Tier 3 TS+XGBoost (30s)   | 8        | 30s    | 0.5658      | -0.060         |

## Sub-Window Aggregation Comparison

| Classifier                  | Window unit | Aggregation | Overall AUC | Delta vs Tier2 MAX |
|-----------------------------|-------------|-------------|-------------|--------------------|
| Tier 2 Corr-Eig MAX         | 1s          | MAX         | 0.7234      | baseline           |
| Tier 3 MDM (30s window)     | 30s         | none        | 0.5664      | -0.157             |
| Tier 3 TS+XGBoost (30s)     | 30s         | none        | 0.5658      | -0.158             |
"""

    # Add new results
    mdm_max = mdm_results['overall_auc_max']
    mdm_mean = mdm_results['overall_auc_mean']
    ts_xgb_max = ts_xgb_results['overall_auc_max']
    ts_xgb_mean = ts_xgb_results['overall_auc_mean']

    tier2_max = BASELINES['Tier2_MAX']

    md += f"| Tier 3 MDM MAX              | 1s          | MAX         | {mdm_max:.4f}      | {mdm_max - tier2_max:+.3f}             |\n"
    md += f"| Tier 3 MDM MEAN             | 1s          | MEAN        | {mdm_mean:.4f}      | {mdm_mean - tier2_max:+.3f}             |\n"
    md += f"| Tier 3 TS+XGBoost MAX       | 1s          | MAX         | {ts_xgb_max:.4f}      | {ts_xgb_max - tier2_max:+.3f}             |\n"
    md += f"| Tier 3 TS+XGBoost MEAN      | 1s          | MEAN        | {ts_xgb_mean:.4f}      | {ts_xgb_mean - tier2_max:+.3f}             |\n"

    md += f"""
## Configuration

- **Channels**: 8 (matching paper)
- **Segment Window**: 30 seconds
- **Sub-window**: 1 second (30 per segment)
- **Covariance Estimator**: Ledoit-Wolf regularization
- **XGBoost**: 5-bag ensemble, same hyperparameters as paper

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return md


def main():
    """Run Tier 3 MAX sub-window aggregation benchmarks."""

    output_dir = Path(__file__).parent.parent / 'results' / 'loso_cv'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Tier 3 Riemannian MAX Sub-Window Aggregation LOSO CV")
    logger.info("=" * 60)

    if not DATA_ROOT.exists():
        logger.error(f"Data directory not found: {DATA_ROOT}")
        return

    all_subjects = sorted([
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith('chb')
        and d.name not in EXCLUDE_SUBJECTS
    ])

    logger.info(f"Found {len(all_subjects)} subjects")

    # Load sub-window covariances for all subjects
    logger.info("\nLoading sub-window covariances...")
    subject_data = {}

    for subject_id in all_subjects:
        logger.info(f"  Processing {subject_id}...")
        segment_covs, segment_labels, segment_n_valid = load_subwindow_data_for_subject(subject_id)

        if len(segment_covs) > 0 and len(set(segment_labels)) == 2:
            subject_data[subject_id] = (segment_covs, segment_labels, segment_n_valid)
            n_pos = sum(segment_labels)
            n_neg = len(segment_labels) - n_pos
            total_subwindows = sum(segment_n_valid)
            logger.info(f"    {subject_id}: {len(segment_covs)} segments, "
                       f"{n_pos} seizure, {n_neg} non-seizure, {total_subwindows} sub-windows")
        else:
            if len(segment_covs) > 0:
                logger.warning(f"    {subject_id}: Skipped (single class)")
            else:
                logger.warning(f"    {subject_id}: Skipped (no valid segments)")

    logger.info(f"\nValid subjects: {len(subject_data)}")

    # Run MDM
    logger.info("\n" + "=" * 60)
    logger.info("TIER 3: MDM with MAX/MEAN Sub-Window Aggregation")
    logger.info("=" * 60)
    mdm_results = run_mdm_max_loso(subject_data)
    mdm_results['method'] = 'MDM'
    mdm_results['n_channels'] = len(LOSO_CHANNELS)
    mdm_results['channels'] = LOSO_CHANNELS
    mdm_results['sub_window_sec'] = SUB_WINDOW_SEC
    mdm_results['timestamp'] = datetime.now().isoformat()

    with open(output_dir / 'tier3_mdm_max_loso_results.json', 'w', encoding='utf-8') as f:
        json.dump(mdm_results, f, indent=2)
    logger.info(f"MDM Overall AUC: MAX={mdm_results['overall_auc_max']:.4f}, MEAN={mdm_results['overall_auc_mean']:.4f}")

    # Run TS+XGBoost
    logger.info("\n" + "=" * 60)
    logger.info("TIER 3: TS+XGBoost with MAX/MEAN Sub-Window Aggregation")
    logger.info("=" * 60)
    ts_xgb_results = run_ts_xgb_max_loso(subject_data)
    ts_xgb_results['method'] = 'TS+XGBoost'
    ts_xgb_results['n_channels'] = len(LOSO_CHANNELS)
    ts_xgb_results['channels'] = LOSO_CHANNELS
    ts_xgb_results['sub_window_sec'] = SUB_WINDOW_SEC
    ts_xgb_results['timestamp'] = datetime.now().isoformat()

    with open(output_dir / 'tier3_ts_xgb_max_loso_results.json', 'w', encoding='utf-8') as f:
        json.dump(ts_xgb_results, f, indent=2)
    logger.info(f"TS+XGB Overall AUC: MAX={ts_xgb_results['overall_auc_max']:.4f}, MEAN={ts_xgb_results['overall_auc_mean']:.4f}")

    # Update comparison markdown
    comparison_md = update_comparison_markdown(mdm_results, ts_xgb_results)
    with open(output_dir / 'loso_comparison_final.md', 'w', encoding='utf-8') as f:
        f.write(comparison_md)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tier 2 MAX (target):     {BASELINES['Tier2_MAX']:.4f}")
    logger.info(f"Tier 3 MDM MAX:          {mdm_results['overall_auc_max']:.4f} "
                f"(Delta = {mdm_results['overall_auc_max'] - BASELINES['Tier2_MAX']:+.4f})")
    logger.info(f"Tier 3 MDM MEAN:         {mdm_results['overall_auc_mean']:.4f} "
                f"(Delta = {mdm_results['overall_auc_mean'] - BASELINES['Tier2_MAX']:+.4f})")
    logger.info(f"Tier 3 TS+XGB MAX:       {ts_xgb_results['overall_auc_max']:.4f} "
                f"(Delta = {ts_xgb_results['overall_auc_max'] - BASELINES['Tier2_MAX']:+.4f})")
    logger.info(f"Tier 3 TS+XGB MEAN:      {ts_xgb_results['overall_auc_mean']:.4f} "
                f"(Delta = {ts_xgb_results['overall_auc_mean'] - BASELINES['Tier2_MAX']:+.4f})")
    logger.info(f"\nResults saved to: {output_dir}")

    return {
        'mdm': mdm_results,
        'ts_xgb': ts_xgb_results,
    }


if __name__ == '__main__':
    main()
