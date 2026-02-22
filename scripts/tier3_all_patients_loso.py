#!/usr/bin/env python3
"""
Run Tier 3 Pipeline on All Eligible CHB-MIT Patients (B-003)

Runs the Tier 3 combined features (log-FFT + CC eigenvalues) with XGBoost
and LOSO cross-validation on all eligible patients.

Output: results/tier3_all_patients_loso.json

Reference: B-003 from research prompts
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

# Add parent directory to path
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

from scripts.features.cc_freq_encoding import extract_cc_freq
from scripts.features.cc_time_encoding import extract_cc_time
from scripts.features.log_fft_encoding import extract_log_fft_per_band

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
DATA_ROOT = Path("H:/Data/PythonDNU/EEG/chbmit")


# ============================================================================
# XGBoost Model
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
# Feature Extraction (Tier 3 Combined)
# ============================================================================

def extract_combined_features(window: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Extract combined features: [log_fft_bands | cc_freq_eigenvalues | cc_time_eigenvalues]
    """
    # Log FFT band power (8 channels x 5 bands = 40 features)
    log_fft_result = extract_log_fft_per_band(window, fs=fs)
    log_fft_features = log_fft_result['features']

    # CC_freq eigenvalues (8 features)
    cc_freq_result = extract_cc_freq(window, fs=fs)
    cc_freq_features = cc_freq_result['cc_eigenvalues']

    # CC_time eigenvalues (8 features)
    cc_time_result = extract_cc_time(window)
    cc_time_features = cc_time_result['cc_eigenvalues']

    # Combine: 40 + 8 + 8 = 56 features
    return np.concatenate([log_fft_features, cc_freq_features, cc_time_features])


# ============================================================================
# Data Loading
# ============================================================================

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


def extract_features_for_subject(subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load subject data and extract Tier 3 combined features."""
    subject_dir = DATA_ROOT / subject_id

    if not subject_dir.exists():
        return np.array([]), np.array([])

    segments = extract_segments_for_subject(subject_dir)

    if len(segments) == 0:
        return np.array([]), np.array([])

    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}

    features_list = []
    valid_labels = []

    for segment in segments:
        try:
            data = load_segment_data(segment, LOSO_CHANNELS)

            if data is None:
                continue

            features = extract_combined_features(data, FS)

            if not np.isnan(features).any() and not np.isinf(features).any():
                features_list.append(features)
                valid_labels.append(label_map.get(segment.label, 0))
        except Exception as e:
            logger.debug(f"Error extracting features for {segment.source_file}: {e}")
            continue

    if len(features_list) == 0:
        return np.array([]), np.array([])

    return np.array(features_list), np.array(valid_labels)


# ============================================================================
# LOSO Cross-Validation
# ============================================================================

def run_loso_cv() -> Dict:
    """Run Leave-One-Subject-Out cross-validation on all patients."""

    if not DATA_ROOT.exists():
        logger.error(f"Data directory not found: {DATA_ROOT}")
        return {}

    all_subjects = sorted([
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith('chb')
        and d.name not in EXCLUDE_SUBJECTS
    ])

    logger.info(f"Running Tier 3 LOSO CV on {len(all_subjects)} subjects")

    # Load all data
    logger.info("Loading and extracting features for all subjects...")
    subject_data = {}

    for subject_id in all_subjects:
        logger.info(f"  Processing {subject_id}...")
        X, y = extract_features_for_subject(subject_id)

        if len(X) > 0 and len(np.unique(y)) == 2:
            subject_data[subject_id] = (X, y)
            n_pos = int(sum(y))
            n_neg = len(y) - n_pos
            logger.info(f"    {subject_id}: {len(X)} samples, {n_pos} pos, {n_neg} neg, {X.shape[1]} features")
        else:
            if len(X) > 0:
                logger.warning(f"    {subject_id}: Skipped (single class)")
            else:
                logger.warning(f"    {subject_id}: Skipped (no valid segments)")

    valid_subjects = list(subject_data.keys())
    logger.info(f"Valid subjects: {len(valid_subjects)}")

    if len(valid_subjects) < 2:
        logger.error("Not enough valid subjects")
        return {}

    # LOSO CV
    results_per_subject = {}
    all_y_true = []
    all_y_prob = []

    for test_subject in valid_subjects:
        logger.info(f"Testing on {test_subject}...")

        X_train_list = []
        y_train_list = []
        for train_subject in valid_subjects:
            if train_subject != test_subject:
                X, y = subject_data[train_subject]
                X_train_list.append(X)
                y_train_list.append(y)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test, y_test = subject_data[test_subject]

        # Train 5-bag ensemble
        y_probs = []
        for bag_idx in range(N_BAGS):
            rng = np.random.RandomState(42 + bag_idx)
            idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            model = make_xgb_model(seed=42 + bag_idx)
            model.fit(X_train[idx], y_train[idx])
            y_prob = model.predict_proba(X_test)[:, 1]
            y_probs.append(y_prob)

        y_prob_mean = np.mean(y_probs, axis=0)

        try:
            auc = roc_auc_score(y_test, y_prob_mean)
        except:
            auc = 0.5

        results_per_subject[test_subject] = {
            'auc': auc,
            'n_samples': len(y_test),
            'n_seizure': int(sum(y_test)),
            'n_non_seizure': int(len(y_test) - sum(y_test)),
        }

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob_mean)

        logger.info(f"  {test_subject}: AUC = {auc:.4f}")

    # Overall metrics
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except:
        overall_auc = 0.5

    aucs = [r['auc'] for r in results_per_subject.values()]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Get feature dimension
    first_subject = list(subject_data.keys())[0]
    n_features = subject_data[first_subject][0].shape[1]

    return {
        'tier': 'tier3_combined',
        'n_channels': len(LOSO_CHANNELS),
        'n_features': n_features,
        'channels': LOSO_CHANNELS,
        'window_sec': WINDOW_SEC,
        'n_subjects': len(valid_subjects),
        'overall_auc': overall_auc,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'per_subject': results_per_subject,
        'timestamp': datetime.now().isoformat(),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run Tier 3 LOSO on all patients."""

    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Tier 3 All Patients LOSO (B-003)")
    logger.info("=" * 60)

    results = run_loso_cv()

    with open(output_dir / 'tier3_all_patients_loso.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"8-patient baseline (Tier 3 Combined): 0.7419")
    logger.info(f"All-patient result: {results['overall_auc']:.4f}")
    logger.info(f"Mean AUC: {results['mean_auc']:.4f} +/- {results['std_auc']:.4f}")
    logger.info(f"N subjects: {results['n_subjects']}")
    logger.info(f"N features: {results['n_features']}")

    delta = results['overall_auc'] - 0.7419
    logger.info(f"Delta vs 8-patient baseline: {delta:+.4f}")

    # Comparison with Tier 2
    tier2_overall = 0.7234
    delta_tier2 = results['overall_auc'] - tier2_overall
    logger.info(f"Delta vs Tier 2 all-patient: {delta_tier2:+.4f}")

    # Per-subject breakdown
    logger.info("\nPer-subject AUC:")
    for subj, data in sorted(results['per_subject'].items()):
        logger.info(f"  {subj}: {data['auc']:.4f} (n={data['n_samples']})")

    logger.info(f"\nResults saved to: {output_dir / 'tier3_all_patients_loso.json'}")

    return results


if __name__ == '__main__':
    main()
