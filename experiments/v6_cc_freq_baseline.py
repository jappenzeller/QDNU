#!/usr/bin/env python3
"""
V6 CC_freq Baseline Benchmark (Tier 3)

Benchmarks CC_freq encoding as a new classical baseline tier using
LOSO cross-validation with XGBoost, matching the existing Tier 1/2 protocols.

Features tested:
1. CC_freq eigenvalues only (8 features for 8 channels)
2. CC_time eigenvalues only (8 features)
3. Combined: [log_fft_bands | cc_freq_eigenvalues | cc_time_eigenvalues]

Comparison target: Tier 2 MAX (0.7234 AUC)

Reference: A-004 from research prompts
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
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

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

from scripts.features.cc_freq_encoding import extract_cc_freq
from scripts.features.cc_time_encoding import extract_cc_time
from scripts.features.log_fft_encoding import extract_log_fft_per_band

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration - matches paper exactly
# ============================================================================

# 8-channel subset used in the paper
LOSO_CHANNELS = [
    "FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
    "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"
]

# XGBoost hyperparameters (same as existing baselines)
N_BAGS = 5

# Paper baselines for comparison
PAPER_BASELINES = {
    'XGBoost-8ch-V1': 0.5989,
    'XGBoost-8ch-V2': 0.6253,
    'Tier 2 MAX': 0.7234,
}

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
# Feature Extraction
# ============================================================================

def extract_cc_freq_features(window: np.ndarray, fs: int = FS) -> np.ndarray:
    """Extract CC_freq eigenvalues only."""
    result = extract_cc_freq(window, fs=fs)
    return result['cc_eigenvalues']


def extract_cc_time_features(window: np.ndarray) -> np.ndarray:
    """Extract CC_time eigenvalues only."""
    result = extract_cc_time(window)
    return result['cc_eigenvalues']


def extract_combined_features(window: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Extract combined features: [log_fft_bands | cc_freq_eigenvalues | cc_time_eigenvalues]

    This tests whether combining spectral and correlation features improves performance.
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


def extract_tier3_features(
    window: np.ndarray,
    fs: int = FS,
    feature_type: str = 'cc_freq'
) -> np.ndarray:
    """
    Extract Tier 3 features based on feature_type.

    Args:
        window: EEG data of shape (n_channels, n_samples)
        fs: Sampling frequency
        feature_type: 'cc_freq', 'cc_time', or 'combined'

    Returns:
        Feature vector
    """
    if feature_type == 'cc_freq':
        return extract_cc_freq_features(window, fs)
    elif feature_type == 'cc_time':
        return extract_cc_time_features(window)
    elif feature_type == 'combined':
        return extract_combined_features(window, fs)
    else:
        raise ValueError(f"Unknown feature_type: {feature_type}")


# ============================================================================
# Data Loading with Channel Subset
# ============================================================================

def load_segments_for_subject(subject_id: str) -> List[EEGSegment]:
    """Load segment metadata for a subject."""
    subject_dir = DATA_ROOT / subject_id

    if not subject_dir.exists():
        logger.warning(f"Subject directory not found: {subject_dir}")
        return []

    return extract_segments_for_subject(subject_dir)


def load_segment_data(segment: EEGSegment, channels: List[str] = LOSO_CHANNELS
                     ) -> Optional[np.ndarray]:
    """Load and preprocess EEG data for a segment."""
    edf_path = DATA_ROOT / segment.subject / segment.source_file

    if not edf_path.exists():
        return None

    # Read data with channel selection
    data, fs = read_edf_segment(
        str(edf_path),
        segment.start_sec,
        segment.end_sec,
        target_channels=channels
    )

    if data is None or data.size == 0:
        return None

    # Minimum duration check (at least half of window)
    min_samples = int(WINDOW_SEC * fs * 0.5)
    if data.shape[1] < min_samples:
        return None

    # Preprocess
    data = preprocess_eeg(data, fs=int(fs))

    return data


def extract_features_for_subject(
    subject_id: str,
    feature_type: str = 'cc_freq',
    channels: List[str] = LOSO_CHANNELS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load subject data and extract features.

    Args:
        subject_id: Subject ID
        feature_type: 'cc_freq', 'cc_time', or 'combined'
        channels: List of channel names to use

    Returns:
        Tuple of (X, y) feature matrix and labels
    """
    segments = load_segments_for_subject(subject_id)

    if len(segments) == 0:
        return np.array([]), np.array([])

    # Label mapping: ictal + preictal = 1, interictal = 0
    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}

    features_list = []
    valid_labels = []

    for segment in segments:
        try:
            data = load_segment_data(segment, channels)

            if data is None:
                continue

            features = extract_tier3_features(data, FS, feature_type)

            if not np.isnan(features).any() and not np.isinf(features).any():
                features_list.append(features)
                valid_labels.append(label_map.get(segment.label, 0))
        except Exception as e:
            logger.debug(f"Error extracting features for {segment.source_file}: {e}")
            continue

    if len(features_list) == 0:
        return np.array([]), np.array([])

    X = np.array(features_list)
    y = np.array(valid_labels)

    return X, y


# ============================================================================
# LOSO Cross-Validation
# ============================================================================

def run_loso_cv(feature_type: str = 'cc_freq') -> Dict:
    """
    Run Leave-One-Subject-Out cross-validation.

    Args:
        feature_type: 'cc_freq', 'cc_time', or 'combined'

    Returns:
        Dictionary with results
    """
    if not DATA_ROOT.exists():
        logger.error(f"Data directory not found: {DATA_ROOT}")
        return {}

    all_subjects = sorted([
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith('chb')
        and d.name not in EXCLUDE_SUBJECTS
    ])

    logger.info(f"Running LOSO CV for Tier 3 ({feature_type})")
    logger.info(f"Found {len(all_subjects)} subjects: {all_subjects}")

    # Load all data first
    logger.info("Loading and extracting features for all subjects...")
    subject_data = {}
    for subject_id in all_subjects:
        logger.info(f"  Processing {subject_id}...")
        X, y = extract_features_for_subject(subject_id, feature_type, LOSO_CHANNELS)
        if len(X) > 0 and len(np.unique(y)) == 2:
            subject_data[subject_id] = (X, y)
            n_pos = int(sum(y))
            n_neg = len(y) - n_pos
            logger.info(f"    {subject_id}: {len(X)} samples, {n_pos} seizure, {n_neg} non-seizure, {X.shape[1]} features")
        else:
            if len(X) > 0:
                logger.warning(f"    {subject_id}: Skipped (single class)")
            else:
                logger.warning(f"    {subject_id}: Skipped (no valid segments)")

    valid_subjects = list(subject_data.keys())
    logger.info(f"Valid subjects for LOSO: {len(valid_subjects)}")

    if len(valid_subjects) < 2:
        logger.error("Not enough valid subjects for LOSO CV")
        return {}

    # LOSO CV
    results_per_subject = {}
    all_y_true = []
    all_y_prob = []

    for test_subject in valid_subjects:
        logger.info(f"Testing on {test_subject}...")

        # Train on all other subjects
        X_train_list = []
        y_train_list = []
        for train_subject in valid_subjects:
            if train_subject != test_subject:
                X, y = subject_data[train_subject]
                X_train_list.append(X)
                y_train_list.append(y)

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)

        # Test data
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

        # Average predictions
        y_prob_mean = np.mean(y_probs, axis=0)

        # Calculate AUC
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

    # Overall AUC
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except:
        overall_auc = 0.5

    # Mean AUC across subjects
    aucs = [r['auc'] for r in results_per_subject.values()]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Get feature dimension from first subject
    first_subject = list(subject_data.keys())[0]
    n_features = subject_data[first_subject][0].shape[1]

    results = {
        'tier': 'tier3',
        'feature_type': feature_type,
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

    return results


# ============================================================================
# Feature Selection for Combined Features
# ============================================================================

def run_loso_cv_with_feature_selection(top_k: int = 20) -> Dict:
    """
    Run LOSO CV with RF-based feature selection on combined features.

    Uses RandomForest feature importance to select top-k features,
    then trains XGBoost on selected features.
    """
    if not DATA_ROOT.exists():
        logger.error(f"Data directory not found: {DATA_ROOT}")
        return {}

    all_subjects = sorted([
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith('chb')
        and d.name not in EXCLUDE_SUBJECTS
    ])

    logger.info(f"Running LOSO CV with feature selection (top_k={top_k})")

    # Load all data
    subject_data = {}
    for subject_id in all_subjects:
        X, y = extract_features_for_subject(subject_id, 'combined', LOSO_CHANNELS)
        if len(X) > 0 and len(np.unique(y)) == 2:
            subject_data[subject_id] = (X, y)

    valid_subjects = list(subject_data.keys())
    logger.info(f"Valid subjects: {len(valid_subjects)}")

    if len(valid_subjects) < 2:
        return {}

    # LOSO CV with feature selection
    results_per_subject = {}
    all_y_true = []
    all_y_prob = []

    for test_subject in valid_subjects:
        logger.info(f"Testing on {test_subject}...")

        # Combine training data
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

        # Feature selection using RF importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)

        # Select top-k features
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[::-1][:top_k]

        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]

        # Train XGBoost on selected features
        y_probs = []
        for bag_idx in range(N_BAGS):
            rng = np.random.RandomState(42 + bag_idx)
            idx = rng.choice(len(X_train_selected), size=len(X_train_selected), replace=True)
            model = make_xgb_model(seed=42 + bag_idx)
            model.fit(X_train_selected[idx], y_train[idx])
            y_prob = model.predict_proba(X_test_selected)[:, 1]
            y_probs.append(y_prob)

        y_prob_mean = np.mean(y_probs, axis=0)

        try:
            auc = roc_auc_score(y_test, y_prob_mean)
        except:
            auc = 0.5

        results_per_subject[test_subject] = {'auc': auc}
        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob_mean)

        logger.info(f"  {test_subject}: AUC = {auc:.4f}")

    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except:
        overall_auc = 0.5

    aucs = [r['auc'] for r in results_per_subject.values()]

    return {
        'tier': 'tier3',
        'feature_type': 'combined_selected',
        'top_k': top_k,
        'overall_auc': overall_auc,
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'per_subject': results_per_subject,
        'timestamp': datetime.now().isoformat(),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all Tier 3 benchmarks and save results."""

    output_dir = Path(__file__).parent.parent / 'results' / 'tier3_cc_freq'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("V6 CC_freq Baseline Benchmark (Tier 3)")
    logger.info("=" * 60)

    results = {}

    # Run CC_freq eigenvalues
    logger.info("\n" + "=" * 60)
    logger.info("TIER 3: CC_freq Eigenvalues")
    logger.info("=" * 60)
    cc_freq_results = run_loso_cv(feature_type='cc_freq')
    results['cc_freq'] = cc_freq_results

    with open(output_dir / 'tier3_cc_freq_loso.json', 'w', encoding='utf-8') as f:
        json.dump(cc_freq_results, f, indent=2)
    logger.info(f"CC_freq Overall AUC: {cc_freq_results.get('overall_auc', 'N/A'):.4f}")

    # Run CC_time eigenvalues
    logger.info("\n" + "=" * 60)
    logger.info("TIER 3: CC_time Eigenvalues")
    logger.info("=" * 60)
    cc_time_results = run_loso_cv(feature_type='cc_time')
    results['cc_time'] = cc_time_results

    with open(output_dir / 'tier3_cc_time_loso.json', 'w', encoding='utf-8') as f:
        json.dump(cc_time_results, f, indent=2)
    logger.info(f"CC_time Overall AUC: {cc_time_results.get('overall_auc', 'N/A'):.4f}")

    # Run combined features
    logger.info("\n" + "=" * 60)
    logger.info("TIER 3: Combined Features")
    logger.info("=" * 60)
    combined_results = run_loso_cv(feature_type='combined')
    results['combined'] = combined_results

    with open(output_dir / 'tier3_combined_loso.json', 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2)
    logger.info(f"Combined Overall AUC: {combined_results.get('overall_auc', 'N/A'):.4f}")

    # Run combined with feature selection
    logger.info("\n" + "=" * 60)
    logger.info("TIER 3: Combined + Feature Selection (top 20)")
    logger.info("=" * 60)
    selected_results = run_loso_cv_with_feature_selection(top_k=20)
    results['combined_selected'] = selected_results

    with open(output_dir / 'tier3_combined_selected_loso.json', 'w', encoding='utf-8') as f:
        json.dump(selected_results, f, indent=2)
    logger.info(f"Combined+Selected Overall AUC: {selected_results.get('overall_auc', 'N/A'):.4f}")

    # Save combined results
    with open(output_dir / 'tier3_cc_freq_baseline.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info("Paper Baselines:")
    for name, auc in PAPER_BASELINES.items():
        logger.info(f"  {name}: {auc:.4f}")

    logger.info("\nTier 3 Results:")
    for name, res in results.items():
        if isinstance(res, dict) and 'overall_auc' in res:
            delta = res['overall_auc'] - PAPER_BASELINES['Tier 2 MAX']
            logger.info(f"  {name}: {res['overall_auc']:.4f} (delta vs Tier2 MAX = {delta:+.4f})")

    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == '__main__':
    main()
