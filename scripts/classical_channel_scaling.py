#!/usr/bin/env python3
"""
Classical Tier 2 MAX Channel Scaling (8 / 12 / 16 channels)

Runs Tier 2 MAX at three channel counts under identical LOSO CV conditions
to characterize how the classical ceiling changes with dimensionality.

CH8 must validate within 0.005 of 0.7234 before proceeding to CH12/CH16.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from scipy.signal import welch
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).parent.parent))

from sagemaker.train_chbmit import (
    COMMON_CHANNELS,
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

# CH8: The 8-channel subset that produced 0.7234 AUC
# (from analysis_results/quantum_loso_v2/summary.json)
CH8 = [
    "FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
    "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"
]

# CH12: Extend CH8 with 4 more channels from COMMON_CHANNELS
# Select channels not already in CH8
CH8_SET = set(CH8)
ADDITIONAL_CH12 = [ch for ch in COMMON_CHANNELS if ch not in CH8_SET][:4]
CH12 = CH8 + ADDITIONAL_CH12

# CH16: Extend CH12 with 4 more channels
CH12_SET = set(CH12)
ADDITIONAL_CH16 = [ch for ch in COMMON_CHANNELS if ch not in CH12_SET][:4]
CH16 = CH12 + ADDITIONAL_CH16

N_BAGS = 5
EXPECTED_CH8_AUC = 0.7234
VALIDATION_TOLERANCE = 0.005

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
# Feature Extraction (Tier 2 MAX - same as tier1_tier2_loso_cv.py)
# ============================================================================

def extract_michaelhills_subwindow(subwindow: np.ndarray, fs: int = FS) -> np.ndarray:
    """Extract MichaelHills features from a 1-second sub-window."""
    n_channels = subwindow.shape[0]

    # FFT magnitude (1-40 Hz)
    fft_result = np.fft.rfft(subwindow, axis=1)
    fft_mag = np.abs(fft_result)[:, 1:41]
    fft_mag = np.log10(fft_mag + 1e-6)

    # Frequency-domain correlation eigenvalues
    try:
        C_freq = np.corrcoef(fft_mag)
        if np.isnan(C_freq).any():
            eigenvalues_freq = np.zeros(n_channels)
        else:
            eigenvalues_freq = np.linalg.eigvalsh(C_freq)
    except:
        eigenvalues_freq = np.zeros(n_channels)

    # Time-domain correlation eigenvalues
    try:
        C_time = np.corrcoef(subwindow)
        if np.isnan(C_time).any():
            eigenvalues_time = np.zeros(n_channels)
        else:
            eigenvalues_time = np.linalg.eigvalsh(C_time)
    except:
        eigenvalues_time = np.zeros(n_channels)

    features = np.concatenate([fft_mag.flatten(), eigenvalues_freq, eigenvalues_time])
    return features


def extract_tier2_features(window: np.ndarray, fs: int = FS) -> np.ndarray:
    """Extract Tier 2 features with MAX pooling across 1-second sub-windows."""
    n_channels, n_samples = window.shape
    samples_per_subwindow = fs
    n_subwindows = n_samples // samples_per_subwindow

    if n_subwindows == 0:
        return extract_michaelhills_subwindow(window[:, :fs], fs)

    subwindow_features = []
    for i in range(n_subwindows):
        start = i * samples_per_subwindow
        end = start + samples_per_subwindow
        subwindow = window[:, start:end]
        features = extract_michaelhills_subwindow(subwindow, fs)
        subwindow_features.append(features)

    subwindow_features = np.array(subwindow_features)
    return np.max(subwindow_features, axis=0)


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


def extract_features_for_subject(
    subject_id: str,
    channels: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Load subject data and extract Tier 2 MAX features."""
    segments = load_segments_for_subject(subject_id)

    if len(segments) == 0:
        return np.array([]), np.array([])

    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}

    features_list = []
    valid_labels = []

    for segment in segments:
        try:
            data = load_segment_data(segment, channels)
            if data is None:
                continue

            features = extract_tier2_features(data, FS)

            if not np.isnan(features).any() and not np.isinf(features).any():
                features_list.append(features)
                valid_labels.append(label_map.get(segment.label, 0))
        except Exception as e:
            logger.debug(f"Error: {e}")
            continue

    if len(features_list) == 0:
        return np.array([]), np.array([])

    return np.array(features_list), np.array(valid_labels)


# ============================================================================
# LOSO CV
# ============================================================================

def run_loso_cv(channels: List[str]) -> Dict:
    """Run Tier 2 MAX LOSO CV with specified channels."""

    if not DATA_ROOT.exists():
        logger.error(f"Data directory not found: {DATA_ROOT}")
        return {}

    all_subjects = sorted([
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith('chb')
        and d.name not in EXCLUDE_SUBJECTS
    ])

    logger.info(f"Running LOSO CV with {len(channels)} channels")
    logger.info(f"Channels: {channels}")

    # Load all data
    subject_data = {}
    for subject_id in all_subjects:
        X, y = extract_features_for_subject(subject_id, channels)
        if len(X) > 0 and len(np.unique(y)) == 2:
            subject_data[subject_id] = (X, y)
            logger.info(f"  {subject_id}: {len(X)} samples, {int(sum(y))} seizure")
        else:
            logger.warning(f"  {subject_id}: Skipped")

    valid_subjects = list(subject_data.keys())
    logger.info(f"Valid subjects: {len(valid_subjects)}")

    # LOSO CV
    results_per_subject = {}
    all_y_true = []
    all_y_prob = []

    for test_subject in valid_subjects:
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

        # 5-bag ensemble with bootstrap
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
        }

        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob_mean)

        logger.info(f"  {test_subject}: AUC = {auc:.4f}")

    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except:
        overall_auc = 0.5

    aucs = [r['auc'] for r in results_per_subject.values()]

    return {
        'channels': channels,
        'n_channels': len(channels),
        'overall_auc': overall_auc,
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'per_subject': results_per_subject,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run channel scaling benchmark."""

    output_dir = Path(__file__).parent.parent / 'results' / 'scaling'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Classical Tier 2 MAX Channel Scaling")
    logger.info("=" * 60)

    logger.info(f"\nChannel sets (nested):")
    logger.info(f"  CH8  ({len(CH8)} channels):  {CH8}")
    logger.info(f"  CH12 ({len(CH12)} channels): {CH12}")
    logger.info(f"  CH16 ({len(CH16)} channels): {CH16}")

    # Step 1: Validate CH8
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: Validate CH8 (expect ~0.7234)")
    logger.info("=" * 60)

    ch8_results = run_loso_cv(CH8)

    if abs(ch8_results['overall_auc'] - EXPECTED_CH8_AUC) > VALIDATION_TOLERANCE:
        logger.error(f"CH8 validation FAILED: {ch8_results['overall_auc']:.4f} vs expected {EXPECTED_CH8_AUC:.4f}")
        logger.error(f"Difference: {abs(ch8_results['overall_auc'] - EXPECTED_CH8_AUC):.4f} > tolerance {VALIDATION_TOLERANCE}")
        logger.error("Not proceeding to CH12/CH16")
        return

    logger.info(f"CH8 validation PASSED: {ch8_results['overall_auc']:.4f}")

    # Step 2: Run CH12
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: CH12")
    logger.info("=" * 60)

    ch12_results = run_loso_cv(CH12)
    logger.info(f"CH12 Overall AUC: {ch12_results['overall_auc']:.4f}")

    # Step 3: Run CH16
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: CH16")
    logger.info("=" * 60)

    ch16_results = run_loso_cv(CH16)
    logger.info(f"CH16 Overall AUC: {ch16_results['overall_auc']:.4f}")

    # Save results
    results = {
        'method': 'Tier2_MAX_XGBoost',
        'timestamp': datetime.now().isoformat(),
        'ch8': ch8_results,
        'ch12': ch12_results,
        'ch16': ch16_results,
    }

    with open(output_dir / 'classical_channel_scaling.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Determine trend
    aucs = [ch8_results['overall_auc'], ch12_results['overall_auc'], ch16_results['overall_auc']]
    if aucs[2] > aucs[0] + 0.01:
        trend = 'up'
    elif aucs[2] < aucs[0] - 0.01:
        trend = 'down'
    else:
        trend = 'flat'

    # Generate comparison markdown
    md = f"""# Channel Scaling Comparison

## Results

| Method          | 8ch AUC | 12ch AUC | 16ch AUC | Trend |
|-----------------|---------|----------|----------|-------|
| Classical Tier2 | {ch8_results['overall_auc']:.4f}  | {ch12_results['overall_auc']:.4f}   | {ch16_results['overall_auc']:.4f}   | {trend} |
| Quantum QPNN    | TBD     | TBD      | TBD      | TBD   |

## Channel Sets (Nested)

- **CH8**: {CH8}
- **CH12**: {CH12}
- **CH16**: {CH16}

## Configuration

- Method: Tier 2 MAX (correlation eigenvalues with MAX pooling)
- Classifier: XGBoost 5-bag ensemble
- CV: Leave-One-Subject-Out

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(output_dir / 'channel_scaling_comparison.md', 'w', encoding='utf-8') as f:
        f.write(md)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"CH8:  {ch8_results['overall_auc']:.4f}")
    logger.info(f"CH12: {ch12_results['overall_auc']:.4f}")
    logger.info(f"CH16: {ch16_results['overall_auc']:.4f}")
    logger.info(f"Trend: {trend}")
    logger.info(f"\nResults saved to: {output_dir}")

    return results


if __name__ == '__main__':
    main()
