#!/usr/bin/env python3
"""
Tier 1 and Tier 2 Benchmarks with LOSO Cross-Patient CV

This script evaluates classical feature extraction methods using Leave-One-Subject-Out
cross-validation to enable direct comparison with the paper's XGBoost baselines.

Paper baselines (from analysis_results/quantum_loso_v2/classical_results.json):
- XGBoost-8ch (V1): 0.5989 AUC
- XGBoost-8ch-V2:   0.6253 AUC

Configuration matches paper:
- 8 channels (subset of 18 common channels)
- 30-second windows
- LOSO cross-patient CV
- XGBoost with 5-bag ensemble
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
# Configuration - matches paper exactly
# ============================================================================

# 8-channel subset used in the paper (from analysis_results/quantum_loso_v2/summary.json)
LOSO_CHANNELS = [
    "FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
    "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"
]

# XGBoost hyperparameters (from scripts/run_loso_xgboost.py)
N_BAGS = 5

# FFT Power Bands for Tier 1 (standard EEG bands)
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40),
}

# Paper baselines for comparison
PAPER_BASELINES = {
    'XGBoost-8ch-V1': 0.5989,
    'XGBoost-8ch-V2': 0.6253,
}

# CHB-MIT data path (matches other scripts)
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
# Tier 1: FFT Power Bands (Welch PSD)
# ============================================================================

def extract_tier1_features(window: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Extract Tier 1 features: FFT Power Bands using Welch PSD.

    Args:
        window: EEG data of shape (n_channels, n_samples)
        fs: Sampling frequency (256 Hz)

    Returns:
        Feature vector of shape (n_channels * n_bands,) = (8 * 5 = 40,)
    """
    n_channels = window.shape[0]
    n_bands = len(FREQ_BANDS)
    features = np.zeros((n_channels, n_bands))

    for ch_idx in range(n_channels):
        # Welch PSD estimation
        freqs, psd = welch(window[ch_idx], fs=fs, nperseg=min(256, window.shape[1]))

        # Extract power in each band
        for band_idx, (band_name, (f_low, f_high)) in enumerate(FREQ_BANDS.items()):
            band_mask = (freqs >= f_low) & (freqs <= f_high)
            if np.any(band_mask):
                features[ch_idx, band_idx] = np.mean(psd[band_mask])
            else:
                features[ch_idx, band_idx] = 0.0

    # Log transform for stability
    features = np.log10(features + 1e-10)

    return features.flatten()


# ============================================================================
# Tier 2: Sub-window Aggregation with MichaelHills Features
# ============================================================================

def extract_michaelhills_subwindow(subwindow: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Extract MichaelHills features from a 1-second sub-window.

    Features:
    - FFT magnitude (1-40 Hz) for each channel
    - Frequency-domain correlation eigenvalues
    - Time-domain correlation eigenvalues

    Args:
        subwindow: EEG data of shape (n_channels, fs) - 1 second
        fs: Sampling frequency

    Returns:
        Feature vector
    """
    n_channels = subwindow.shape[0]

    # FFT magnitude (1-40 Hz)
    fft_result = np.fft.rfft(subwindow, axis=1)
    fft_mag = np.abs(fft_result)[:, 1:41]  # 1-40 Hz
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


def extract_tier2_features(window: np.ndarray, fs: int = FS,
                           aggregation: str = 'max') -> np.ndarray:
    """
    Extract Tier 2 features: Sub-window aggregation of MichaelHills features.

    Splits 30-second window into 30 1-second sub-windows, extracts features
    from each, then aggregates using MAX or MEAN pooling.

    Args:
        window: EEG data of shape (n_channels, n_samples) - 30 seconds
        fs: Sampling frequency (256 Hz)
        aggregation: 'max' or 'mean' pooling

    Returns:
        Aggregated feature vector
    """
    n_channels, n_samples = window.shape
    samples_per_subwindow = fs  # 1 second = 256 samples
    n_subwindows = n_samples // samples_per_subwindow

    if n_subwindows == 0:
        # Fallback for short windows
        return extract_michaelhills_subwindow(window[:, :fs], fs)

    # Extract features from each sub-window
    subwindow_features = []
    for i in range(n_subwindows):
        start = i * samples_per_subwindow
        end = start + samples_per_subwindow
        subwindow = window[:, start:end]
        features = extract_michaelhills_subwindow(subwindow, fs)
        subwindow_features.append(features)

    subwindow_features = np.array(subwindow_features)

    # Aggregate across sub-windows
    if aggregation == 'max':
        return np.max(subwindow_features, axis=0)
    elif aggregation == 'mean':
        return np.mean(subwindow_features, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


# ============================================================================
# Data Loading with Channel Subset
# ============================================================================

def load_segments_for_subject(subject_id: str) -> List[EEGSegment]:
    """
    Load segment metadata for a subject.

    Args:
        subject_id: Subject ID (e.g., 'chb01')

    Returns:
        List of EEGSegment objects (without data loaded)
    """
    subject_dir = DATA_ROOT / subject_id

    if not subject_dir.exists():
        logger.warning(f"Subject directory not found: {subject_dir}")
        return []

    return extract_segments_for_subject(subject_dir)


def load_segment_data(segment: EEGSegment, channels: List[str] = LOSO_CHANNELS
                     ) -> Optional[np.ndarray]:
    """
    Load and preprocess EEG data for a segment.

    Args:
        segment: EEGSegment object with metadata
        channels: List of channel names to use

    Returns:
        Preprocessed EEG data of shape (n_channels, n_samples) or None
    """
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
    tier: str = 'tier1',
    aggregation: str = 'max',
    channels: List[str] = LOSO_CHANNELS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load subject data and extract features.

    Args:
        subject_id: Subject ID
        tier: 'tier1' or 'tier2'
        aggregation: For tier2, 'max' or 'mean' pooling
        channels: List of channel names to use

    Returns:
        Tuple of (X, y) feature matrix and labels (binary: 0=interictal, 1=ictal/preictal)
    """
    segments = load_segments_for_subject(subject_id)

    if len(segments) == 0:
        return np.array([]), np.array([])

    # Label mapping: ictal + preictal = 1, interictal = 0
    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}

    # Extract features from each segment
    features_list = []
    valid_labels = []

    for segment in segments:
        try:
            # Load and preprocess data
            data = load_segment_data(segment, channels)

            if data is None:
                continue

            # Extract features
            if tier == 'tier1':
                features = extract_tier1_features(data, FS)
            else:
                features = extract_tier2_features(data, FS, aggregation)

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

def run_loso_cv(
    tier: str = 'tier1',
    aggregation: str = 'max',
) -> Dict:
    """
    Run Leave-One-Subject-Out cross-validation.

    Args:
        tier: 'tier1' or 'tier2'
        aggregation: For tier2, 'max' or 'mean'

    Returns:
        Dictionary with results
    """
    # Get all subjects
    if not DATA_ROOT.exists():
        logger.error(f"Data directory not found: {DATA_ROOT}")
        return {}

    all_subjects = sorted([
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith('chb')
        and d.name not in EXCLUDE_SUBJECTS
    ])

    logger.info(f"Running LOSO CV for {tier} (aggregation={aggregation})")
    logger.info(f"Found {len(all_subjects)} subjects: {all_subjects}")

    # Load all data first
    logger.info("Loading and extracting features for all subjects...")
    subject_data = {}
    for subject_id in all_subjects:
        logger.info(f"  Processing {subject_id}...")
        X, y = extract_features_for_subject(subject_id, tier, aggregation, LOSO_CHANNELS)
        if len(X) > 0 and len(np.unique(y)) == 2:
            subject_data[subject_id] = (X, y)
            n_pos = int(sum(y))
            n_neg = len(y) - n_pos
            logger.info(f"    {subject_id}: {len(X)} samples, {n_pos} seizure, {n_neg} non-seizure")
        else:
            if len(X) > 0:
                logger.warning(f"    {subject_id}: Skipped (single class - {len(X)} samples, unique labels: {np.unique(y)})")
            else:
                logger.warning(f"    {subject_id}: Skipped (no valid segments)")

    valid_subjects = list(subject_data.keys())
    logger.info(f"Valid subjects for LOSO: {len(valid_subjects)}")

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

        # Train 5-bag ensemble with bootstrap sampling
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

    # Overall AUC (pooled predictions)
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except:
        overall_auc = 0.5

    # Mean AUC across subjects
    aucs = [r['auc'] for r in results_per_subject.values()]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    results = {
        'tier': tier,
        'aggregation': aggregation if tier == 'tier2' else None,
        'n_channels': len(LOSO_CHANNELS),
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
# Comparison Output
# ============================================================================

def generate_comparison_markdown(
    tier1_results: Dict,
    tier2_max_results: Dict,
    tier2_mean_results: Dict,
) -> str:
    """Generate markdown comparison table."""

    md = """# LOSO Cross-Patient CV Results

## Comparison with Paper Baselines

| Method | Overall AUC | Mean AUC ± Std | Δ vs XGBoost-V2 |
|--------|-------------|----------------|-----------------|
"""

    # Paper baselines
    for name, auc in PAPER_BASELINES.items():
        delta = auc - PAPER_BASELINES['XGBoost-8ch-V2']
        md += f"| {name} (paper) | {auc:.4f} | - | {delta:+.4f} |\n"

    # Our results
    for name, results in [
        ('Tier 1 (FFT Power)', tier1_results),
        ('Tier 2 (MAX pool)', tier2_max_results),
        ('Tier 2 (MEAN pool)', tier2_mean_results),
    ]:
        overall = results['overall_auc']
        mean = results['mean_auc']
        std = results['std_auc']
        delta = overall - PAPER_BASELINES['XGBoost-8ch-V2']
        md += f"| {name} | {overall:.4f} | {mean:.4f} ± {std:.4f} | {delta:+.4f} |\n"

    md += """
## Configuration

- **Channels**: 8 (matching paper)
- **Window**: 30 seconds
- **CV Method**: Leave-One-Subject-Out (LOSO)
- **Classifier**: XGBoost with 5-bag ensemble
- **Hyperparameters**: n_estimators=200, max_depth=6, learning_rate=0.1

## Per-Subject Results

| Subject | Tier 1 AUC | Tier 2 MAX | Tier 2 MEAN | Best |
|---------|------------|------------|-------------|------|
"""

    # Collect all subjects
    all_subjects = set()
    for results in [tier1_results, tier2_max_results, tier2_mean_results]:
        all_subjects.update(results['per_subject'].keys())

    for subject in sorted(all_subjects):
        t1 = tier1_results['per_subject'].get(subject, {}).get('auc', '-')
        t2_max = tier2_max_results['per_subject'].get(subject, {}).get('auc', '-')
        t2_mean = tier2_mean_results['per_subject'].get(subject, {}).get('auc', '-')

        if isinstance(t1, float) and isinstance(t2_max, float) and isinstance(t2_mean, float):
            best_val = max(t1, t2_max, t2_mean)
            if best_val == t1:
                best = 'Tier 1'
            elif best_val == t2_max:
                best = 'Tier 2 MAX'
            else:
                best = 'Tier 2 MEAN'
            md += f"| {subject} | {t1:.4f} | {t2_max:.4f} | {t2_mean:.4f} | {best} |\n"
        else:
            md += f"| {subject} | {t1} | {t2_max} | {t2_mean} | - |\n"

    md += f"""
## Summary

- **Tier 1 Features**: FFT Power Bands (5 bands × 8 channels = 40 features)
- **Tier 2 Features**: MichaelHills correlation eigenvalues with sub-window aggregation

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    return md


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all benchmarks and save results."""

    output_dir = Path(__file__).parent.parent / 'results' / 'loso_cv'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("LOSO Cross-Patient CV Benchmark")
    logger.info("=" * 60)

    # Run Tier 1
    logger.info("\n" + "=" * 60)
    logger.info("TIER 1: FFT Power Bands")
    logger.info("=" * 60)
    tier1_results = run_loso_cv(tier='tier1')

    with open(output_dir / 'tier1_loso_results.json', 'w', encoding='utf-8') as f:
        json.dump(tier1_results, f, indent=2)
    logger.info(f"Tier 1 Overall AUC: {tier1_results['overall_auc']:.4f}")

    # Run Tier 2 with MAX pooling
    logger.info("\n" + "=" * 60)
    logger.info("TIER 2: MichaelHills with MAX pooling")
    logger.info("=" * 60)
    tier2_max_results = run_loso_cv(tier='tier2', aggregation='max')

    with open(output_dir / 'tier2_loso_max_results.json', 'w', encoding='utf-8') as f:
        json.dump(tier2_max_results, f, indent=2)
    logger.info(f"Tier 2 MAX Overall AUC: {tier2_max_results['overall_auc']:.4f}")

    # Run Tier 2 with MEAN pooling
    logger.info("\n" + "=" * 60)
    logger.info("TIER 2: MichaelHills with MEAN pooling")
    logger.info("=" * 60)
    tier2_mean_results = run_loso_cv(tier='tier2', aggregation='mean')

    with open(output_dir / 'tier2_loso_mean_results.json', 'w', encoding='utf-8') as f:
        json.dump(tier2_mean_results, f, indent=2)
    logger.info(f"Tier 2 MEAN Overall AUC: {tier2_mean_results['overall_auc']:.4f}")

    # Generate comparison
    comparison_md = generate_comparison_markdown(
        tier1_results, tier2_max_results, tier2_mean_results
    )

    with open(output_dir / 'loso_comparison.md', 'w', encoding='utf-8') as f:
        f.write(comparison_md)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Paper XGBoost-8ch-V1: {PAPER_BASELINES['XGBoost-8ch-V1']:.4f}")
    logger.info(f"Paper XGBoost-8ch-V2: {PAPER_BASELINES['XGBoost-8ch-V2']:.4f}")
    logger.info(f"Tier 1 (FFT Power):   {tier1_results['overall_auc']:.4f} (Δ = {tier1_results['overall_auc'] - PAPER_BASELINES['XGBoost-8ch-V2']:+.4f})")
    logger.info(f"Tier 2 (MAX pool):    {tier2_max_results['overall_auc']:.4f} (Δ = {tier2_max_results['overall_auc'] - PAPER_BASELINES['XGBoost-8ch-V2']:+.4f})")
    logger.info(f"Tier 2 (MEAN pool):   {tier2_mean_results['overall_auc']:.4f} (Δ = {tier2_mean_results['overall_auc'] - PAPER_BASELINES['XGBoost-8ch-V2']:+.4f})")
    logger.info(f"\nResults saved to: {output_dir}")

    return {
        'tier1': tier1_results,
        'tier2_max': tier2_max_results,
        'tier2_mean': tier2_mean_results,
    }


if __name__ == '__main__':
    main()
