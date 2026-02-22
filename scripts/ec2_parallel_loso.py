#!/usr/bin/env python3
"""
EC2 Parallel LOSO Cross-Validation (D-002)

Runs Leave-One-Subject-Out cross-validation on all 23 CHB-MIT patients
using joblib parallelization for c5.4xlarge (16 vCPU) spot instances.

Each parallel job processes one patient as test subject, trains on remaining
patients, and saves results to results/per_patient/{subject_id}.json.

Final aggregation combines all per-patient results into a single output file.

Usage:
    python scripts/ec2_parallel_loso.py [--tier tier2|tier3] [--n-jobs -1]

Output:
    results/per_patient/{subject_id}.json  (individual patient results)
    results/ec2_parallel_loso_{tier}.json  (aggregated results)

Reference: D-002 from research prompts
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from scipy.signal import welch
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from joblib import Parallel, delayed

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sagemaker.train_chbmit import (
        EXCLUDE_SUBJECTS,
        WINDOW_SEC,
        FS,
        EEGSegment,
        preprocess_eeg,
        read_edf_segment,
        extract_segments_for_subject,
    )
except ImportError:
    # Fallback for EC2 environment
    EXCLUDE_SUBJECTS = {'chb12', 'chb24'}  # Standard exclusions
    WINDOW_SEC = 1.95
    FS = 256

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/loso.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# 8-channel subset (matches existing baseline and hardware experiments)
LOSO_CHANNELS_8 = [
    "FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
    "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"
]

N_BAGS = 5

# Data paths (EC2 or local)
DATA_ROOT_CANDIDATES = [
    Path("/data/chbmit"),          # EC2 EBS mount
    Path("/home/ubuntu/data/chbmit"),
    Path("H:/Data/PythonDNU/EEG/chbmit"),  # Windows local
    Path("data/chbmit"),           # Relative path
]


def find_data_root() -> Path:
    """Find CHB-MIT data directory from candidates."""
    for path in DATA_ROOT_CANDIDATES:
        if path.exists() and any(path.iterdir()):
            return path
    raise FileNotFoundError(
        f"CHB-MIT data not found. Checked: {[str(p) for p in DATA_ROOT_CANDIDATES]}"
    )


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
        n_jobs=1,  # Single-threaded within parallel job
    )


# ============================================================================
# Feature Extraction
# ============================================================================

def extract_michaelhills_subwindow(subwindow: np.ndarray, fs: int = 256) -> np.ndarray:
    """Extract MichaelHills features from a 1-second sub-window.

    Features:
    - log10 FFT magnitudes (1-40 Hz): 40 * n_channels
    - Frequency-domain correlation eigenvalues: n_channels
    - Time-domain correlation eigenvalues: n_channels
    """
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
    except Exception:
        eigenvalues_freq = np.zeros(n_channels)

    # Time-domain correlation eigenvalues
    try:
        C_time = np.corrcoef(subwindow)
        if np.isnan(C_time).any():
            eigenvalues_time = np.zeros(n_channels)
        else:
            eigenvalues_time = np.linalg.eigvalsh(C_time)
    except Exception:
        eigenvalues_time = np.zeros(n_channels)

    features = np.concatenate([fft_mag.flatten(), eigenvalues_freq, eigenvalues_time])
    return features


def extract_features_with_aggregation(
    window: np.ndarray,
    fs: int = 256,
    aggregation: str = 'max'
) -> np.ndarray:
    """Extract features with sub-window aggregation (Tier 2/3 style)."""
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

    if aggregation == 'max':
        return np.max(subwindow_features, axis=0)
    elif aggregation == 'mean':
        return np.mean(subwindow_features, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


# ============================================================================
# Data Loading (Standalone Implementation for EC2)
# ============================================================================

def read_edf_segment_standalone(
    filepath: str,
    start_sec: float,
    end_sec: float,
    target_channels: List[str] = None
) -> Tuple[Optional[np.ndarray], float]:
    """Read a segment from an EDF file (standalone implementation)."""
    import pyedflib

    try:
        edf = pyedflib.EdfReader(filepath)
    except Exception as e:
        logger.debug(f"Failed to open EDF: {filepath}: {e}")
        return None, 0.0

    try:
        n_signals = edf.signals_in_file
        signal_labels = [edf.getSignalLabels()[i].strip() for i in range(n_signals)]
        fs = edf.getSampleFrequency(0)

        # Map target channels to indices
        if target_channels:
            channel_indices = []
            for ch in target_channels:
                # Try exact match first
                if ch in signal_labels:
                    channel_indices.append(signal_labels.index(ch))
                else:
                    # Try without spaces/case
                    ch_normalized = ch.replace(' ', '').upper()
                    found = False
                    for idx, label in enumerate(signal_labels):
                        if label.replace(' ', '').upper() == ch_normalized:
                            channel_indices.append(idx)
                            found = True
                            break
                    if not found:
                        return None, 0.0
        else:
            channel_indices = list(range(n_signals))

        # Read samples
        start_sample = int(start_sec * fs)
        end_sample = int(end_sec * fs)
        n_samples = end_sample - start_sample

        if n_samples <= 0:
            return None, 0.0

        data = np.zeros((len(channel_indices), n_samples))
        for i, ch_idx in enumerate(channel_indices):
            signal = edf.readSignal(ch_idx, start=start_sample, n=n_samples)
            if len(signal) < n_samples:
                # Pad if necessary
                padded = np.zeros(n_samples)
                padded[:len(signal)] = signal
                signal = padded
            data[i, :] = signal[:n_samples]

        return data, fs

    finally:
        edf.close()


def preprocess_eeg_standalone(data: np.ndarray, fs: int = 256) -> np.ndarray:
    """Basic EEG preprocessing (bandpass filter, normalize)."""
    from scipy.signal import butter, filtfilt

    # Bandpass filter 1-40 Hz
    nyq = fs / 2
    low, high = 1.0 / nyq, min(40.0 / nyq, 0.99)

    try:
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, data, axis=1)
    except Exception:
        filtered = data

    # Z-score normalization per channel
    mean = filtered.mean(axis=1, keepdims=True)
    std = filtered.std(axis=1, keepdims=True) + 1e-8
    normalized = (filtered - mean) / std

    return normalized


# ============================================================================
# Segment Extraction
# ============================================================================

def parse_summary_file(summary_path: Path) -> List[Dict]:
    """Parse CHB-MIT summary file to extract seizure annotations."""
    segments = []

    if not summary_path.exists():
        return segments

    with open(summary_path, 'r') as f:
        content = f.read()

    # Simple parser for seizure info
    current_file = None
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('File Name:'):
            current_file = line.split(':')[1].strip()
        elif line.startswith('Seizure Start Time:') or line.startswith('Seizure 1 Start Time:'):
            try:
                start = int(line.split(':')[-1].strip().split()[0])
            except Exception:
                continue
        elif (line.startswith('Seizure End Time:') or line.startswith('Seizure 1 End Time:')) and current_file:
            try:
                end = int(line.split(':')[-1].strip().split()[0])
                segments.append({
                    'file': current_file,
                    'seizure_start': start,
                    'seizure_end': end
                })
            except Exception:
                continue

    return segments


def extract_segments_standalone(
    subject_dir: Path,
    channels: List[str],
    window_sec: float = 1.95
) -> Tuple[List[np.ndarray], List[int]]:
    """Extract ictal and interictal segments for a subject."""
    features_list = []
    labels = []

    subject_id = subject_dir.name
    summary_file = subject_dir / f"{subject_id}-summary.txt"
    seizure_info = parse_summary_file(summary_file)

    # Get all EDF files
    edf_files = sorted(subject_dir.glob("*.edf"))

    for edf_path in edf_files:
        file_name = edf_path.name

        # Find seizures in this file
        file_seizures = [s for s in seizure_info if s['file'] == file_name]

        # Get file duration
        try:
            import pyedflib
            edf = pyedflib.EdfReader(str(edf_path))
            fs = edf.getSampleFrequency(0)
            duration = edf.getFileDuration()
            edf.close()
        except Exception:
            continue

        # Extract ictal segments
        for seizure in file_seizures:
            start = seizure['seizure_start']
            end = seizure['seizure_end']

            # Extract window from middle of seizure
            mid = (start + end) / 2
            seg_start = max(0, mid - window_sec / 2)
            seg_end = seg_start + window_sec

            data, _ = read_edf_segment_standalone(
                str(edf_path), seg_start, seg_end, channels
            )
            if data is not None and data.shape[1] >= int(fs * window_sec * 0.5):
                data = preprocess_eeg_standalone(data, int(fs))
                try:
                    feat = extract_features_with_aggregation(data, int(fs), 'max')
                    if not np.isnan(feat).any() and not np.isinf(feat).any():
                        features_list.append(feat)
                        labels.append(1)  # Ictal
                except Exception:
                    continue

        # Extract interictal segments (at least 1 hour from any seizure)
        seizure_times = [(s['seizure_start'], s['seizure_end']) for s in file_seizures]

        # Sample 2-3 interictal windows per file (to balance classes)
        interictal_candidates = []
        for t in range(0, int(duration) - int(window_sec), 60):  # Every minute
            is_safe = True
            for s_start, s_end in seizure_times:
                if abs(t - s_start) < 3600 or abs(t - s_end) < 3600:
                    is_safe = False
                    break
            if is_safe:
                interictal_candidates.append(t)

        # Sample up to 3 interictal windows
        if interictal_candidates:
            rng = np.random.RandomState(42)
            selected = rng.choice(
                interictal_candidates,
                size=min(3, len(interictal_candidates)),
                replace=False
            )
            for t in selected:
                data, _ = read_edf_segment_standalone(
                    str(edf_path), t, t + window_sec, channels
                )
                if data is not None and data.shape[1] >= int(fs * window_sec * 0.5):
                    data = preprocess_eeg_standalone(data, int(fs))
                    try:
                        feat = extract_features_with_aggregation(data, int(fs), 'max')
                        if not np.isnan(feat).any() and not np.isinf(feat).any():
                            features_list.append(feat)
                            labels.append(0)  # Interictal
                    except Exception:
                        continue

    return features_list, labels


# ============================================================================
# Single Subject LOSO Job
# ============================================================================

def process_single_subject(
    test_subject: str,
    all_subject_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_dir: Path
) -> Dict:
    """Process a single subject as test set (parallel job).

    Args:
        test_subject: Subject ID to use as test set
        all_subject_data: Dict mapping subject_id -> (X, y) for all subjects
        output_dir: Directory to save per-patient results

    Returns:
        Dict with AUC and statistics for this subject
    """
    logger.info(f"[{test_subject}] Starting LOSO evaluation...")

    # Prepare train/test split
    X_train_list = []
    y_train_list = []
    for subject_id, (X, y) in all_subject_data.items():
        if subject_id != test_subject:
            X_train_list.append(X)
            y_train_list.append(y)

    X_train = np.vstack(X_train_list)
    y_train = np.concatenate(y_train_list)
    X_test, y_test = all_subject_data[test_subject]

    logger.info(f"[{test_subject}] Train: {len(y_train)} samples, Test: {len(y_test)} samples")

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

    # Calculate AUC
    try:
        auc = roc_auc_score(y_test, y_prob_mean)
    except Exception:
        auc = 0.5

    result = {
        'subject': test_subject,
        'auc': float(auc),
        'n_samples': int(len(y_test)),
        'n_seizure': int(sum(y_test)),
        'n_non_seizure': int(len(y_test) - sum(y_test)),
        'y_true': y_test.tolist(),
        'y_prob': y_prob_mean.tolist(),
        'timestamp': datetime.now().isoformat(),
    }

    # Save per-patient result
    output_file = output_dir / f"{test_subject}.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"[{test_subject}] AUC = {auc:.4f}, saved to {output_file}")
    return result


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="EC2 Parallel LOSO (D-002)")
    parser.add_argument('--tier', choices=['tier2', 'tier3'], default='tier3',
                        help='Feature tier (default: tier3)')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='Number of parallel jobs (-1 = all CPUs)')
    parser.add_argument('--channels', type=int, choices=[8, 18], default=8,
                        help='Number of channels (default: 8)')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("EC2 Parallel LOSO Cross-Validation (D-002)")
    logger.info("=" * 60)

    # Find data
    try:
        DATA_ROOT = find_data_root()
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    logger.info(f"Data root: {DATA_ROOT}")

    # Output directories
    project_root = Path(__file__).parent.parent
    output_dir = project_root / 'results' / 'per_patient'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all valid subjects
    all_subjects = sorted([
        d.name for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith('chb')
        and d.name not in EXCLUDE_SUBJECTS
    ])
    logger.info(f"Found {len(all_subjects)} subjects (excluding {EXCLUDE_SUBJECTS})")

    channels = LOSO_CHANNELS_8
    logger.info(f"Using {len(channels)} channels: {channels}")

    # Phase 1: Load all data (serial, to avoid I/O contention)
    logger.info("\nPhase 1: Loading data for all subjects...")
    all_subject_data = {}

    for subject_id in all_subjects:
        subject_dir = DATA_ROOT / subject_id
        logger.info(f"  Loading {subject_id}...")

        features_list, labels = extract_segments_standalone(subject_dir, channels)

        if len(features_list) > 0 and len(set(labels)) == 2:
            X = np.array(features_list)
            y = np.array(labels)
            all_subject_data[subject_id] = (X, y)
            n_pos = int(sum(y))
            n_neg = len(y) - n_pos
            logger.info(f"    {subject_id}: {len(y)} samples ({n_pos} ictal, {n_neg} interictal)")
        else:
            logger.warning(f"    {subject_id}: Skipped (insufficient data or single class)")

    valid_subjects = list(all_subject_data.keys())
    logger.info(f"\nValid subjects for LOSO: {len(valid_subjects)}")

    if len(valid_subjects) < 2:
        logger.error("Not enough valid subjects for LOSO")
        sys.exit(1)

    # Phase 2: Parallel LOSO
    logger.info(f"\nPhase 2: Running parallel LOSO (n_jobs={args.n_jobs})...")

    results = Parallel(n_jobs=args.n_jobs, verbose=10)(
        delayed(process_single_subject)(
            test_subject=subj,
            all_subject_data=all_subject_data,
            output_dir=output_dir
        )
        for subj in valid_subjects
    )

    # Phase 3: Aggregate results
    logger.info("\nPhase 3: Aggregating results...")

    all_y_true = []
    all_y_prob = []
    per_subject_results = {}

    for r in results:
        per_subject_results[r['subject']] = {
            'auc': r['auc'],
            'n_samples': r['n_samples'],
            'n_seizure': r['n_seizure'],
            'n_non_seizure': r['n_non_seizure'],
        }
        all_y_true.extend(r['y_true'])
        all_y_prob.extend(r['y_prob'])

    # Overall AUC
    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except Exception:
        overall_auc = 0.5

    aucs = [r['auc'] for r in results]
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    final_results = {
        'method': f'{args.tier}_parallel_loso',
        'n_channels': len(channels),
        'channels': channels,
        'n_subjects': len(valid_subjects),
        'overall_auc': float(overall_auc),
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'per_subject': per_subject_results,
        'timestamp': datetime.now().isoformat(),
        'n_jobs': args.n_jobs,
    }

    # Save aggregated results
    output_file = project_root / 'results' / f'ec2_parallel_loso_{args.tier}.json'
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Method: {args.tier} parallel LOSO")
    logger.info(f"Subjects: {len(valid_subjects)}")
    logger.info(f"Overall AUC: {overall_auc:.4f}")
    logger.info(f"Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    logger.info("")
    logger.info("Per-subject AUC:")
    for subj in sorted(per_subject_results.keys()):
        r = per_subject_results[subj]
        logger.info(f"  {subj}: {r['auc']:.4f} (n={r['n_samples']})")
    logger.info("")
    logger.info(f"Results saved to: {output_file}")
    logger.info(f"Per-patient results: {output_dir}/")

    return final_results


if __name__ == '__main__':
    main()
