#!/usr/bin/env python3
"""
================================================================================
PROMPT 018 - FULL COHORT POLARITY ANALYSIS (FIX2)
================================================================================

SageMaker-compatible training script for full CHB-MIT cohort analysis.

Three-tier evaluation:
- Tier 2A: Minimal eigenvalues (8 features) - fair quantum comparator
- Tier 2B: Full MichaelHills features (~368 features) - classical ceiling
- Tier 3: Riemannian TangentSpace + LDA (Ledoit-Wolf regularization)

All tiers perform:
- LOSO cross-validation
- Per-subject AUC computation
- Polarity calibration analysis

Four-phase execution with checkpointing:
- Phase 1: Feature extraction (~25 min)
- Phase 2: Validation gate (~3 min)
- Phase 3: Full cohort LOSO (~8 min)
- Phase 4: Analysis + report (~1 min)

Environment:
    SM_CHANNEL_DATA: Path to input data (CHB-MIT EDF files)
    SM_MODEL_DIR: Path to save model outputs
    SM_OUTPUT_DATA_DIR: Path for additional outputs

Usage (local validation):
    python sagemaker/train_full_cohort.py --local --data-dir H:/Data/PythonDNU/EEG/chbmit

================================================================================
"""

import subprocess
import sys
import traceback

# Install dependencies only if running in SageMaker
import os
if os.environ.get('SM_CHANNEL_DATA'):
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'pyedflib'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'pyriemann'])
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'xgboost'])
    print("Dependencies installed.")

import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
import logging
from scipy import signal
from scipy.signal import welch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

FS = 256  # Hz

# Segment parameters
PREICTAL_DURATION = 60
INTERICTAL_MIN_GAP = 300
WINDOW_SEC = 20.0  # 20-second windows (PROMPT 015 optimal)
MAX_INTERICTAL_PER_FILE = 5

# 8 channels for quantum compatibility (PROMPT 006 configuration)
QUANTUM_CHANNELS = [
    "FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
    "FP2-F8", "F8-T8", "FP2-F4", "F4-C4",
]

# Subjects to exclude:
# - chb12: Major montage changes mid-recording
# - chb24: Re-recording of chb01
EXCLUDE_SUBJECTS = {'chb12', 'chb24'}

# Validation subjects (matches PROMPT 006 hardware validation)
VALIDATION_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']

# FIX2: Updated validation thresholds for dual Tier 2 variants
VALIDATION_THRESHOLDS = {
    # Tier 2A: minimal eigenvalues (quantum comparator)
    'tier2a_min': 0.62,        # Local dry run got 0.666
    'tier2a_max': 0.80,        # Ceiling — if above this, feature leak

    # Tier 2B: full features (classical ceiling)
    'tier2b_min': 0.84,        # PROMPT 015 got 0.873

    # Tier 3: Riemannian
    'tier3_raw_max': 0.55,     # Must show polarity drag
    'tier3_cal_min': 0.65,     # Must recover with calibration
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class EEGSegment:
    """A labeled EEG segment."""
    subject: str
    source_file: str
    label: str
    start_sec: float
    end_sec: float
    data: Optional[np.ndarray] = field(default=None, repr=False)
    fs: float = 256.0


@dataclass
class PatientFeatures:
    """Extracted features for a single patient."""
    subject: str
    tier2a_X: np.ndarray  # (n_samples, 8) eigenvalues only
    tier2b_X: np.ndarray  # (n_samples, ~368) full MichaelHills features
    tier3_covs: np.ndarray  # (n_samples, n_ch, n_ch) covariance matrices
    y: np.ndarray  # (n_samples,) labels
    n_seizures: int
    n_segments: int


# =============================================================================
# PREPROCESSING
# =============================================================================

def preprocess_eeg(data: np.ndarray, fs: int = 256,
                   lowcut: float = 0.5, highcut: float = 40.0,
                   notch_freq: float = 60.0) -> np.ndarray:
    """Preprocess EEG: demean, bandpass, notch, z-score."""
    data = data.copy().astype(np.float64)
    data -= np.mean(data, axis=1, keepdims=True)

    nyq = fs / 2.0
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)

    if high > low > 0:
        try:
            b, a = signal.butter(5, [low, high], btype='band')
            data = signal.filtfilt(b, a, data, axis=1)
        except ValueError:
            pass

    if 0 < notch_freq < nyq:
        try:
            b_n, a_n = signal.iirnotch(notch_freq, Q=30, fs=fs)
            data = signal.filtfilt(b_n, a_n, data, axis=1)
        except ValueError:
            pass

    # Z-score normalization per channel
    std = np.std(data, axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    data = data / std

    return data


# =============================================================================
# FEATURE EXTRACTION - TIER 2A (MINIMAL EIGENVALUES)
# =============================================================================

def compute_covariance_ledoit_wolf(data: np.ndarray) -> np.ndarray:
    """Compute covariance matrix with Ledoit-Wolf shrinkage."""
    from sklearn.covariance import LedoitWolf

    n_ch, n_samples = data.shape
    data = data - np.mean(data, axis=1, keepdims=True)

    # Transpose for sklearn (samples x features)
    lw = LedoitWolf()
    lw.fit(data.T)
    cov = lw.covariance_

    return cov


def extract_eigenvalues(cov: np.ndarray) -> np.ndarray:
    """Extract sorted eigenvalues from covariance matrix."""
    eigvals = np.real(np.linalg.eigvals(cov))
    eigvals = np.sort(eigvals)[::-1]  # Descending order
    return np.log10(eigvals + 1e-12)  # Log transform


def make_spd(cov: np.ndarray) -> np.ndarray:
    """Ensure matrix is symmetric positive definite."""
    cov = (cov + cov.T) / 2
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-6)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


# =============================================================================
# FEATURE EXTRACTION - TIER 2B (FULL MICHAELHILLS)
# =============================================================================

def extract_michaelhills_subwindow(subwindow: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Extract MichaelHills features from a 1-second sub-window.

    Features per sub-window (8 channels):
    - FFT magnitude 1-40 Hz: 8 * 40 = 320
    - Frequency-domain correlation eigenvalues: 8
    - Time-domain correlation eigenvalues: 8
    Total: 336 features
    """
    n_channels = subwindow.shape[0]

    # FFT magnitude (1-40 Hz)
    fft_result = np.fft.rfft(subwindow, axis=1)
    fft_mag = np.abs(fft_result)[:, 1:41]  # 40 frequency bins
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


def extract_tier2b_features(window: np.ndarray, fs: int = FS,
                            aggregation: str = 'max') -> np.ndarray:
    """
    Extract full MichaelHills features with sub-window aggregation.

    For 20s window at 256 Hz = 20 sub-windows of 1s each.
    MAX aggregation across sub-windows.
    """
    n_channels, n_samples = window.shape
    samples_per_subwindow = fs  # 1 second
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


# =============================================================================
# EDF READING
# =============================================================================

def normalize_channel_label(label: str) -> str:
    """Normalize channel label for matching."""
    label = label.strip().upper()
    for prefix in ['EEG ', 'EEG-']:
        if label.startswith(prefix):
            label = label[len(prefix):]
    label = label.replace('--', '-')
    return label


def read_edf_segment(filepath: str, start_sec: float, end_sec: float,
                     target_channels: List[str] = None) -> Tuple[Optional[np.ndarray], float]:
    """Read a segment from an EDF file with channel selection."""
    import pyedflib

    try:
        with pyedflib.EdfReader(str(filepath)) as f:
            n_signals = f.signals_in_file
            fs = f.getSampleFrequency(0)

            start_sample = int(start_sec * fs)
            end_sample = int(end_sec * fs)
            n_samples = end_sample - start_sample

            if n_samples <= 0:
                return None, fs

            if target_channels is None:
                channel_indices = list(range(n_signals))
            else:
                label_to_idx = {}
                for i in range(n_signals):
                    label = normalize_channel_label(f.getLabel(i))
                    label_to_idx[label] = i

                channel_indices = []
                for target in target_channels:
                    target_norm = normalize_channel_label(target)
                    if target_norm in label_to_idx:
                        channel_indices.append(label_to_idx[target_norm])
                    else:
                        return None, fs

            n_ch = len(channel_indices)
            data = np.zeros((n_ch, n_samples), dtype=np.float32)

            for out_idx, ch_idx in enumerate(channel_indices):
                try:
                    sig = f.readSignal(ch_idx, start=start_sample, n=n_samples)
                    if len(sig) == n_samples:
                        data[out_idx] = sig
                    elif len(sig) > 0:
                        data[out_idx, :len(sig)] = sig
                except Exception:
                    pass

            return data, fs

    except Exception as e:
        return None, 256.0


# =============================================================================
# SUMMARY PARSING
# =============================================================================

def parse_summary_file(summary_path: Path) -> List[Dict]:
    """Parse CHB-MIT summary file to extract seizure info."""
    seizures = []

    if not summary_path.exists():
        return seizures

    try:
        with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception:
        return seizures

    current_file = None
    lines = content.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()

        if line.startswith('File Name:'):
            current_file = line.split(':')[1].strip()

        elif 'Seizure' in line and 'Start' in line and current_file:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    time_part = parts[-1].strip()
                    start_sec = int(time_part.replace('seconds', '').strip())

                    for j in range(i + 1, min(i + 5, len(lines))):
                        if 'End' in lines[j]:
                            end_parts = lines[j].split(':')
                            if len(end_parts) >= 2:
                                end_time = end_parts[-1].strip()
                                end_sec = int(end_time.replace('seconds', '').strip())

                                seizures.append({
                                    'file': current_file,
                                    'start': start_sec,
                                    'end': end_sec
                                })
                            break
            except (ValueError, IndexError):
                pass

    return seizures


# =============================================================================
# SEGMENT EXTRACTION
# =============================================================================

def extract_segments_for_subject(subject_dir: Path) -> List[EEGSegment]:
    """Extract ictal, preictal, and interictal segments for a subject."""
    segments = []
    subject = subject_dir.name

    summary_files = list(subject_dir.glob('*-summary.txt'))
    if not summary_files:
        return segments

    seizures = parse_summary_file(summary_files[0])

    seizure_files = {}
    for sz in seizures:
        fname = sz['file']
        if fname not in seizure_files:
            seizure_files[fname] = []
        seizure_files[fname].append(sz)

    edf_files = sorted(subject_dir.glob('*.edf'))

    for edf_path in edf_files:
        fname = edf_path.name

        if fname in seizure_files:
            for sz in seizure_files[fname]:
                # Ictal segment
                segments.append(EEGSegment(
                    subject=subject,
                    source_file=fname,
                    label='ictal',
                    start_sec=sz['start'],
                    end_sec=sz['end']
                ))

                # Preictal segment
                preictal_end = sz['start']
                preictal_start = max(0, preictal_end - PREICTAL_DURATION)
                if preictal_start < preictal_end:
                    segments.append(EEGSegment(
                        subject=subject,
                        source_file=fname,
                        label='preictal',
                        start_sec=preictal_start,
                        end_sec=preictal_end
                    ))

        else:
            # Interictal segments
            file_duration = 3600
            n_interictal = min(MAX_INTERICTAL_PER_FILE, int(file_duration / WINDOW_SEC / 4))

            for i in range(n_interictal):
                start = i * WINDOW_SEC * 2
                end = start + WINDOW_SEC
                if end < file_duration:
                    segments.append(EEGSegment(
                        subject=subject,
                        source_file=fname,
                        label='interictal',
                        start_sec=start,
                        end_sec=end
                    ))

    return segments


# =============================================================================
# XGBOOST MODEL
# =============================================================================

def make_xgb_model(seed=42):
    """Create XGBoost classifier matching PROMPT 015 configuration."""
    from xgboost import XGBClassifier

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


# =============================================================================
# POLARITY CALIBRATION
# =============================================================================

def calibrate_auc(raw_auc: float) -> Tuple[float, str]:
    """Apply polarity calibration to a single AUC value."""
    if raw_auc < 0.5:
        return 1.0 - raw_auc, 'inverted'
    else:
        return raw_auc, 'standard'


# =============================================================================
# PHASE 1: FEATURE EXTRACTION
# =============================================================================

def phase1_extract_features(data_dir: Path, output_dir: Path,
                            subjects: List[Path]) -> Dict[str, PatientFeatures]:
    """
    Phase 1: Load and extract features for all patients.

    Returns dict mapping patient_id -> PatientFeatures
    """
    logger.info("Phase 1: Loading patients...")
    start_time = datetime.now()

    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    patient_features = {}

    for subject_dir in subjects:
        subj = subject_dir.name
        subj_start = datetime.now()

        segments = extract_segments_for_subject(subject_dir)
        if not segments:
            logger.info(f"Phase 1: {subj} skipped (no segments)")
            continue

        tier2a_features = []  # Eigenvalues only
        tier2b_features = []  # Full MichaelHills
        tier3_covs = []
        labels = []
        n_seizures = 0

        for seg in segments:
            edf_path = data_dir / seg.subject / seg.source_file
            if not edf_path.exists():
                continue

            data, fs = read_edf_segment(str(edf_path), seg.start_sec, seg.end_sec,
                                        target_channels=QUANTUM_CHANNELS)

            if data is None or data.size == 0:
                continue

            min_samples = int(WINDOW_SEC * fs * 0.5)
            if data.shape[1] < min_samples:
                continue

            # Preprocess
            data = preprocess_eeg(data, fs=int(fs))

            # Tier 2A: Eigenvalues only
            cov = compute_covariance_ledoit_wolf(data)
            eigvals = extract_eigenvalues(cov)
            tier2a_features.append(eigvals)

            # Tier 2B: Full MichaelHills features
            full_features = extract_tier2b_features(data, fs=int(fs), aggregation='max')
            tier2b_features.append(full_features)

            # Tier 3: SPD covariance
            cov_spd = make_spd(cov)
            tier3_covs.append(cov_spd)

            # Label
            label = label_map.get(seg.label, 0)
            labels.append(label)
            if label == 1:
                n_seizures += 1

        if not tier2a_features:
            logger.info(f"Phase 1: {subj} skipped (no valid segments)")
            continue

        patient_features[subj] = PatientFeatures(
            subject=subj,
            tier2a_X=np.array(tier2a_features),
            tier2b_X=np.array(tier2b_features),
            tier3_covs=np.array(tier3_covs),
            y=np.array(labels),
            n_seizures=n_seizures,
            n_segments=len(labels),
        )

        elapsed = (datetime.now() - subj_start).total_seconds()
        n_feat_2b = tier2b_features[0].shape[0] if tier2b_features else 0
        logger.info(f"Phase 1: {subj} loaded ({n_seizures} sz, "
                    f"{len(labels)} seg, {n_feat_2b} T2B feat, {elapsed:.1f}s)")

    # Checkpoint
    checkpoint = {
        'patients': list(patient_features.keys()),
        'n_patients': len(patient_features),
        'n_segments': sum(pf.n_segments for pf in patient_features.values()),
        'tier2a_features': 8,
        'tier2b_features': patient_features[list(patient_features.keys())[0]].tier2b_X.shape[1] if patient_features else 0,
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / 'checkpoint_features.json', 'w') as f:
        json.dump(checkpoint, f, indent=2)

    total_time = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"Phase 1 complete: {len(patient_features)} patients, "
                f"{checkpoint['n_segments']} segments, "
                f"T2B={checkpoint['tier2b_features']} features ({total_time:.1f} min)")

    return patient_features


# =============================================================================
# TIER 2A LOSO (MINIMAL EIGENVALUES)
# =============================================================================

def run_tier2a_loso(patient_features: Dict[str, PatientFeatures],
                    subjects_subset: List[str] = None) -> Dict[str, Any]:
    """Run Tier 2A LOSO with eigenvalues only + XGBoost."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    if subjects_subset:
        pf = {k: v for k, v in patient_features.items() if k in subjects_subset}
    else:
        pf = patient_features

    subjects = sorted(pf.keys())
    results = {'per_subject': {}, 'raw_aucs': [], 'calibrated_aucs': []}

    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]

        X_train = np.vstack([pf[s].tier2a_X for s in train_subjects])
        y_train = np.concatenate([pf[s].y for s in train_subjects])
        X_test = pf[test_subject].tier2a_X
        y_test = pf[test_subject].y

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = make_xgb_model()
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        try:
            raw_auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            raw_auc = 0.5

        cal_auc, polarity = calibrate_auc(raw_auc)

        results['per_subject'][test_subject] = {
            'raw_auc': float(raw_auc),
            'calibrated_auc': float(cal_auc),
            'polarity': polarity,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_seizure': int(np.sum(y_test)),
        }

        results['raw_aucs'].append(raw_auc)
        results['calibrated_aucs'].append(cal_auc)

    if results['raw_aucs']:
        results['overall_raw_auc'] = float(np.mean(results['raw_aucs']))
        results['overall_calibrated_auc'] = float(np.mean(results['calibrated_aucs']))
        results['calibration_lift'] = results['overall_calibrated_auc'] - results['overall_raw_auc']
        results['n_inverted'] = sum(1 for d in results['per_subject'].values()
                                    if d['polarity'] == 'inverted')

    return results


# =============================================================================
# TIER 2B LOSO (FULL MICHAELHILLS)
# =============================================================================

def run_tier2b_loso(patient_features: Dict[str, PatientFeatures],
                    subjects_subset: List[str] = None) -> Dict[str, Any]:
    """Run Tier 2B LOSO with full MichaelHills features + XGBoost."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    if subjects_subset:
        pf = {k: v for k, v in patient_features.items() if k in subjects_subset}
    else:
        pf = patient_features

    subjects = sorted(pf.keys())
    results = {'per_subject': {}, 'raw_aucs': [], 'calibrated_aucs': []}

    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]

        X_train = np.vstack([pf[s].tier2b_X for s in train_subjects])
        y_train = np.concatenate([pf[s].y for s in train_subjects])
        X_test = pf[test_subject].tier2b_X
        y_test = pf[test_subject].y

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # Handle any NaN/Inf in features
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=10.0, neginf=-10.0)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = make_xgb_model()
        model.fit(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        try:
            raw_auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            raw_auc = 0.5

        cal_auc, polarity = calibrate_auc(raw_auc)

        results['per_subject'][test_subject] = {
            'raw_auc': float(raw_auc),
            'calibrated_auc': float(cal_auc),
            'polarity': polarity,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_seizure': int(np.sum(y_test)),
        }

        results['raw_aucs'].append(raw_auc)
        results['calibrated_aucs'].append(cal_auc)

    if results['raw_aucs']:
        results['overall_raw_auc'] = float(np.mean(results['raw_aucs']))
        results['overall_calibrated_auc'] = float(np.mean(results['calibrated_aucs']))
        results['calibration_lift'] = results['overall_calibrated_auc'] - results['overall_raw_auc']
        results['n_inverted'] = sum(1 for d in results['per_subject'].values()
                                    if d['polarity'] == 'inverted')

    return results


# =============================================================================
# TIER 3 LOSO (RIEMANNIAN)
# =============================================================================

def run_tier3_loso(patient_features: Dict[str, PatientFeatures],
                   subjects_subset: List[str] = None) -> Dict[str, Any]:
    """Run Tier 3 LOSO with TangentSpace + LDA."""
    try:
        from pyriemann.tangentspace import TangentSpace
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {'error': 'pyriemann not installed'}

    if subjects_subset:
        pf = {k: v for k, v in patient_features.items() if k in subjects_subset}
    else:
        pf = patient_features

    subjects = sorted(pf.keys())
    results = {'per_subject': {}, 'raw_aucs': [], 'calibrated_aucs': []}

    for test_subject in subjects:
        train_subjects = [s for s in subjects if s != test_subject]

        X_train_cov = np.vstack([pf[s].tier3_covs for s in train_subjects])
        y_train = np.concatenate([pf[s].y for s in train_subjects])
        X_test_cov = pf[test_subject].tier3_covs
        y_test = pf[test_subject].y

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        try:
            ts = TangentSpace()
            X_train = ts.fit_transform(X_train_cov)
            X_test = ts.transform(X_test_cov)

            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            lda.fit(X_train, y_train)
            y_proba = lda.predict_proba(X_test)[:, 1]

            raw_auc = roc_auc_score(y_test, y_proba)
        except Exception as e:
            continue

        cal_auc, polarity = calibrate_auc(raw_auc)

        results['per_subject'][test_subject] = {
            'raw_auc': float(raw_auc),
            'calibrated_auc': float(cal_auc),
            'polarity': polarity,
            'n_train': len(X_train_cov),
            'n_test': len(X_test_cov),
            'n_seizure': int(np.sum(y_test)),
        }

        results['raw_aucs'].append(raw_auc)
        results['calibrated_aucs'].append(cal_auc)

    if results['raw_aucs']:
        results['overall_raw_auc'] = float(np.mean(results['raw_aucs']))
        results['overall_calibrated_auc'] = float(np.mean(results['calibrated_aucs']))
        results['calibration_lift'] = results['overall_calibrated_auc'] - results['overall_raw_auc']
        results['n_inverted'] = sum(1 for d in results['per_subject'].values()
                                    if d['polarity'] == 'inverted')

    return results


# =============================================================================
# PHASE 2: VALIDATION GATE
# =============================================================================

def phase2_validation_gate(patient_features: Dict[str, PatientFeatures],
                           output_dir: Path) -> Tuple[bool, Dict]:
    """
    Phase 2: Validation gate on 7 reference subjects.

    Returns (passed, results_dict)
    """
    logger.info("Phase 2: Validation gate (7 reference patients)...")
    start_time = datetime.now()

    # Filter to validation subjects
    val_subjects = [s for s in VALIDATION_SUBJECTS if s in patient_features]
    if len(val_subjects) < 7:
        logger.warning(f"Only {len(val_subjects)}/7 validation subjects found")

    # Run all three tiers
    tier2a = run_tier2a_loso(patient_features, val_subjects)
    tier2a_auc = tier2a.get('overall_calibrated_auc', 0.0)

    tier2b = run_tier2b_loso(patient_features, val_subjects)
    tier2b_auc = tier2b.get('overall_calibrated_auc', 0.0)

    tier3 = run_tier3_loso(patient_features, val_subjects)
    tier3_raw = tier3.get('overall_raw_auc', 0.0)
    tier3_cal = tier3.get('overall_calibrated_auc', 0.0)

    # Check thresholds
    checks = {
        'tier2a_calibrated_min': {
            'actual': tier2a_auc,
            'threshold': VALIDATION_THRESHOLDS['tier2a_min'],
            'comparison': '>=',
            'passed': tier2a_auc >= VALIDATION_THRESHOLDS['tier2a_min'],
        },
        'tier2a_calibrated_max': {
            'actual': tier2a_auc,
            'threshold': VALIDATION_THRESHOLDS['tier2a_max'],
            'comparison': '<=',
            'passed': tier2a_auc <= VALIDATION_THRESHOLDS['tier2a_max'],
        },
        'tier2b_calibrated': {
            'actual': tier2b_auc,
            'threshold': VALIDATION_THRESHOLDS['tier2b_min'],
            'comparison': '>=',
            'passed': tier2b_auc >= VALIDATION_THRESHOLDS['tier2b_min'],
        },
        'tier3_raw': {
            'actual': tier3_raw,
            'threshold': VALIDATION_THRESHOLDS['tier3_raw_max'],
            'comparison': '<=',
            'passed': tier3_raw <= VALIDATION_THRESHOLDS['tier3_raw_max'],
        },
        'tier3_calibrated': {
            'actual': tier3_cal,
            'threshold': VALIDATION_THRESHOLDS['tier3_cal_min'],
            'comparison': '>=',
            'passed': tier3_cal >= VALIDATION_THRESHOLDS['tier3_cal_min'],
        },
    }

    all_passed = all(c['passed'] for c in checks.values())

    validation_results = {
        'passed': all_passed,
        'checks': checks,
        'tier2a_results': tier2a,
        'tier2b_results': tier2b,
        'tier3_results': tier3,
        'validation_subjects': val_subjects,
        'timestamp': datetime.now().isoformat(),
    }

    # Checkpoint
    checkpoint_file = 'checkpoint_validation.json' if all_passed else 'validation_failed.json'
    with open(output_dir / checkpoint_file, 'w') as f:
        json.dump(validation_results, f, indent=2)

    elapsed = (datetime.now() - start_time).total_seconds()

    if all_passed:
        logger.info(f"Phase 2: PASSED (T2A={tier2a_auc:.3f}, T2B={tier2b_auc:.3f}, "
                    f"T3_raw={tier3_raw:.3f}, T3_cal={tier3_cal:.3f}) [{elapsed:.1f}s]")
    else:
        failed = [k for k, v in checks.items() if not v['passed']]
        logger.error(f"Phase 2: FAILED - {failed}")
        for check_name, check_data in checks.items():
            status = "PASS" if check_data['passed'] else "FAIL"
            logger.info(f"  {check_name}: {check_data['actual']:.3f} "
                        f"{check_data['comparison']} {check_data['threshold']} [{status}]")

    return all_passed, validation_results


# =============================================================================
# PHASE 3: FULL COHORT LOSO
# =============================================================================

def phase3_full_loso(patient_features: Dict[str, PatientFeatures],
                     output_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """
    Phase 3: Full cohort LOSO with incremental checkpointing.

    Returns (tier2a_results, tier2b_results, tier3_results)
    """
    logger.info(f"Phase 3: Full cohort LOSO ({len(patient_features)} patients)...")
    start_time = datetime.now()

    subjects = sorted(patient_features.keys())

    # Initialize progress tracking
    progress = {
        'completed_patients': [],
        'tier2a_results': {},
        'tier2b_results': {},
        'tier3_results': {},
        'last_updated': datetime.now().isoformat(),
    }

    # Run LOSO with per-fold checkpointing
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    try:
        from pyriemann.tangentspace import TangentSpace
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        has_pyriemann = True
    except ImportError:
        has_pyriemann = False

    for fold_idx, test_subject in enumerate(subjects, 1):
        train_subjects = [s for s in subjects if s != test_subject]
        pf = patient_features

        # Tier 2A (eigenvalues only)
        X_train = np.vstack([pf[s].tier2a_X for s in train_subjects])
        y_train = np.concatenate([pf[s].y for s in train_subjects])
        X_test = pf[test_subject].tier2a_X
        y_test = pf[test_subject].y

        if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = make_xgb_model()
            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            try:
                t2a_raw = roc_auc_score(y_test, y_proba)
            except ValueError:
                t2a_raw = 0.5

            t2a_cal, t2a_pol = calibrate_auc(t2a_raw)

            progress['tier2a_results'][test_subject] = {
                'raw_auc': float(t2a_raw),
                'calibrated_auc': float(t2a_cal),
                'polarity': t2a_pol,
                'n_test': len(X_test),
                'n_seizure': int(np.sum(y_test)),
            }

        # Tier 2B (full features)
        X_train = np.vstack([pf[s].tier2b_X for s in train_subjects])
        X_test = pf[test_subject].tier2b_X

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=10.0, neginf=-10.0)

        if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = make_xgb_model()
            model.fit(X_train_scaled, y_train)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]

            try:
                t2b_raw = roc_auc_score(y_test, y_proba)
            except ValueError:
                t2b_raw = 0.5

            t2b_cal, t2b_pol = calibrate_auc(t2b_raw)

            progress['tier2b_results'][test_subject] = {
                'raw_auc': float(t2b_raw),
                'calibrated_auc': float(t2b_cal),
                'polarity': t2b_pol,
                'n_test': len(X_test),
                'n_seizure': int(np.sum(y_test)),
            }

        # Tier 3 (Riemannian)
        if has_pyriemann:
            X_train_cov = np.vstack([pf[s].tier3_covs for s in train_subjects])
            y_train = np.concatenate([pf[s].y for s in train_subjects])
            X_test_cov = pf[test_subject].tier3_covs
            y_test = pf[test_subject].y

            if len(np.unique(y_train)) >= 2 and len(np.unique(y_test)) >= 2:
                try:
                    ts = TangentSpace()
                    X_train_ts = ts.fit_transform(X_train_cov)
                    X_test_ts = ts.transform(X_test_cov)

                    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                    lda.fit(X_train_ts, y_train)
                    y_proba = lda.predict_proba(X_test_ts)[:, 1]

                    t3_raw = roc_auc_score(y_test, y_proba)
                    t3_cal, t3_pol = calibrate_auc(t3_raw)

                    progress['tier3_results'][test_subject] = {
                        'raw_auc': float(t3_raw),
                        'calibrated_auc': float(t3_cal),
                        'polarity': t3_pol,
                        'n_test': len(X_test_cov),
                        'n_seizure': int(np.sum(y_test)),
                    }
                except Exception:
                    pass

        progress['completed_patients'].append(test_subject)
        progress['last_updated'] = datetime.now().isoformat()

        # Save incremental checkpoint
        with open(output_dir / 'checkpoint_loso_progress.json', 'w') as f:
            json.dump(progress, f, indent=2)

        # Log progress
        t2a_str = f"T2A={progress['tier2a_results'].get(test_subject, {}).get('raw_auc', 0):.2f}"
        t2b_str = f"T2B={progress['tier2b_results'].get(test_subject, {}).get('raw_auc', 0):.2f}"
        t3_str = f"T3={progress['tier3_results'].get(test_subject, {}).get('raw_auc', 0):.2f}"
        logger.info(f"Phase 3: Fold {fold_idx}/{len(subjects)} ({test_subject}) {t2a_str} {t2b_str} {t3_str}")

    # Compute overall metrics for each tier
    def compute_overall(results_dict):
        results = {
            'per_subject': results_dict,
            'raw_aucs': [d['raw_auc'] for d in results_dict.values()],
            'calibrated_aucs': [d['calibrated_auc'] for d in results_dict.values()],
        }
        if results['raw_aucs']:
            results['overall_raw_auc'] = float(np.mean(results['raw_aucs']))
            results['overall_calibrated_auc'] = float(np.mean(results['calibrated_aucs']))
            results['calibration_lift'] = results['overall_calibrated_auc'] - results['overall_raw_auc']
            results['n_inverted'] = sum(1 for d in results['per_subject'].values()
                                        if d['polarity'] == 'inverted')
        return results

    tier2a_results = compute_overall(progress['tier2a_results'])
    tier2b_results = compute_overall(progress['tier2b_results'])
    tier3_results = compute_overall(progress['tier3_results'])

    elapsed = (datetime.now() - start_time).total_seconds() / 60
    logger.info(f"Phase 3 complete: {len(subjects)} folds ({elapsed:.1f} min)")

    return tier2a_results, tier2b_results, tier3_results


# =============================================================================
# PHASE 4: ANALYSIS + REPORT
# =============================================================================

def phase4_analysis(tier2a_results: Dict, tier2b_results: Dict, tier3_results: Dict,
                    patient_features: Dict[str, PatientFeatures],
                    output_dir: Path) -> Dict:
    """Phase 4: Analysis and report generation."""
    logger.info("Phase 4: Analysis + report generation...")
    start_time = datetime.now()

    # Polarity concordance
    concordance = analyze_polarity_concordance(tier2a_results, tier2b_results, tier3_results)

    # Subgroup analysis
    subgroups = analyze_subgroups(tier2a_results, tier2b_results, tier3_results)

    # Wilcoxon signed-rank tests
    wilcoxon_tests = run_wilcoxon_tests(tier2a_results, tier2b_results, tier3_results)

    # Full results
    n_tier2b_features = patient_features[list(patient_features.keys())[0]].tier2b_X.shape[1]

    full_results = {
        'config': {
            'channels': QUANTUM_CHANNELS,
            'n_channels': len(QUANTUM_CHANNELS),
            'window_sec': WINDOW_SEC,
            'exclude_subjects': list(EXCLUDE_SUBJECTS),
            'validation_thresholds': VALIDATION_THRESHOLDS,
        },
        'n_subjects': len(patient_features),
        'n_segments': sum(pf.n_segments for pf in patient_features.values()),
        'tier2a_eigenvalues': {
            'description': 'Minimal 8 eigenvalues - fair quantum comparator',
            'n_features': 8,
            **tier2a_results,
        },
        'tier2b_full_features': {
            'description': 'Full MichaelHills features - classical ceiling',
            'n_features': n_tier2b_features,
            **tier2b_results,
        },
        'tier3': tier3_results,
        'polarity_concordance': concordance,
        'subgroup_analysis': subgroups,
        'wilcoxon_tests': wilcoxon_tests,
        'timestamp': datetime.now().isoformat(),
    }

    # Save JSON
    with open(output_dir / 'full_cohort_polarity.json', 'w') as f:
        json.dump(full_results, f, indent=2)

    # Generate report
    generate_report(full_results, output_dir)

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Phase 4 complete ({elapsed:.1f}s)")

    return full_results


def analyze_subgroups(tier2a: Dict, tier2b: Dict, tier3: Dict) -> Dict:
    """Analyze polarity subgroups across methods."""
    subgroups = {
        'tier2a_inverted': [],
        'tier2a_standard': [],
        'tier2b_inverted': [],
        'tier2b_standard': [],
        'tier3_inverted': [],
        'tier3_standard': [],
    }

    for subj, data in tier2a.get('per_subject', {}).items():
        if data['polarity'] == 'inverted':
            subgroups['tier2a_inverted'].append(subj)
        else:
            subgroups['tier2a_standard'].append(subj)

    for subj, data in tier2b.get('per_subject', {}).items():
        if data['polarity'] == 'inverted':
            subgroups['tier2b_inverted'].append(subj)
        else:
            subgroups['tier2b_standard'].append(subj)

    for subj, data in tier3.get('per_subject', {}).items():
        if data['polarity'] == 'inverted':
            subgroups['tier3_inverted'].append(subj)
        else:
            subgroups['tier3_standard'].append(subj)

    # Sort all lists
    for key in subgroups:
        subgroups[key] = sorted(subgroups[key])

    subgroups['summary'] = {
        'tier2a_inverted_count': len(subgroups['tier2a_inverted']),
        'tier2a_standard_count': len(subgroups['tier2a_standard']),
        'tier2b_inverted_count': len(subgroups['tier2b_inverted']),
        'tier2b_standard_count': len(subgroups['tier2b_standard']),
        'tier3_inverted_count': len(subgroups['tier3_inverted']),
        'tier3_standard_count': len(subgroups['tier3_standard']),
    }

    return subgroups


def run_wilcoxon_tests(tier2a: Dict, tier2b: Dict, tier3: Dict) -> Dict:
    """Run Wilcoxon signed-rank tests comparing calibrated AUCs."""
    from scipy.stats import wilcoxon

    results = {}

    # Get common subjects
    common = set(tier2a.get('per_subject', {}).keys()) & \
             set(tier2b.get('per_subject', {}).keys()) & \
             set(tier3.get('per_subject', {}).keys())
    common = sorted(common)

    if len(common) < 5:
        return {'error': 'Not enough subjects for Wilcoxon test'}

    # Extract calibrated AUCs
    t2a_cal = [tier2a['per_subject'][s]['calibrated_auc'] for s in common]
    t2b_cal = [tier2b['per_subject'][s]['calibrated_auc'] for s in common]
    t3_cal = [tier3['per_subject'][s]['calibrated_auc'] for s in common]

    # T3 cal vs T2A cal
    try:
        stat, p = wilcoxon(t3_cal, t2a_cal, alternative='two-sided')
        t3_wins = sum(1 for a, b in zip(t3_cal, t2a_cal) if a > b)
        results['t3_vs_t2a'] = {
            'statistic': float(stat),
            'p_value': float(p),
            'n_subjects': len(common),
            't3_wins': t3_wins,
            't2a_wins': len(common) - t3_wins,
            't3_mean': float(np.mean(t3_cal)),
            't2a_mean': float(np.mean(t2a_cal)),
            'significant_0.05': bool(p < 0.05),
        }
    except Exception as e:
        results['t3_vs_t2a'] = {'error': str(e)}

    # T3 cal vs T2B cal
    try:
        stat, p = wilcoxon(t3_cal, t2b_cal, alternative='two-sided')
        t3_wins = sum(1 for a, b in zip(t3_cal, t2b_cal) if a > b)
        results['t3_vs_t2b'] = {
            'statistic': float(stat),
            'p_value': float(p),
            'n_subjects': len(common),
            't3_wins': t3_wins,
            't2b_wins': len(common) - t3_wins,
            't3_mean': float(np.mean(t3_cal)),
            't2b_mean': float(np.mean(t2b_cal)),
            'significant_0.05': bool(p < 0.05),
        }
    except Exception as e:
        results['t3_vs_t2b'] = {'error': str(e)}

    # T2A cal vs T2B cal
    try:
        stat, p = wilcoxon(t2a_cal, t2b_cal, alternative='two-sided')
        t2a_wins = sum(1 for a, b in zip(t2a_cal, t2b_cal) if a > b)
        results['t2a_vs_t2b'] = {
            'statistic': float(stat),
            'p_value': float(p),
            'n_subjects': len(common),
            't2a_wins': t2a_wins,
            't2b_wins': len(common) - t2a_wins,
            't2a_mean': float(np.mean(t2a_cal)),
            't2b_mean': float(np.mean(t2b_cal)),
            'significant_0.05': bool(p < 0.05),
        }
    except Exception as e:
        results['t2a_vs_t2b'] = {'error': str(e)}

    return results


def analyze_polarity_concordance(tier2a: Dict, tier2b: Dict, tier3: Dict) -> Dict:
    """Analyze polarity concordance across all three methods."""
    concordance = {
        'per_subject': {},
        't2a_t2b_agree': 0,
        't2a_t3_agree': 0,
        't2b_t3_agree': 0,
        'all_agree': 0,
        'total': 0,
    }

    all_subjects = set(tier2a.get('per_subject', {}).keys()) & \
                   set(tier2b.get('per_subject', {}).keys()) & \
                   set(tier3.get('per_subject', {}).keys())

    for subj in sorted(all_subjects):
        t2a_pol = tier2a['per_subject'][subj]['polarity']
        t2b_pol = tier2b['per_subject'][subj]['polarity']
        t3_pol = tier3['per_subject'][subj]['polarity']

        concordance['per_subject'][subj] = {
            'tier2a': t2a_pol,
            'tier2b': t2b_pol,
            'tier3': t3_pol,
        }

        concordance['total'] += 1
        if t2a_pol == t2b_pol:
            concordance['t2a_t2b_agree'] += 1
        if t2a_pol == t3_pol:
            concordance['t2a_t3_agree'] += 1
        if t2b_pol == t3_pol:
            concordance['t2b_t3_agree'] += 1
        if t2a_pol == t2b_pol == t3_pol:
            concordance['all_agree'] += 1

    if concordance['total'] > 0:
        concordance['t2a_t2b_rate'] = concordance['t2a_t2b_agree'] / concordance['total']
        concordance['t2a_t3_rate'] = concordance['t2a_t3_agree'] / concordance['total']
        concordance['t2b_t3_rate'] = concordance['t2b_t3_agree'] / concordance['total']
        concordance['all_agree_rate'] = concordance['all_agree'] / concordance['total']

    return concordance


def generate_report(results: Dict, output_dir: Path):
    """Generate markdown report with feature regime analysis."""
    report = []
    report.append("# PROMPT 018 - Full Cohort Polarity Analysis\n")
    report.append(f"Generated: {results['timestamp']}\n")

    report.append("\n## Configuration\n")
    report.append(f"- Channels: {results['config']['n_channels']} (8-channel quantum configuration)")
    report.append(f"- Window: {results['config']['window_sec']}s")
    report.append(f"- Subjects: {results['n_subjects']} (excluding {results['config']['exclude_subjects']})")
    report.append(f"- Total segments: {results['n_segments']}")

    report.append("\n## Full Cohort Results\n")
    report.append("| Method | Features | Raw AUC | Calibrated | Lift | Inverted |")
    report.append("|--------|----------|---------|------------|------|----------|")

    t2a = results['tier2a_eigenvalues']
    report.append(f"| Tier 2A (eigenvals) | {t2a['n_features']} | "
                  f"{t2a.get('overall_raw_auc', 0):.3f} | "
                  f"{t2a.get('overall_calibrated_auc', 0):.3f} | "
                  f"+{t2a.get('calibration_lift', 0):.3f} | "
                  f"{t2a.get('n_inverted', 0)}/{results['n_subjects']} |")

    t2b = results['tier2b_full_features']
    report.append(f"| Tier 2B (full) | {t2b['n_features']} | "
                  f"{t2b.get('overall_raw_auc', 0):.3f} | "
                  f"{t2b.get('overall_calibrated_auc', 0):.3f} | "
                  f"+{t2b.get('calibration_lift', 0):.3f} | "
                  f"{t2b.get('n_inverted', 0)}/{results['n_subjects']} |")

    t3 = results['tier3']
    report.append(f"| Tier 3 (Riemannian) | 36 | "
                  f"{t3.get('overall_raw_auc', 0):.3f} | "
                  f"{t3.get('overall_calibrated_auc', 0):.3f} | "
                  f"+{t3.get('calibration_lift', 0):.3f} | "
                  f"{t3.get('n_inverted', 0)}/{results['n_subjects']} |")

    report.append("| Quantum PLV CH8* | 8 | 0.524 | 0.637 | +0.113 | 3/7 |")
    report.append("\n*Quantum results from 7-patient subset only")

    report.append("\n## Feature Regime Analysis\n")
    report.append("| Regime | Tier 2 | Tier 3 Cal | Winner | Why |")
    report.append("|--------|--------|------------|--------|-----|")
    t2a_cal = t2a.get('overall_calibrated_auc', 0)
    t2b_cal = t2b.get('overall_calibrated_auc', 0)
    t3_cal = t3.get('overall_calibrated_auc', 0)
    winner_min = "Tier 3" if t3_cal > t2a_cal else "Tier 2A"
    winner_full = "Tier 2B" if t2b_cal > t3_cal else "Tier 3"
    report.append(f"| Minimal (8) | {t2a_cal:.2f} | {t3_cal:.2f} | {winner_min} | Eigenvalues lose info |")
    report.append(f"| Full (~{t2b['n_features']}) | {t2b_cal:.2f} | {t3_cal:.2f} | {winner_full} | Redundancy absorbs |")

    report.append("\n## Polarity Concordance Matrix\n")
    report.append("| Patient | T2A Pol | T2B Pol | T3 Pol |")
    report.append("|---------|---------|---------|--------|")

    conc = results['polarity_concordance']
    for subj, data in sorted(conc.get('per_subject', {}).items()):
        report.append(f"| {subj} | {data['tier2a'][0]} | {data['tier2b'][0]} | {data['tier3'][0]} |")

    report.append(f"\n**Concordance rates:**")
    report.append(f"- T2A vs T2B: {conc.get('t2a_t2b_rate', 0):.1%}")
    report.append(f"- T2A vs T3: {conc.get('t2a_t3_rate', 0):.1%}")
    report.append(f"- T2B vs T3: {conc.get('t2b_t3_rate', 0):.1%}")
    report.append(f"- All three agree: {conc.get('all_agree_rate', 0):.1%}")

    report.append("\n## Per-Subject Details\n")
    report.append("| Subject | T2A Raw | T2A Cal | T2B Raw | T2B Cal | T3 Raw | T3 Cal |")
    report.append("|---------|---------|---------|---------|---------|--------|--------|")

    all_subjects = sorted(set(t2a.get('per_subject', {}).keys()) |
                          set(t2b.get('per_subject', {}).keys()) |
                          set(t3.get('per_subject', {}).keys()))

    for subj in all_subjects:
        t2a_d = t2a.get('per_subject', {}).get(subj, {})
        t2b_d = t2b.get('per_subject', {}).get(subj, {})
        t3_d = t3.get('per_subject', {}).get(subj, {})

        report.append(f"| {subj} | "
                      f"{t2a_d.get('raw_auc', 0):.3f} | {t2a_d.get('calibrated_auc', 0):.3f} | "
                      f"{t2b_d.get('raw_auc', 0):.3f} | {t2b_d.get('calibrated_auc', 0):.3f} | "
                      f"{t3_d.get('raw_auc', 0):.3f} | {t3_d.get('calibrated_auc', 0):.3f} |")

    # Subgroup analysis
    report.append("\n## Subgroup Analysis\n")
    sg = results.get('subgroup_analysis', {})
    report.append(f"**Tier 2A (eigenvalues) polarity:**")
    report.append(f"- Inverted ({sg.get('summary', {}).get('tier2a_inverted_count', 0)}): {', '.join(sg.get('tier2a_inverted', []))}")
    report.append(f"- Standard ({sg.get('summary', {}).get('tier2a_standard_count', 0)}): {', '.join(sg.get('tier2a_standard', []))}")
    report.append(f"\n**Tier 2B (full features) polarity:**")
    report.append(f"- Inverted ({sg.get('summary', {}).get('tier2b_inverted_count', 0)}): {', '.join(sg.get('tier2b_inverted', []))}")
    report.append(f"- Standard ({sg.get('summary', {}).get('tier2b_standard_count', 0)}): {', '.join(sg.get('tier2b_standard', []))}")
    report.append(f"\n**Tier 3 (Riemannian) polarity:**")
    report.append(f"- Inverted ({sg.get('summary', {}).get('tier3_inverted_count', 0)}): {', '.join(sg.get('tier3_inverted', []))}")
    report.append(f"- Standard ({sg.get('summary', {}).get('tier3_standard_count', 0)}): {', '.join(sg.get('tier3_standard', []))}")

    # Wilcoxon tests
    report.append("\n## Wilcoxon Signed-Rank Tests\n")
    wt = results.get('wilcoxon_tests', {})

    report.append("| Comparison | Mean A | Mean B | p-value | Significant |")
    report.append("|------------|--------|--------|---------|-------------|")

    t3_t2a = wt.get('t3_vs_t2a', {})
    if 'error' not in t3_t2a:
        sig = "Yes" if t3_t2a.get('significant_0.05', False) else "No"
        report.append(f"| T3 cal vs T2A cal | {t3_t2a.get('t3_mean', 0):.3f} | {t3_t2a.get('t2a_mean', 0):.3f} | "
                      f"{t3_t2a.get('p_value', 1):.4f} | {sig} |")

    t3_t2b = wt.get('t3_vs_t2b', {})
    if 'error' not in t3_t2b:
        sig = "Yes" if t3_t2b.get('significant_0.05', False) else "No"
        report.append(f"| T3 cal vs T2B cal | {t3_t2b.get('t3_mean', 0):.3f} | {t3_t2b.get('t2b_mean', 0):.3f} | "
                      f"{t3_t2b.get('p_value', 1):.4f} | {sig} |")

    t2a_t2b = wt.get('t2a_vs_t2b', {})
    if 'error' not in t2a_t2b:
        sig = "Yes" if t2a_t2b.get('significant_0.05', False) else "No"
        report.append(f"| T2A cal vs T2B cal | {t2a_t2b.get('t2a_mean', 0):.3f} | {t2a_t2b.get('t2b_mean', 0):.3f} | "
                      f"{t2a_t2b.get('p_value', 1):.4f} | {sig} |")

    report.append("\n*Wilcoxon signed-rank test, two-sided, alpha=0.05*")

    with open(output_dir / 'full_cohort_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(data_dir: Path, output_dir: Path, local_mode: bool = False):
    """Main training function with 4-phase execution."""
    logger.info("=" * 80)
    logger.info("PROMPT 018 - FULL COHORT POLARITY ANALYSIS (FIX2)")
    logger.info("=" * 80)
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Local mode: {local_mode}")
    logger.info(f"Window: {WINDOW_SEC}s")
    logger.info(f"Started: {datetime.now()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_start = datetime.now()

    # Find all subjects
    all_subjects = sorted([d for d in data_dir.iterdir()
                           if d.is_dir() and d.name.startswith('chb')])
    subjects = [d for d in all_subjects if d.name not in EXCLUDE_SUBJECTS]

    logger.info(f"Found {len(all_subjects)} subjects, using {len(subjects)} "
                f"(excluding {EXCLUDE_SUBJECTS})")

    try:
        # PHASE 1: Feature extraction
        patient_features = phase1_extract_features(data_dir, output_dir, subjects)

        if not patient_features:
            logger.error("No patient features extracted. Aborting.")
            return

        # PHASE 2: Validation gate
        passed, validation_results = phase2_validation_gate(patient_features, output_dir)

        if not passed:
            logger.error("VALIDATION FAILED - Aborting full cohort analysis")
            logger.error("See validation_failed.json for details")
            return  # Exit cleanly (exit code 0)

        # PHASE 3: Full cohort LOSO
        tier2a_results, tier2b_results, tier3_results = phase3_full_loso(patient_features, output_dir)

        # PHASE 4: Analysis + report
        full_results = phase4_analysis(tier2a_results, tier2b_results, tier3_results,
                                       patient_features, output_dir)

        # Print final summary
        total_time = (datetime.now() - pipeline_start).total_seconds() / 60

        logger.info("\n" + "=" * 80)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Subjects analyzed: {len(patient_features)}")
        logger.info(f"Total segments: {full_results['n_segments']}")
        logger.info(f"\nTier 2A (Eigenvalues only, {full_results['tier2a_eigenvalues']['n_features']} features):")
        logger.info(f"  Raw AUC:        {tier2a_results.get('overall_raw_auc', 0):.3f}")
        logger.info(f"  Calibrated AUC: {tier2a_results.get('overall_calibrated_auc', 0):.3f}")
        logger.info(f"  Inverted:       {tier2a_results.get('n_inverted', 0)}/{len(patient_features)}")
        logger.info(f"\nTier 2B (Full features, {full_results['tier2b_full_features']['n_features']} features):")
        logger.info(f"  Raw AUC:        {tier2b_results.get('overall_raw_auc', 0):.3f}")
        logger.info(f"  Calibrated AUC: {tier2b_results.get('overall_calibrated_auc', 0):.3f}")
        logger.info(f"  Inverted:       {tier2b_results.get('n_inverted', 0)}/{len(patient_features)}")
        logger.info(f"\nTier 3 (Riemannian):")
        logger.info(f"  Raw AUC:        {tier3_results.get('overall_raw_auc', 0):.3f}")
        logger.info(f"  Calibrated AUC: {tier3_results.get('overall_calibrated_auc', 0):.3f}")
        logger.info(f"  Inverted:       {tier3_results.get('n_inverted', 0)}/{len(patient_features)}")
        logger.info(f"\nResults saved to: {output_dir}")
        logger.info(f"DONE. Total runtime: {total_time:.1f} min")

    except Exception as e:
        logger.error(f"Pipeline failed with exception: {e}")
        logger.error(traceback.format_exc())

        error_results = {
            'error': str(e),
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat(),
        }
        with open(output_dir / 'pipeline_error.json', 'w') as f:
            json.dump(error_results, f, indent=2)

        raise


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PROMPT 018 - Full Cohort Analysis')

    parser.add_argument('--model-dir', type=str,
                        default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data-dir', type=str,
                        default=os.environ.get('SM_CHANNEL_DATA', '/opt/ml/input/data/data'))
    parser.add_argument('--output-dir', type=str,
                        default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    parser.add_argument('--local', action='store_true',
                        help='Run in local validation mode')

    args = parser.parse_args()

    # Debug output
    print("=" * 80)
    print("PROMPT 018 - FULL COHORT POLARITY ANALYSIS (FIX2)")
    print("=" * 80)
    print(f"SM_CHANNEL_DATA: {os.environ.get('SM_CHANNEL_DATA', 'NOT SET')}")
    print(f"SM_MODEL_DIR: {os.environ.get('SM_MODEL_DIR', 'NOT SET')}")
    print(f"SM_OUTPUT_DATA_DIR: {os.environ.get('SM_OUTPUT_DATA_DIR', 'NOT SET')}")
    print(f"Data dir arg: {args.data_dir}")
    print(f"Output dir arg: {args.output_dir}")
    print(f"Local mode: {args.local}")

    data_path = Path(args.data_dir)
    print(f"\nData directory exists: {data_path.exists()}")

    if data_path.exists():
        print(f"Contents of {data_path}:")
        for item in list(data_path.iterdir())[:10]:
            print(f"  {item.name}")
    else:
        # Try alternative paths in SageMaker
        for alt_path in ['/opt/ml/input/data', '/opt/ml/input/data/data', '/opt/ml/input']:
            p = Path(alt_path)
            if p.exists():
                print(f"\nFound data at: {alt_path}")
                print(f"Contents:")
                for item in list(p.iterdir())[:10]:
                    print(f"  {item.name}")
                args.data_dir = alt_path
                break

    print("=" * 80)

    # Set output dir for local mode
    if args.local:
        args.output_dir = str(Path(__file__).parent.parent / 'results' / 'full_cohort')

    train(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        local_mode=args.local,
    )
