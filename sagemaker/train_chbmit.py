#!/usr/bin/env python3
"""
================================================================================
CHB-MIT SAGEMAKER TRAINING SCRIPT - V2 (Kaggle Winner Features)
================================================================================

SageMaker-compatible training script with ClassicalBaselineV2 features:
- Hills: eigenvalues, FFT magnitudes, frequency cross-correlations
- Barachant: relative log power, MAX aggregation
- Temko: Hjorth, AR error, spectral entropy, zero crossings, fine spectral

Environment:
    SM_CHANNEL_DATA: Path to input data (CHB-MIT EDF files)
    SM_MODEL_DIR: Path to save model outputs
    SM_OUTPUT_DATA_DIR: Path for additional outputs

================================================================================
"""

import subprocess
import sys

# Install pyedflib only - use sklearn's RandomForest instead of XGBoost
# to avoid numpy/scipy version conflicts in the container
print("Installing pyedflib...")
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', 'pyedflib'])
print("pyedflib installed.")

import os
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import logging
from scipy import signal
from scipy.stats import skew, kurtosis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

FS = 256  # Hz

# Segment parameters
PREICTAL_DURATION = 60
INTERICTAL_MIN_GAP = 300
WINDOW_SEC = 30.0
MAX_INTERICTAL_PER_FILE = 5

# Channel harmonization: 18 channels common to all usable subjects
# Discovered by scripts/discover_channels.py
COMMON_CHANNELS = [
    "C3-P3",
    "C4-P4",
    "CZ-PZ",
    "F3-C3",
    "F4-C4",
    "F7-T7",
    "F8-T8",
    "FP1-F3",
    "FP1-F7",
    "FP2-F4",
    "FP2-F8",
    "FZ-CZ",
    "P3-O1",
    "P4-O2",
    "P7-O1",
    "P8-O2",
    "T7-P7",
    "T8-P8",
]

# Subjects to exclude:
# - chb12: Major montage changes mid-recording (different reference systems)
# - chb24: Re-recording of chb01, often has no EDF files
EXCLUDE_SUBJECTS = {'chb12', 'chb24'}

# Frequency bands
FREQ_BANDS_COARSE = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 15),
    'beta': (15, 30),
    'low_gamma': (30, 70),
    'high_gamma': (70, 128),
}

FREQ_BANDS_FINE = {f'{lo}-{lo+2}Hz': (lo, lo + 2) for lo in range(0, 40, 2)}


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
class FeatureSegment:
    """A segment with extracted features."""
    subject: str
    label: str
    features: np.ndarray = field(repr=False)


@dataclass
class SegmentResult:
    subject: str
    source_file: str
    label: str
    start_sec: float
    end_sec: float
    n_channels: int
    n_features: int


# =============================================================================
# PREPROCESSING (Temko style)
# =============================================================================

def preprocess_eeg(data: np.ndarray, fs: int = 256,
                   lowcut: float = 0.5, highcut: float = 128.0,
                   notch_freq: float = 60.0) -> np.ndarray:
    """Preprocess EEG: demean, bandpass, notch."""
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

    return data


# =============================================================================
# FEATURE EXTRACTION - KAGGLE WINNER TECHNIQUES
# =============================================================================

def _welch_psd(x: np.ndarray, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD with sane defaults."""
    nperseg = min(512, len(x) // 2)
    if nperseg < 16:
        nperseg = len(x)
    noverlap = nperseg // 4
    return signal.welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap)


def extract_band_power(data: np.ndarray, fs: int,
                       bands: Dict = None) -> np.ndarray:
    """Relative log power in frequency bands (Barachant)."""
    if bands is None:
        bands = FREQ_BANDS_COARSE

    n_ch = data.shape[0]
    n_bands = len(bands)
    feat = np.zeros((n_ch, n_bands))

    for ch in range(n_ch):
        freqs, psd = _welch_psd(data[ch], fs)
        total = np.sum(psd) + 1e-12

        for i, (_, (lo, hi)) in enumerate(bands.items()):
            idx = np.where((freqs >= lo) & (freqs <= hi))[0]
            if len(idx) > 0:
                feat[ch, i] = np.log10(np.sum(psd[idx]) / total + 1e-12)
            else:
                feat[ch, i] = -10.0

    return feat.flatten()


def extract_fine_spectral(data: np.ndarray, fs: int) -> np.ndarray:
    """Fine 2 Hz filterbank log energies (Temko)."""
    return extract_band_power(data, fs, bands=FREQ_BANDS_FINE)


def extract_fft_magnitudes(data: np.ndarray, fs: int,
                           fmin: float = 1.0, fmax: float = 47.0,
                           bin_width: float = 1.0) -> np.ndarray:
    """Log10 FFT magnitude in 1 Hz bins (Hills)."""
    n_ch, n_samp = data.shape
    freqs = np.fft.rfftfreq(n_samp, d=1.0 / fs)

    bin_edges = np.arange(fmin, fmax + bin_width, bin_width)
    n_bins = len(bin_edges) - 1

    feat = np.zeros((n_ch, n_bins))
    for ch in range(n_ch):
        fft_mag = np.abs(np.fft.rfft(data[ch]))
        for b in range(n_bins):
            idx = np.where((freqs >= bin_edges[b]) & (freqs < bin_edges[b + 1]))[0]
            if len(idx) > 0:
                feat[ch, b] = np.log10(np.mean(fft_mag[idx]) + 1e-12)
            else:
                feat[ch, b] = -12.0

    return feat.flatten()


def extract_statistics(data: np.ndarray) -> np.ndarray:
    """Per-channel statistics: mean, std, min, max, skewness, kurtosis."""
    n_ch = data.shape[0]
    feat = np.zeros((n_ch, 6))
    for ch in range(n_ch):
        x = data[ch]
        feat[ch] = [np.mean(x), np.std(x), np.min(x), np.max(x),
                    skew(x), kurtosis(x)]
    return feat.flatten()


def extract_hjorth(data: np.ndarray) -> np.ndarray:
    """Hjorth parameters: activity, mobility, complexity (Temko)."""
    n_ch = data.shape[0]
    feat = np.zeros((n_ch, 3))
    for ch in range(n_ch):
        x = data[ch]
        dx = np.diff(x)
        ddx = np.diff(dx)

        var_x = np.var(x) + 1e-12
        var_dx = np.var(dx) + 1e-12
        var_ddx = np.var(ddx) + 1e-12

        activity = var_x
        mobility = np.sqrt(var_dx / var_x)
        complexity = np.sqrt(var_ddx / var_dx) / mobility if mobility > 1e-12 else 0

        feat[ch] = [activity, mobility, complexity]
    return feat.flatten()


def extract_spectral_entropy(data: np.ndarray, fs: int) -> np.ndarray:
    """Spectral entropy per channel (Temko)."""
    n_ch = data.shape[0]
    feat = np.zeros(n_ch)
    for ch in range(n_ch):
        _, psd = _welch_psd(data[ch], fs)
        psd_norm = psd / (np.sum(psd) + 1e-12)
        psd_norm = psd_norm[psd_norm > 0]
        feat[ch] = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    return feat


def extract_ar_error(data: np.ndarray, max_order: int = 6) -> np.ndarray:
    """Autoregressive model prediction error (Temko)."""
    n_ch = data.shape[0]
    feat = np.zeros((n_ch, max_order))

    for ch in range(n_ch):
        x = data[ch]
        x = x - np.mean(x)
        n = len(x)
        if n < max_order + 2:
            continue

        r = np.correlate(x, x, mode='full')
        r = r[n - 1:] / n

        for order in range(1, max_order + 1):
            try:
                R = np.zeros((order, order))
                for i in range(order):
                    for j in range(order):
                        R[i, j] = r[abs(i - j)]
                rhs = r[1:order + 1]
                a = np.linalg.solve(R, rhs)

                pred = np.zeros(n)
                for k in range(order, n):
                    pred[k] = np.dot(a, x[k - order:k][::-1])
                err = x[order:] - pred[order:]
                feat[ch, order - 1] = np.log10(np.mean(err ** 2) + 1e-12)
            except (np.linalg.LinAlgError, ValueError):
                feat[ch, order - 1] = 0.0

    return feat.flatten()


def extract_zero_crossings(data: np.ndarray) -> np.ndarray:
    """Zero crossing rate on raw, delta, delta-delta (Temko)."""
    n_ch = data.shape[0]
    feat = np.zeros((n_ch, 3))

    for ch in range(n_ch):
        x = data[ch]
        dx = np.diff(x)
        ddx = np.diff(dx)

        n = len(x)
        feat[ch, 0] = np.sum(np.diff(np.sign(x)) != 0) / max(n, 1)
        feat[ch, 1] = np.sum(np.diff(np.sign(dx)) != 0) / max(len(dx), 1)
        feat[ch, 2] = np.sum(np.diff(np.sign(ddx)) != 0) / max(len(ddx), 1)

    return feat.flatten()


def extract_nonlinear_energy(data: np.ndarray) -> np.ndarray:
    """Teager-Kaiser nonlinear energy (Temko)."""
    n_ch = data.shape[0]
    feat = np.zeros(n_ch)
    for ch in range(n_ch):
        x = data[ch]
        if len(x) > 2:
            nle = x[1:-1] ** 2 - x[:-2] * x[2:]
            feat[ch] = np.mean(np.abs(nle))
    return feat


def extract_rms(data: np.ndarray) -> np.ndarray:
    """RMS amplitude per channel."""
    return np.sqrt(np.mean(data ** 2, axis=1))


def extract_correlation_features(data: np.ndarray) -> np.ndarray:
    """Upper triangle of time-domain cross-correlation matrix."""
    corr = np.corrcoef(data)
    corr = np.nan_to_num(corr, nan=0.0)
    idx = np.triu_indices(corr.shape[0], k=1)
    return corr[idx]


def extract_correlation_eigenvalues(data: np.ndarray) -> np.ndarray:
    """Sorted eigenvalues of cross-correlation matrix (Hills)."""
    corr = np.corrcoef(data)
    corr = np.nan_to_num(corr, nan=0.0)
    eigvals = np.sort(np.real(np.linalg.eigvals(corr)))[::-1]
    return eigvals


def extract_freq_correlation(data: np.ndarray, fs: int,
                             fmin: float = 1.0, fmax: float = 47.0) -> np.ndarray:
    """Frequency-domain cross-correlation + eigenvalues (Hills)."""
    n_ch, n_samp = data.shape
    freqs = np.fft.rfftfreq(n_samp, d=1.0 / fs)
    idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]

    spectra = np.zeros((n_ch, len(idx)))
    for ch in range(n_ch):
        mag = np.abs(np.fft.rfft(data[ch]))[idx]
        norm = np.linalg.norm(mag) + 1e-12
        spectra[ch] = mag / norm

    corr = np.corrcoef(spectra)
    corr = np.nan_to_num(corr, nan=0.0)

    triu = corr[np.triu_indices(n_ch, k=1)]
    eigvals = np.sort(np.real(np.linalg.eigvals(corr)))[::-1]

    return np.concatenate([triu, eigvals])


def extract_all_features(data: np.ndarray, fs: int) -> np.ndarray:
    """
    Extract full Kaggle winner feature set.

    For 23 channels: ~2500 features
    """
    features = []
    n_ch = data.shape[0]

    # Spectral features
    features.append(extract_band_power(data, fs))           # n_ch * 6
    features.append(extract_fine_spectral(data, fs))        # n_ch * 20

    fmax = min(47.0, fs / 2.0 - 1)
    if fmax > 1.0:
        features.append(extract_fft_magnitudes(data, fs, 1.0, fmax))  # n_ch * 46

    # Statistical features
    features.append(extract_statistics(data))               # n_ch * 6
    features.append(extract_hjorth(data))                   # n_ch * 3
    features.append(extract_spectral_entropy(data, fs))     # n_ch
    features.append(extract_ar_error(data, max_order=6))    # n_ch * 6
    features.append(extract_zero_crossings(data))           # n_ch * 3
    features.append(extract_nonlinear_energy(data))         # n_ch
    features.append(extract_rms(data))                      # n_ch

    # Cross-channel features
    if n_ch > 1:
        features.append(extract_correlation_features(data))      # n_ch*(n_ch-1)/2
        features.append(extract_correlation_eigenvalues(data))   # n_ch
        fmax = min(47.0, fs / 2.0 - 1)
        if fmax > 1.0:
            features.append(extract_freq_correlation(data, fs, 1.0, fmax))

    result = np.concatenate(features)
    result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    return result


# =============================================================================
# EDF READING (with channel selection by label)
# =============================================================================

def normalize_channel_label(label: str) -> str:
    """
    Normalize channel label for consistent matching.

    - Strip whitespace
    - Convert to uppercase
    - Remove common prefixes like 'EEG '
    - Standardize separators
    """
    label = label.strip().upper()

    # Remove common prefixes
    for prefix in ['EEG ', 'EEG-']:
        if label.startswith(prefix):
            label = label[len(prefix):]

    # Standardize separators
    label = label.replace('--', '-')

    return label


def read_edf_segment(filepath: str, start_sec: float, end_sec: float,
                     target_channels: List[str] = None) -> Tuple[Optional[np.ndarray], float]:
    """
    Read a segment from an EDF file, selecting specific channels by label.

    Args:
        filepath: Path to EDF file
        start_sec: Start time in seconds
        end_sec: End time in seconds
        target_channels: List of channel labels to read (normalized uppercase).
                        If None, reads all channels.

    Returns:
        Tuple of (data array, sample frequency)
        Data array shape: (n_channels, n_samples)
        Returns None, fs if there's an error or channels not found.
    """
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

            # If no target channels specified, read all
            if target_channels is None:
                channel_indices = list(range(n_signals))
            else:
                # Build mapping of normalized label -> channel index
                label_to_idx = {}
                for i in range(n_signals):
                    label = normalize_channel_label(f.getLabel(i))
                    label_to_idx[label] = i

                # Find indices for target channels in order
                channel_indices = []
                for target in target_channels:
                    target_norm = normalize_channel_label(target)
                    if target_norm in label_to_idx:
                        channel_indices.append(label_to_idx[target_norm])
                    else:
                        # Channel not found - skip this file
                        logger.debug(f"Channel {target} not found in {filepath}")
                        return None, fs

            # Read selected channels
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
        logger.warning(f"Error reading {filepath}: {e}")
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
# SEGMENT ANALYSIS
# =============================================================================

def analyze_segment(segment: EEGSegment, data_dir: Path,
                    target_channels: List[str] = None) -> Optional[Tuple[SegmentResult, FeatureSegment]]:
    """Load, preprocess, and extract features from a segment."""
    edf_path = data_dir / segment.subject / segment.source_file

    if not edf_path.exists():
        return None

    # Read data with channel selection
    data, fs = read_edf_segment(str(edf_path), segment.start_sec, segment.end_sec,
                                target_channels=target_channels)

    if data is None or data.size == 0:
        return None

    # Minimum duration check
    min_samples = int(WINDOW_SEC * fs * 0.5)
    if data.shape[1] < min_samples:
        return None

    # Preprocess
    data = preprocess_eeg(data, fs=int(fs))

    # Extract features
    features = extract_all_features(data, int(fs))

    result = SegmentResult(
        subject=segment.subject,
        source_file=segment.source_file,
        label=segment.label,
        start_sec=segment.start_sec,
        end_sec=segment.end_sec,
        n_channels=data.shape[0],
        n_features=len(features)
    )

    feat_seg = FeatureSegment(
        subject=segment.subject,
        label=segment.label,
        features=features
    )

    return result, feat_seg


# =============================================================================
# LOSO CROSS-VALIDATION
# =============================================================================

def run_loso_cv(feature_segments: List[FeatureSegment]) -> Dict:
    """Run Leave-One-Subject-Out cross-validation with RandomForest."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

    # Filter for consistent feature dimensions
    feature_lengths = [len(s.features) for s in feature_segments]
    from collections import Counter
    length_counts = Counter(feature_lengths)
    most_common_length = length_counts.most_common(1)[0][0]

    filtered_segments = [s for s in feature_segments if len(s.features) == most_common_length]
    logger.info(f"Filtered to {len(filtered_segments)} segments with {most_common_length} features "
                f"(removed {len(feature_segments) - len(filtered_segments)} with different lengths)")

    subjects = sorted(set(s.subject for s in filtered_segments))
    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}

    results = {'per_subject': {}, 'overall': {}, 'model_params': {}}
    all_preds, all_true, all_proba = [], [], []

    logger.info(f"LOSO CV: {len(subjects)} subjects, {len(filtered_segments)} segments")

    for test_subject in subjects:
        train_segs = [s for s in filtered_segments if s.subject != test_subject]
        test_segs = [s for s in filtered_segments if s.subject == test_subject]

        if not train_segs or not test_segs:
            continue

        X_train = np.nan_to_num(np.array([s.features for s in train_segs]))
        y_train = np.array([label_map.get(s.label, 0) for s in train_segs])
        X_test = np.nan_to_num(np.array([s.features for s in test_segs]))
        y_test = np.array([label_map.get(s.label, 0) for s in test_segs])

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.info(f"  {test_subject}: skipped (single class)")
            continue

        # RandomForest with more estimators for larger feature set
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5

        results['per_subject'][test_subject] = {
            'n_train': len(X_train), 'n_test': len(X_test),
            'accuracy': float(acc), 'precision': float(prec),
            'recall': float(rec), 'f1': float(f1), 'auc': float(auc),
        }

        all_preds.extend(y_pred.tolist())
        all_true.extend(y_test.tolist())
        all_proba.extend(y_proba.tolist())

        logger.info(f"  {test_subject}: Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")

    if all_preds:
        results['overall'] = {
            'n_segments': len(all_true),
            'accuracy': float(accuracy_score(all_true, all_preds)),
            'precision': float(precision_score(all_true, all_preds, zero_division=0)),
            'recall': float(recall_score(all_true, all_preds, zero_division=0)),
            'f1': float(f1_score(all_true, all_preds, zero_division=0)),
        }
        try:
            results['overall']['auc'] = float(roc_auc_score(all_true, all_proba))
        except ValueError:
            results['overall']['auc'] = 0.5

    results['model_params'] = {
        'classifier': 'RandomForest',
        'n_estimators': 300,
        'window_sec': WINDOW_SEC,
        'n_features': most_common_length,
        'n_channels': len(COMMON_CHANNELS),
        'feature_set': 'ClassicalBaselineV3_ChannelHarmonized'
    }

    return results


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train(data_dir: Path, output_dir: Path):
    """Main training function."""
    logger.info("=" * 80)
    logger.info("CHB-MIT SAGEMAKER TRAINING - V2 (Kaggle Winner Features)")
    logger.info("=" * 80)
    logger.info(f"Data dir: {data_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Started: {datetime.now()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Find subjects (excluding problematic ones)
    all_subjects = sorted([d for d in data_dir.iterdir()
                           if d.is_dir() and d.name.startswith('chb')])
    subjects = [d for d in all_subjects if d.name not in EXCLUDE_SUBJECTS]
    logger.info(f"Found {len(all_subjects)} subjects, using {len(subjects)} (excluding {EXCLUDE_SUBJECTS})")
    logger.info(f"Using {len(COMMON_CHANNELS)} common channels for consistent feature dimensions")

    # Process each subject
    all_results = []
    feature_segments = []

    for subject_dir in subjects:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {subject_dir.name}")

        segments = extract_segments_for_subject(subject_dir)

        ictal = sum(1 for s in segments if s.label == 'ictal')
        preictal = sum(1 for s in segments if s.label == 'preictal')
        interictal = sum(1 for s in segments if s.label == 'interictal')
        logger.info(f"  Segments: {ictal} ictal, {preictal} preictal, {interictal} interictal")

        subject_results = []
        for seg in segments:
            result = analyze_segment(seg, data_dir, target_channels=COMMON_CHANNELS)
            if result:
                seg_result, feat_seg = result
                subject_results.append(seg_result)
                feature_segments.append(feat_seg)

        all_results.extend(subject_results)
        logger.info(f"  Done: {len(subject_results)} results")

    logger.info(f"\nTotal results: {len(all_results)}")
    logger.info(f"Total feature segments: {len(feature_segments)}")

    # Run LOSO CV
    logger.info("\n" + "=" * 80)
    logger.info("LOSO CROSS-VALIDATION")
    logger.info("=" * 80)

    loso_results = run_loso_cv(feature_segments)

    # Save results
    results_data = [asdict(r) for r in all_results]
    with open(output_dir / 'segment_results.json', 'w') as f:
        json.dump(results_data, f, indent=2)

    with open(output_dir / 'loso_classification.json', 'w') as f:
        json.dump(loso_results, f, indent=2)

    # Summary
    n_ictal = sum(1 for r in all_results if r.label == 'ictal')
    n_preictal = sum(1 for r in all_results if r.label == 'preictal')
    n_interictal = sum(1 for r in all_results if r.label == 'interictal')

    summary = {
        'n_subjects': len(subjects),
        'n_segments': len(all_results),
        'n_ictal': n_ictal,
        'n_preictal': n_preictal,
        'n_interictal': n_interictal,
        'loso_results': loso_results.get('overall', {}),
        'model_params': loso_results.get('model_params', {}),
        'completed': datetime.now().isoformat()
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Completed: {datetime.now()}")

    # Print summary
    if 'overall' in loso_results:
        overall = loso_results['overall']
        logger.info(f"\n{'='*60}")
        logger.info("FINAL RESULTS")
        logger.info(f"{'='*60}")
        logger.info(f"Accuracy:  {overall.get('accuracy', 0):.3f}")
        logger.info(f"F1 Score:  {overall.get('f1', 0):.3f}")
        logger.info(f"AUC:       {overall.get('auc', 0):.3f}")
        logger.info(f"Precision: {overall.get('precision', 0):.3f}")
        logger.info(f"Recall:    {overall.get('recall', 0):.3f}")


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CHB-MIT SageMaker Training V2')

    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_DATA', '/opt/ml/input/data/data'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))

    args = parser.parse_args()

    print("=" * 80)
    print("SAGEMAKER ENVIRONMENT DEBUG")
    print("=" * 80)
    print(f"SM_CHANNEL_DATA: {os.environ.get('SM_CHANNEL_DATA', 'NOT SET')}")
    print(f"SM_MODEL_DIR: {os.environ.get('SM_MODEL_DIR', 'NOT SET')}")
    print(f"SM_OUTPUT_DATA_DIR: {os.environ.get('SM_OUTPUT_DATA_DIR', 'NOT SET')}")
    print(f"Data dir arg: {args.data_dir}")
    print(f"Output dir arg: {args.output_dir}")

    data_path = Path(args.data_dir)
    print(f"\nData directory exists: {data_path.exists()}")
    if data_path.exists():
        print(f"Contents of {data_path}:")
        for item in list(data_path.iterdir())[:20]:
            print(f"  {item.name}")
    else:
        for alt_path in ['/opt/ml/input/data', '/opt/ml/input/data/data', '/opt/ml/input']:
            p = Path(alt_path)
            if p.exists():
                print(f"\nFound data at: {alt_path}")
                print(f"Contents:")
                for item in list(p.iterdir())[:20]:
                    print(f"  {item.name}")
                args.data_dir = alt_path
                break

    print("=" * 80)

    train(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )
