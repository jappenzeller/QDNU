#!/usr/bin/env python3
"""
Tier 1 Classical Baseline: FFT Power Band Features

Standardized benchmark runner for the weak classical baseline.
Uses corrected preprocessing from Prompt 1 with FFT power band features.

Feature extraction:
- 5 bands (delta, theta, alpha, beta, gamma) per channel
- log10 mean power per band
- 1-second non-overlapping windows

Classifier:
- Random Forest, 500 trees
- Leave-one-seizure-out cross-validation (per patient)
- Reports mean ± std AUC

Output:
- JSON per patient: results/tier1_fft_power/{patient_id}.json
- Summary CSV: results/tier1_summary.csv

Usage:
    python tier1_benchmark.py --data_dir /path/to/chb-mit [--patient chb01]
"""

import numpy as np
import json
import csv
from scipy import signal
from scipy.signal import welch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import argparse
import sys
import warnings

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# CONSTANTS
# =============================================================================

# Standard EEG frequency bands (for 40 Hz cutoff)
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}

# Preprocessing parameters (from Prompt 1)
PREPROCESSING = {
    'bandpass': [0.5, 40],
    'notch': 60,
    'window_sec': 1,
    'normalize': 'channel'
}

# CHB-MIT patients (22 total, chb12 is missing from dataset)
CHBMIT_PATIENTS = [
    'chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06', 'chb07', 'chb08',
    'chb09', 'chb10', 'chb11', 'chb13', 'chb14', 'chb15', 'chb16', 'chb17',
    'chb18', 'chb19', 'chb20', 'chb21', 'chb22', 'chb23'
]


# =============================================================================
# PREPROCESSING (from Prompt 1)
# =============================================================================

def preprocess_eeg(data: np.ndarray, fs: int = 256,
                   bandpass: Tuple[float, float] = (0.5, 40),
                   notch: float = 60,
                   normalize: str = 'channel') -> np.ndarray:
    """
    Preprocess EEG using validated clinical pipeline.

    Args:
        data: EEG array (n_channels, n_samples)
        fs: Sampling frequency
        bandpass: (low, high) cutoff frequencies
        notch: Notch filter frequency
        normalize: 'channel' for per-channel z-score

    Returns:
        Preprocessed EEG array (same shape)
    """
    data = data.astype(np.float64).copy()

    # Demean per channel
    data = data - np.mean(data, axis=1, keepdims=True)

    # Bandpass filter
    nyq = fs / 2
    low = bandpass[0] / nyq
    high = min(bandpass[1] / nyq, 0.99)

    if low > 0 and low < high:
        try:
            b, a = signal.butter(5, [low, high], btype='band')
            data = signal.filtfilt(b, a, data, axis=1)
        except ValueError:
            pass

    # Notch filter
    if notch and notch < nyq:
        try:
            b_notch, a_notch = signal.iirnotch(notch, Q=30, fs=fs)
            data = signal.filtfilt(b_notch, a_notch, data, axis=1)
        except ValueError:
            pass

    # Per-channel z-score normalization
    if normalize == 'channel':
        for ch in range(data.shape[0]):
            std = np.std(data[ch])
            if std > 1e-10:
                data[ch] = (data[ch] - np.mean(data[ch])) / std

    return data


def extract_windows(data: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Extract non-overlapping windows (critical for CV integrity).

    Args:
        data: (n_channels, n_samples)
        window_samples: samples per window

    Returns:
        (n_windows, n_channels, window_samples)
    """
    n_channels, n_samples = data.shape
    n_windows = n_samples // window_samples

    if n_windows == 0:
        return np.array([])

    # Truncate and reshape
    data = data[:, :n_windows * window_samples]
    windows = data.reshape(n_channels, n_windows, window_samples)
    windows = np.transpose(windows, (1, 0, 2))

    return windows


# =============================================================================
# FFT POWER BAND FEATURES (Tier 1)
# =============================================================================

def extract_fft_power_bands(window: np.ndarray, fs: int = 256,
                            bands: Dict = None) -> np.ndarray:
    """
    Extract log power in frequency bands for one window.

    Args:
        window: (n_channels, n_samples)
        fs: Sampling frequency
        bands: Dict of {band_name: (low, high)}

    Returns:
        Feature vector (n_channels * n_bands,)
    """
    if bands is None:
        bands = FREQ_BANDS

    n_channels = window.shape[0]
    n_bands = len(bands)
    features = np.zeros((n_channels, n_bands))

    for ch in range(n_channels):
        # Compute PSD using Welch's method
        nperseg = min(256, window.shape[1])
        freqs, psd = welch(window[ch], fs=fs, nperseg=nperseg)

        # Total power for normalization (relative power)
        total_power = np.sum(psd)
        if total_power < 1e-10:
            total_power = 1e-10

        # Extract power in each band
        for i, (band_name, (low, high)) in enumerate(bands.items()):
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            if len(idx) > 0:
                band_power = np.sum(psd[idx])
                # Log10 of relative power (more Gaussian-distributed)
                features[ch, i] = np.log10(band_power / total_power + 1e-10)
            else:
                features[ch, i] = -10  # Very small power

    return features.flatten()


# =============================================================================
# CHB-MIT DATA LOADING
# =============================================================================

def load_edf(edf_path: Path) -> Tuple[np.ndarray, int, List[str]]:
    """Load EDF file, returns (data, fs, channel_names)."""
    try:
        import mne
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        data = raw.get_data() * 1e6  # Convert to microvolts
        fs = int(raw.info['sfreq'])
        ch_names = raw.info['ch_names']
        return data, fs, ch_names
    except ImportError:
        raise ImportError("MNE required. Install with: pip install mne")


def parse_summary(patient_dir: Path) -> Dict:
    """Parse CHB-MIT summary file for seizure annotations."""
    summary_file = patient_dir / f"{patient_dir.name}-summary.txt"

    if not summary_file.exists():
        return {}

    seizures = {}
    current_file = None

    with open(summary_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('File Name:'):
                current_file = line.split(':')[1].strip()
                seizures[current_file] = []
            elif 'Start Time' in line and 'Seizure' in line and current_file:
                # Handle both "Seizure Start Time: X" and "Seizure N Start Time: X"
                try:
                    # Split on last colon to get the time value
                    val = line.split(':')[-1].strip().replace(' seconds', '')
                    start = int(val)
                    seizures[current_file].append({'start': start})
                except:
                    pass
            elif 'End Time' in line and 'Seizure' in line and current_file and seizures[current_file]:
                try:
                    val = line.split(':')[-1].strip().replace(' seconds', '')
                    end = int(val)
                    seizures[current_file][-1]['end'] = end
                except:
                    pass

    return seizures


def load_patient_data(patient_dir: Path, max_interictal_ratio: float = 3.0
                      ) -> Tuple[List[Dict], int]:
    """
    Load all seizure data for a patient.

    Returns list of seizures, each containing:
    - ictal_windows: (n_windows, n_channels, window_samples)
    - interictal_windows: balanced interictal from same file
    - file_name: source file

    Args:
        patient_dir: Path to patient folder
        max_interictal_ratio: Max ratio of interictal:ictal windows

    Returns:
        (list of seizure dicts, sampling_rate)
    """
    seizure_info = parse_summary(patient_dir)

    if not seizure_info:
        return [], 0

    seizures = []
    fs = None

    for edf_name, file_seizures in seizure_info.items():
        if not file_seizures:
            continue

        edf_path = patient_dir / edf_name
        if not edf_path.exists():
            continue

        try:
            data, file_fs, ch_names = load_edf(edf_path)
            fs = file_fs
        except Exception as e:
            warnings.warn(f"Failed to load {edf_path}: {e}")
            continue

        # Filter to EEG channels only (exclude ECG, VNS trigger, etc.)
        # Note: CHB-MIT uses bipolar montage like "FP1-F7" - don't filter on '-'
        eeg_idx = [i for i, ch in enumerate(ch_names)
                   if not any(x in ch.upper() for x in ['ECG', 'VNS', 'PHOTIC', 'EKG', 'LOC', 'ROC'])]

        if not eeg_idx:
            continue

        data = data[eeg_idx]
        n_channels = data.shape[0]
        window_samples = int(PREPROCESSING['window_sec'] * fs)

        # Preprocess entire file
        data = preprocess_eeg(data, fs=fs,
                              bandpass=tuple(PREPROCESSING['bandpass']),
                              notch=PREPROCESSING['notch'],
                              normalize=PREPROCESSING['normalize'])

        # Process each seizure in this file
        for sz in file_seizures:
            if 'start' not in sz or 'end' not in sz:
                continue

            start_sample = int(sz['start'] * fs)
            end_sample = int(sz['end'] * fs)

            if end_sample > data.shape[1] or start_sample >= end_sample:
                continue

            # Extract ictal segment
            ictal_data = data[:, start_sample:end_sample]
            ictal_windows = extract_windows(ictal_data, window_samples)

            if len(ictal_windows) == 0:
                continue

            n_ictal = len(ictal_windows)

            # Extract interictal (before seizure, >30s gap)
            interictal_end = max(0, start_sample - 30 * fs)
            n_interictal_needed = int(n_ictal * max_interictal_ratio)
            interictal_start = max(0, interictal_end - n_interictal_needed * window_samples)

            if interictal_start < interictal_end:
                interictal_data = data[:, interictal_start:interictal_end]
                interictal_windows = extract_windows(interictal_data, window_samples)

                # Limit to ratio
                if len(interictal_windows) > n_interictal_needed:
                    interictal_windows = interictal_windows[:n_interictal_needed]
            else:
                interictal_windows = np.array([])

            seizures.append({
                'ictal_windows': ictal_windows,
                'interictal_windows': interictal_windows,
                'file_name': edf_name,
                'n_channels': n_channels,
                'seizure_start': sz['start'],
                'seizure_end': sz['end']
            })

    return seizures, fs


# =============================================================================
# CLASSIFIER AND CV
# =============================================================================

def run_loso_cv(seizures: List[Dict], fs: int) -> Dict:
    """
    Leave-one-seizure-out cross-validation.

    For each fold, train on N-1 seizures, test on 1.

    Returns dict with AUC stats.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    n_seizures = len(seizures)

    if n_seizures < 2:
        return {
            'cv_auc_mean': 0.0,
            'cv_auc_std': 0.0,
            'cv_auc_per_fold': [],
            'n_seizures': n_seizures,
            'error': 'Insufficient seizures for CV'
        }

    fold_aucs = []

    for test_idx in range(n_seizures):
        # Split
        test_seizure = seizures[test_idx]
        train_seizures = [s for i, s in enumerate(seizures) if i != test_idx]

        # Build training set
        X_train = []
        y_train = []

        for sz in train_seizures:
            # Ictal
            for window in sz['ictal_windows']:
                feat = extract_fft_power_bands(window, fs)
                X_train.append(feat)
                y_train.append(1)

            # Interictal
            for window in sz['interictal_windows']:
                feat = extract_fft_power_bands(window, fs)
                X_train.append(feat)
                y_train.append(0)

        # Build test set
        X_test = []
        y_test = []

        for window in test_seizure['ictal_windows']:
            feat = extract_fft_power_bands(window, fs)
            X_test.append(feat)
            y_test.append(1)

        for window in test_seizure['interictal_windows']:
            feat = extract_fft_power_bands(window, fs)
            X_test.append(feat)
            y_test.append(0)

        if len(X_train) < 10 or len(X_test) < 2:
            continue

        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            continue

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Train Random Forest (500 trees per plan feedback)
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        # Predict probabilities
        y_prob = clf.predict_proba(X_test)[:, 1]

        # Compute AUC
        try:
            auc = roc_auc_score(y_test, y_prob)
            fold_aucs.append(auc)
        except:
            pass

    if not fold_aucs:
        return {
            'cv_auc_mean': 0.0,
            'cv_auc_std': 0.0,
            'cv_auc_per_fold': [],
            'n_seizures': n_seizures,
            'error': 'No valid folds'
        }

    return {
        'cv_auc_mean': float(np.mean(fold_aucs)),
        'cv_auc_std': float(np.std(fold_aucs)),
        'cv_auc_per_fold': [float(a) for a in fold_aucs],
        'n_seizures': n_seizures,
        'n_valid_folds': len(fold_aucs)
    }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_patient_benchmark(patient_dir: Path, output_dir: Path) -> Dict:
    """
    Run Tier 1 benchmark for one patient.

    Returns results dict.
    """
    patient_id = patient_dir.name

    print(f"\n{'='*60}")
    print(f"TIER 1 BENCHMARK: {patient_id}")
    print(f"{'='*60}")

    # Load data
    print("Loading patient data...")
    seizures, fs = load_patient_data(patient_dir)

    if not seizures:
        print(f"No seizures found for {patient_id}")
        return {'patient_id': patient_id, 'error': 'No seizures found'}

    print(f"Found {len(seizures)} seizures")

    # Get feature dimension from first seizure
    first_window = seizures[0]['ictal_windows'][0]
    feature_dim = len(extract_fft_power_bands(first_window, fs))
    n_channels = seizures[0]['n_channels']

    print(f"Channels: {n_channels}")
    print(f"Feature dimension: {feature_dim} ({n_channels} channels × 5 bands)")

    # Run LOSO CV
    print("Running leave-one-seizure-out CV...")
    cv_results = run_loso_cv(seizures, fs)

    # Build results
    results = {
        'patient_id': patient_id,
        'feature_dim': feature_dim,
        'n_channels': n_channels,
        'n_seizures': len(seizures),
        'cv_auc_mean': cv_results['cv_auc_mean'],
        'cv_auc_std': cv_results['cv_auc_std'],
        'cv_auc_per_fold': cv_results['cv_auc_per_fold'],
        'n_valid_folds': cv_results.get('n_valid_folds', 0),
        'preprocessing': PREPROCESSING,
        'classifier': {
            'type': 'RandomForest',
            'n_estimators': 500,
            'max_depth': 10
        },
        'timestamp': datetime.now().isoformat()
    }

    if 'error' in cv_results:
        results['error'] = cv_results['error']

    # Print summary
    print(f"\nResults:")
    print(f"  AUC: {results['cv_auc_mean']:.3f} ± {results['cv_auc_std']:.3f}")
    print(f"  Valid folds: {results['n_valid_folds']}/{len(seizures)}")

    # Save JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{patient_id}.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    return results


def run_all_patients(data_dir: Path, output_dir: Path) -> List[Dict]:
    """Run benchmark for all CHB-MIT patients."""

    all_results = []

    for patient_id in CHBMIT_PATIENTS:
        patient_dir = data_dir / patient_id

        if not patient_dir.exists():
            print(f"Skipping {patient_id} (not found)")
            continue

        try:
            results = run_patient_benchmark(patient_dir, output_dir)
            all_results.append(results)
        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            all_results.append({
                'patient_id': patient_id,
                'error': str(e)
            })

    return all_results


def save_summary_csv(results: List[Dict], output_path: Path):
    """Save summary CSV with one row per patient."""

    fieldnames = [
        'patient_id', 'n_seizures', 'n_channels', 'feature_dim',
        'cv_auc_mean', 'cv_auc_std', 'n_valid_folds', 'error'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()

        for r in results:
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)

    print(f"\nSaved summary: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tier 1 FFT Power Band Benchmark')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CHB-MIT dataset directory')
    parser.add_argument('--patient', type=str, default=None,
                        help='Run single patient (e.g., chb01)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: results/tier1_fft_power)')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else \
                 Path(__file__).parent.parent / 'results' / 'tier1_fft_power'

    print("="*60)
    print("TIER 1 CLASSICAL BASELINE: FFT Power Band Features")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Preprocessing: {PREPROCESSING}")
    print(f"Classifier: Random Forest (500 trees)")
    print(f"CV: Leave-one-seizure-out (per patient)")

    if args.patient:
        # Single patient
        patient_dir = data_dir / args.patient
        if not patient_dir.exists():
            print(f"Patient not found: {patient_dir}")
            sys.exit(1)
        results = [run_patient_benchmark(patient_dir, output_dir)]
    else:
        # All patients
        results = run_all_patients(data_dir, output_dir)

    # Save summary CSV
    summary_path = output_dir.parent / 'tier1_summary.csv'
    save_summary_csv(results, summary_path)

    # Print overall summary
    valid_results = [r for r in results if 'error' not in r and r.get('cv_auc_mean', 0) > 0]

    if valid_results:
        aucs = [r['cv_auc_mean'] for r in valid_results]
        print(f"\n{'='*60}")
        print("TIER 1 OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Patients processed: {len(valid_results)}/{len(results)}")
        print(f"Mean AUC: {np.mean(aucs):.3f}")
        print(f"Min AUC: {np.min(aucs):.3f}")
        print(f"Max AUC: {np.max(aucs):.3f}")
        print(f"Std AUC: {np.std(aucs):.3f}")
