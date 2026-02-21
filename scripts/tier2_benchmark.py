#!/usr/bin/env python3
"""
Tier 2 Classical Baseline: MichaelHills Correlation Eigenvalue Features

This is the STRONG classical baseline from the Kaggle UPenn/Mayo seizure detection
competition winner. The key insight: eigenvalues of inter-channel correlation matrices
capture synchronization structure that raw power features miss.

Feature extraction (per 1-second window):
1. FFT magnitude (1-40 Hz, log scale)
2. Correlation eigenvalues in frequency domain
3. Correlation eigenvalues in time domain

The eigenvalues of correlation matrices are the classical approximation to quantum
density matrix eigenvalues - central to the quantum practicality argument.

Usage:
    python tier2_benchmark.py --data_dir /path/to/chb-mit [--patient chb01]
"""

import numpy as np
import json
import csv
from scipy import signal
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import argparse
import sys
import warnings

sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# CONSTANTS
# =============================================================================

PREPROCESSING = {
    'bandpass': [0.5, 40],
    'notch': 60,
    'window_sec': 1,
    'normalize': 'channel'
}

# CHB-MIT patients (22 total, chb12 missing)
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
    """Preprocess EEG using validated clinical pipeline."""
    data = data.astype(np.float64).copy()
    data = data - np.mean(data, axis=1, keepdims=True)

    nyq = fs / 2
    low = bandpass[0] / nyq
    high = min(bandpass[1] / nyq, 0.99)

    if low > 0 and low < high:
        try:
            b, a = signal.butter(5, [low, high], btype='band')
            data = signal.filtfilt(b, a, data, axis=1)
        except ValueError:
            pass

    if notch and notch < nyq:
        try:
            b_notch, a_notch = signal.iirnotch(notch, Q=30, fs=fs)
            data = signal.filtfilt(b_notch, a_notch, data, axis=1)
        except ValueError:
            pass

    if normalize == 'channel':
        for ch in range(data.shape[0]):
            std = np.std(data[ch])
            if std > 1e-10:
                data[ch] = (data[ch] - np.mean(data[ch])) / std

    return data


def extract_windows(data: np.ndarray, window_samples: int) -> np.ndarray:
    """Extract non-overlapping windows."""
    n_channels, n_samples = data.shape
    n_windows = n_samples // window_samples

    if n_windows == 0:
        return np.array([])

    data = data[:, :n_windows * window_samples]
    windows = data.reshape(n_channels, n_windows, window_samples)
    windows = np.transpose(windows, (1, 0, 2))

    return windows


# =============================================================================
# TIER 2: MICHAELHILLS CORRELATION EIGENVALUE FEATURES
# =============================================================================

def extract_michaelhills_features(window: np.ndarray, fs: int = 256) -> np.ndarray:
    """
    Extract MichaelHills-style correlation eigenvalue features.

    This is the winning feature set from the Kaggle UPenn/Mayo seizure detection
    competition. Key insight: eigenvalues of inter-channel correlation matrices
    capture synchronization structure.

    Args:
        window: (n_channels, n_samples) - preprocessed EEG window
        fs: Sampling frequency

    Returns:
        Feature vector containing:
        - FFT log magnitude (1-40 Hz): n_channels * 40
        - Frequency-domain correlation eigenvalues: n_channels
        - Time-domain correlation eigenvalues: n_channels
    """
    n_channels, n_samples = window.shape

    # Step 1: FFT magnitude (1-40 Hz)
    # For 256 Hz sampling with 256 samples (1 sec): freq resolution = 1 Hz
    # Bins 1-40 correspond to 1-40 Hz
    fft_result = np.fft.rfft(window, axis=1)
    fft_mag = np.abs(fft_result)

    # Keep 1-40 Hz bins only
    # For 1-second window at 256 Hz: bin k corresponds to k Hz
    # So bins 1:41 give us 1-40 Hz
    max_bin = min(41, fft_mag.shape[1])
    fft_mag = fft_mag[:, 1:max_bin]  # (n_channels, up to 40)

    # Pad if needed (for shorter windows)
    if fft_mag.shape[1] < 40:
        pad_width = 40 - fft_mag.shape[1]
        fft_mag = np.pad(fft_mag, ((0, 0), (0, pad_width)), mode='constant',
                         constant_values=1e-6)

    # Log magnitude (more stable)
    fft_mag = np.log10(fft_mag + 1e-6)

    # Step 2: Correlation in frequency domain
    # Each channel becomes a 40-dimensional frequency vector
    # Correlation across channels in frequency space
    try:
        C_freq = np.corrcoef(fft_mag)
        if np.any(np.isnan(C_freq)):
            C_freq = np.eye(n_channels)
    except:
        C_freq = np.eye(n_channels)

    # Eigenvalues (eigvalsh for symmetric matrices - numerically stable)
    eigenvalues_freq = np.linalg.eigvalsh(C_freq)  # ascending order

    # Step 3: Correlation in time domain
    # Correlation across channels in raw time series
    try:
        C_time = np.corrcoef(window)
        if np.any(np.isnan(C_time)):
            C_time = np.eye(n_channels)
    except:
        C_time = np.eye(n_channels)

    eigenvalues_time = np.linalg.eigvalsh(C_time)  # ascending order

    # Step 4: Concatenate feature vector
    features = np.concatenate([
        fft_mag.flatten(),      # n_channels * 40
        eigenvalues_freq,       # n_channels
        eigenvalues_time        # n_channels
    ])

    return features


# =============================================================================
# CHB-MIT DATA LOADING (reused from tier1)
# =============================================================================

def load_edf(edf_path: Path) -> Tuple[np.ndarray, int, List[str]]:
    """Load EDF file."""
    try:
        import mne
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        data = raw.get_data() * 1e6
        fs = int(raw.info['sfreq'])
        ch_names = raw.info['ch_names']
        return data, fs, ch_names
    except ImportError:
        raise ImportError("MNE required. Install with: pip install mne")


def parse_summary(patient_dir: Path) -> Dict:
    """Parse CHB-MIT summary file."""
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
    """Load all seizure data for a patient."""
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

        # Preprocess
        data = preprocess_eeg(data, fs=fs,
                              bandpass=tuple(PREPROCESSING['bandpass']),
                              notch=PREPROCESSING['notch'],
                              normalize=PREPROCESSING['normalize'])

        # Process each seizure
        for sz in file_seizures:
            if 'start' not in sz or 'end' not in sz:
                continue

            start_sample = int(sz['start'] * fs)
            end_sample = int(sz['end'] * fs)

            if end_sample > data.shape[1] or start_sample >= end_sample:
                continue

            # Ictal windows
            ictal_data = data[:, start_sample:end_sample]
            ictal_windows = extract_windows(ictal_data, window_samples)

            if len(ictal_windows) == 0:
                continue

            n_ictal = len(ictal_windows)

            # Interictal (before seizure, >30s gap)
            interictal_end = max(0, start_sample - 30 * fs)
            n_interictal_needed = int(n_ictal * max_interictal_ratio)
            interictal_start = max(0, interictal_end - n_interictal_needed * window_samples)

            if interictal_start < interictal_end:
                interictal_data = data[:, interictal_start:interictal_end]
                interictal_windows = extract_windows(interictal_data, window_samples)
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
    """Leave-one-seizure-out cross-validation with Tier 2 features."""
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
        test_seizure = seizures[test_idx]
        train_seizures = [s for i, s in enumerate(seizures) if i != test_idx]

        # Build training set
        X_train = []
        y_train = []

        for sz in train_seizures:
            for window in sz['ictal_windows']:
                feat = extract_michaelhills_features(window, fs)
                X_train.append(feat)
                y_train.append(1)

            for window in sz['interictal_windows']:
                feat = extract_michaelhills_features(window, fs)
                X_train.append(feat)
                y_train.append(0)

        # Build test set
        X_test = []
        y_test = []

        for window in test_seizure['ictal_windows']:
            feat = extract_michaelhills_features(window, fs)
            X_test.append(feat)
            y_test.append(1)

        for window in test_seizure['interictal_windows']:
            feat = extract_michaelhills_features(window, fs)
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

        # Handle NaN/Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=10.0, neginf=-10.0)

        # Train RF (500 trees per feedback)
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]

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
    """Run Tier 2 benchmark for one patient."""
    patient_id = patient_dir.name

    print(f"\n{'='*60}")
    print(f"TIER 2 BENCHMARK: {patient_id}")
    print(f"{'='*60}")

    print("Loading patient data...")
    seizures, fs = load_patient_data(patient_dir)

    if not seizures:
        print(f"No seizures found for {patient_id}")
        return {'patient_id': patient_id, 'error': 'No seizures found'}

    print(f"Found {len(seizures)} seizures")

    # Get feature dimension
    first_window = seizures[0]['ictal_windows'][0]
    feature_example = extract_michaelhills_features(first_window, fs)
    feature_dim = len(feature_example)
    n_channels = seizures[0]['n_channels']

    # Feature breakdown
    fft_dim = n_channels * 40
    eig_dim = n_channels * 2

    print(f"Channels: {n_channels}")
    print(f"Feature dimension: {feature_dim}")
    print(f"  - FFT magnitude (1-40 Hz): {fft_dim}")
    print(f"  - Freq eigenvalues: {n_channels}")
    print(f"  - Time eigenvalues: {n_channels}")

    print("Running leave-one-seizure-out CV...")
    cv_results = run_loso_cv(seizures, fs)

    results = {
        'patient_id': patient_id,
        'feature_dim': feature_dim,
        'feature_breakdown': {
            'fft_magnitude': fft_dim,
            'freq_eigenvalues': n_channels,
            'time_eigenvalues': n_channels
        },
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
        'feature_type': 'MichaelHills_correlation_eigenvalue',
        'timestamp': datetime.now().isoformat()
    }

    if 'error' in cv_results:
        results['error'] = cv_results['error']

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
    """Run benchmark for all patients."""
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
    """Save summary CSV."""
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


def generate_comparison(tier1_dir: Path, tier2_results: List[Dict],
                        output_path: Path):
    """Generate Tier 1 vs Tier 2 comparison CSV."""

    comparisons = []

    for t2 in tier2_results:
        patient_id = t2.get('patient_id', '')
        if not patient_id or 'error' in t2:
            continue

        # Load Tier 1 results
        t1_path = tier1_dir / f"{patient_id}.json"
        if not t1_path.exists():
            continue

        with open(t1_path, 'r') as f:
            t1 = json.load(f)

        if 'error' in t1:
            continue

        tier1_auc = t1.get('cv_auc_mean', 0)
        tier2_auc = t2.get('cv_auc_mean', 0)
        delta = tier2_auc - tier1_auc

        comparisons.append({
            'patient_id': patient_id,
            'tier1_auc': tier1_auc,
            'tier2_auc': tier2_auc,
            'delta_auc': delta,
            'tier2_wins': delta > 0
        })

    if comparisons:
        fieldnames = ['patient_id', 'tier1_auc', 'tier2_auc', 'delta_auc', 'tier2_wins']

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for c in comparisons:
                writer.writerow(c)

        print(f"\nSaved comparison: {output_path}")

        # Summary stats
        wins = sum(1 for c in comparisons if c['tier2_wins'])
        total = len(comparisons)
        mean_delta = np.mean([c['delta_auc'] for c in comparisons])

        print(f"\n{'='*60}")
        print("TIER 1 vs TIER 2 COMPARISON")
        print(f"{'='*60}")
        print(f"Tier 2 wins: {wins}/{total} ({100*wins/total:.1f}%)")
        print(f"Mean AUC improvement: {mean_delta:+.3f}")

        if wins < total / 2:
            print("\nWARNING: Tier 2 should outperform Tier 1 on most patients.")
            print("Check for preprocessing or feature extraction bugs.")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tier 2 MichaelHills Benchmark')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to CHB-MIT dataset')
    parser.add_argument('--patient', type=str, default=None,
                        help='Run single patient (e.g., chb01)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--tier1_dir', type=str, default=None,
                        help='Tier 1 results directory for comparison')

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else \
                 Path(__file__).parent.parent / 'results' / 'tier2_correlation_eigenvalue'

    tier1_dir = Path(args.tier1_dir) if args.tier1_dir else \
                Path(__file__).parent.parent / 'results' / 'tier1_fft_power'

    print("="*60)
    print("TIER 2 CLASSICAL BASELINE: MichaelHills Correlation Eigenvalues")
    print("="*60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tier 1 directory: {tier1_dir}")
    print(f"Preprocessing: {PREPROCESSING}")
    print(f"Features: FFT magnitude + correlation eigenvalues (time & freq)")
    print(f"Classifier: Random Forest (500 trees)")

    if args.patient:
        patient_dir = data_dir / args.patient
        if not patient_dir.exists():
            print(f"Patient not found: {patient_dir}")
            sys.exit(1)
        results = [run_patient_benchmark(patient_dir, output_dir)]
    else:
        results = run_all_patients(data_dir, output_dir)

    # Save Tier 2 summary
    summary_path = output_dir.parent / 'tier2_summary.csv'
    save_summary_csv(results, summary_path)

    # Generate comparison if Tier 1 results exist
    if tier1_dir.exists():
        comparison_path = output_dir.parent / 'tier1_vs_tier2_comparison.csv'
        generate_comparison(tier1_dir, results, comparison_path)

    # Overall summary
    valid_results = [r for r in results if 'error' not in r and r.get('cv_auc_mean', 0) > 0]

    if valid_results:
        aucs = [r['cv_auc_mean'] for r in valid_results]
        print(f"\n{'='*60}")
        print("TIER 2 OVERALL SUMMARY")
        print(f"{'='*60}")
        print(f"Patients processed: {len(valid_results)}/{len(results)}")
        print(f"Mean AUC: {np.mean(aucs):.3f}")
        print(f"Min AUC: {np.min(aucs):.3f}")
        print(f"Max AUC: {np.max(aucs):.3f}")
        print(f"Std AUC: {np.std(aucs):.3f}")
