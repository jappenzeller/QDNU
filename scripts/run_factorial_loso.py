#!/usr/bin/env python3
"""
================================================================================
2x2 Factorial LOSO Evaluation
================================================================================
IV1: Paradigm (Classical / Quantum)
IV2: Feature type (Spectral / Non-spectral)

Cell 1: Classical x Spectral   — XGBoost on band power (40 features)
Cell 2: Classical x Non-spectral — XGBoost on PLV coherence (28 features)
Cell 3: Quantum x Spectral     — V2 band power encoding (from saved results)
Cell 4: Quantum x Non-spectral — V3 PLV theta-alpha encoding (from saved results)

Same LOSO protocol as quantum_loso_v2: 22 subjects, 504 segments,
5-bag XGBoost ensemble, 30s windows.
================================================================================
"""

import sys
import json
import csv
import numpy as np
from pathlib import Path
from scipy.signal import welch, hilbert, butter, filtfilt
from sklearn.metrics import accuracy_score, roc_auc_score
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from sagemaker.train_chbmit import (
    EXCLUDE_SUBJECTS, FS, WINDOW_SEC,
    normalize_channel_label,
    extract_segments_for_subject as _extract_segments_for_subject,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
OUTPUT_DIR = PROJECT_ROOT

QUANTUM_CHANNELS = [
    'FP1-F7', 'F7-T7', 'FP1-F3', 'F3-C3',
    'FP2-F8', 'F8-T8', 'FP2-F4', 'F4-C4',
]

N_BAGS = 5
SUBJECTS_22 = sorted([
    d.name for d in DATA_DIR.iterdir()
    if d.is_dir() and d.name.startswith('chb') and d.name not in EXCLUDE_SUBJECTS
])


# =============================================================================
# DATA LOADING (same as quantum_loso_v2)
# =============================================================================

def load_eeg_segment(seg, subject_dir, channels):
    """Load EEG segment for specified channels."""
    import pyedflib

    edf_path = subject_dir / seg.source_file
    if not edf_path.exists():
        return None

    try:
        with pyedflib.EdfReader(str(edf_path)) as f:
            n_signals = f.signals_in_file
            file_channels = [normalize_channel_label(f.getLabel(i)) for i in range(n_signals)]

            channel_indices = []
            for ch in channels:
                ch_norm = normalize_channel_label(ch)
                if ch_norm in file_channels:
                    channel_indices.append(file_channels.index(ch_norm))
                else:
                    return None

            fs = f.getSampleFrequency(0)
            start_sample = int(seg.start_sec * fs)
            n_samples = int(WINDOW_SEC * fs)

            data = np.zeros((len(channels), n_samples))
            for i, ch_idx in enumerate(channel_indices):
                signal = f.readSignal(ch_idx)
                end_sample = min(start_sample + n_samples, len(signal))
                actual = end_sample - start_sample
                if actual < n_samples:
                    return None
                data[i] = signal[start_sample:end_sample]

            return data
    except Exception:
        return None


def load_all_data():
    """Load all segments across all subjects."""
    eeg_list = []
    labels = []
    subjects = []

    for subject_id in SUBJECTS_22:
        subject_dir = DATA_DIR / subject_id
        if not subject_dir.exists():
            continue

        segs = _extract_segments_for_subject(subject_dir)
        for seg in segs:
            eeg = load_eeg_segment(seg, subject_dir, QUANTUM_CHANNELS)
            if eeg is not None:
                eeg_list.append(eeg)
                labels.append(seg.label)
                subjects.append(subject_id)

    logger.info(f"Loaded {len(eeg_list)} segments from {len(set(subjects))} subjects")
    return eeg_list, labels, subjects


# =============================================================================
# CELL 1: SPECTRAL FEATURES (band power only, 40 features)
# =============================================================================

def extract_spectral_features(data: np.ndarray, fs: float = 256.0) -> np.ndarray:
    """
    Extract band power for 5 standard EEG bands per channel.
    delta (0.5-4), theta (4-8), alpha (8-13), beta (13-30), gamma (30-100)
    8 channels x 5 bands = 40 features.
    """
    data = np.asarray(data, dtype=np.float64)
    n_channels = data.shape[0]

    bands = [
        (0.5, 4.0),    # delta
        (4.0, 8.0),    # theta
        (8.0, 13.0),   # alpha
        (13.0, 30.0),  # beta
        (30.0, 100.0),  # gamma
    ]

    features = []
    for ch in range(n_channels):
        freqs, psd = welch(data[ch], fs=fs, nperseg=min(512, len(data[ch])))
        total_power = np.sum(psd)
        if total_power < 1e-20:
            total_power = 1e-20

        for f_low, f_high in bands:
            mask = (freqs >= f_low) & (freqs < f_high)
            band_power = np.sum(psd[mask])
            features.append(np.log10(band_power / total_power + 1e-20))

    return np.array(features, dtype=np.float64)


# =============================================================================
# CELL 2: NON-SPECTRAL FEATURES (PLV coherence, 28 features)
# =============================================================================

def extract_plv_features(data: np.ndarray, fs: float = 256.0,
                         band: tuple = (4, 13)) -> np.ndarray:
    """
    Extract Phase Locking Value between all channel pairs.
    Matches V3 feature space: Hilbert PLV in theta-alpha band (4-13 Hz).
    8 channels -> C(8,2) = 28 pairwise PLVs.
    """
    data = np.asarray(data, dtype=np.float64)
    n_channels, n_samples = data.shape

    # Bandpass filter (same as V3 encoding)
    nyq = fs / 2
    low, high = band[0] / nyq, min(band[1] / nyq, 0.99)
    try:
        b_filt, a_filt = butter(4, [low, high], btype='band')
    except ValueError:
        return np.zeros(n_channels * (n_channels - 1) // 2)

    # Extract phases via Hilbert transform
    phases = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        try:
            filtered = filtfilt(b_filt, a_filt, data[ch])
            analytic = hilbert(filtered)
            phases[ch] = np.angle(analytic)
        except Exception:
            phases[ch] = np.zeros(n_samples)

    # Compute pairwise PLV (upper triangle)
    features = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phases[i] - phases[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            features.append(plv)

    return np.array(features, dtype=np.float64)


# =============================================================================
# LOSO CV ENGINE (same protocol as quantum_loso_v2)
# =============================================================================

def run_loso_cv(features: np.ndarray, labels: list, subjects: list,
                cell_name: str) -> dict:
    """Run LOSO CV with 5-bag XGBoost ensemble."""
    from xgboost import XGBClassifier

    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    y = np.array([label_map.get(l, 0) for l in labels])
    unique_subjects = sorted(set(subjects))

    results = {'per_subject': {}}
    all_preds, all_true, all_proba = [], [], []

    logger.info(f"\n{cell_name}: LOSO CV ({features.shape[1]} features, "
                f"{len(unique_subjects)} subjects, {len(features)} segments)")

    for test_subject in unique_subjects:
        train_mask = np.array([s != test_subject for s in subjects])
        test_mask = np.array([s == test_subject for s in subjects])

        X_train, y_train = features[train_mask], y[train_mask]
        X_test, y_test = features[test_mask], y[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        # 5-bag ensemble (identical to quantum_loso_v2)
        models = []
        for bag in range(N_BAGS):
            rng = np.random.RandomState(42 + bag)
            idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            model = XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42 + bag,
                eval_metric='logloss', n_jobs=-1,
            )
            model.fit(X_train[idx], y_train[idx])
            models.append(model)

        y_proba = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
        y_pred = (y_proba >= 0.5).astype(int)

        acc = float(accuracy_score(y_test, y_pred))
        try:
            auc = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            auc = 0.5

        results['per_subject'][test_subject] = {
            'n_test': int(len(y_test)),
            'accuracy': acc,
            'auc': auc,
        }

        all_preds.extend(y_pred.tolist())
        all_true.extend(y_test.tolist())
        all_proba.extend(y_proba.tolist())

        logger.info(f"  {test_subject}: Acc={acc:.3f} AUC={auc:.4f}")

    if all_preds:
        results['overall'] = {
            'n_segments': len(all_true),
            'accuracy': float(accuracy_score(all_true, all_preds)),
            'auc': float(roc_auc_score(all_true, all_proba)),
        }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("2x2 Factorial LOSO Evaluation")
    logger.info("=" * 70)

    # Load quantum results (Cells 3 & 4 - already computed)
    v2_path = PROJECT_ROOT / 'analysis_results/quantum_loso_v2/quantum_results.json'
    v3_path = PROJECT_ROOT / 'analysis_results/quantum_loso_v3/quantum_theta_alpha_results.json'

    with open(v2_path) as f:
        quantum_spectral = json.load(f)
    with open(v3_path) as f:
        quantum_nonspectral = json.load(f)

    # Verify quantum aggregates
    v2_auc = quantum_spectral['overall']['auc']
    v3_auc = quantum_nonspectral['overall']['auc']
    assert abs(v2_auc - 0.534) < 0.005, f"V2 AUC mismatch: {v2_auc}"
    assert abs(v3_auc - 0.460) < 0.005, f"V3 AUC mismatch: {v3_auc}"
    logger.info(f"Quantum V2 (spectral) pooled AUC: {v2_auc:.4f} (verified)")
    logger.info(f"Quantum V3 (non-spectral) pooled AUC: {v3_auc:.4f} (verified)")

    # Load EEG data for classical cells
    logger.info("\nLoading EEG data...")
    eeg_list, labels, subjects = load_all_data()

    # Cell 1: Classical x Spectral (band power)
    logger.info("\nExtracting spectral features (band power)...")
    spectral_features = np.array([extract_spectral_features(eeg, FS) for eeg in eeg_list])
    spectral_features = np.nan_to_num(spectral_features, nan=0.0, posinf=10.0, neginf=-10.0)
    classical_spectral = run_loso_cv(spectral_features, labels, subjects,
                                     "Cell 1: Classical x Spectral")

    # Cell 2: Classical x Non-spectral (PLV)
    logger.info("\nExtracting non-spectral features (PLV coherence)...")
    plv_features = np.array([extract_plv_features(eeg, FS, band=(4, 13)) for eeg in eeg_list])
    plv_features = np.nan_to_num(plv_features, nan=0.0, posinf=1.0, neginf=0.0)
    classical_nonspectral = run_loso_cv(plv_features, labels, subjects,
                                        "Cell 2: Classical x Non-spectral")

    # Report
    print("\n" + "=" * 70)
    print("FACTORIAL RESULTS")
    print("=" * 70)
    print(f"{'Cell':<35} {'Pooled AUC':<12} {'Pooled Acc':<12}")
    print("-" * 70)
    print(f"{'Cell 1: Classical x Spectral':<35} "
          f"{classical_spectral['overall']['auc']:.4f}       "
          f"{classical_spectral['overall']['accuracy']:.4f}")
    print(f"{'Cell 2: Classical x Non-spectral':<35} "
          f"{classical_nonspectral['overall']['auc']:.4f}       "
          f"{classical_nonspectral['overall']['accuracy']:.4f}")
    print(f"{'Cell 3: Quantum x Spectral':<35} "
          f"{quantum_spectral['overall']['auc']:.4f}       "
          f"{quantum_spectral['overall']['accuracy']:.4f}")
    print(f"{'Cell 4: Quantum x Non-spectral':<35} "
          f"{quantum_nonspectral['overall']['auc']:.4f}       "
          f"{quantum_nonspectral['overall']['accuracy']:.4f}")
    print("=" * 70)

    # Build CSV data
    all_subjects = sorted(set(
        list(classical_spectral['per_subject'].keys()) +
        list(classical_nonspectral['per_subject'].keys()) +
        list(quantum_spectral['per_subject'].keys()) +
        list(quantum_nonspectral['per_subject'].keys())
    ))

    def get_val(results, subject, key):
        entry = results.get('per_subject', {}).get(subject)
        if entry is None:
            return 'NaN'
        return f"{entry[key]:.4f}"

    # AUC CSV
    auc_comment = (
        "# classical_spectral: XGBoost 5-bag on band power (5 bands x 8ch = 40 features) | "
        "classical_nonspectral: XGBoost 5-bag on PLV coherence (28 channel pairs, 4-13Hz) | "
        "quantum_spectral: analysis_results/quantum_loso_v2/quantum_results.json (V2 band-power encoding, agg=0.534) | "
        "quantum_nonspectral: analysis_results/quantum_loso_v3/quantum_theta_alpha_results.json (V3 PLV 4-13Hz, agg=0.460)\n"
    )

    auc_path = OUTPUT_DIR / 'QDNU_factorial_AUC.csv'
    with open(auc_path, 'w', newline='') as f:
        f.write(auc_comment)
        writer = csv.writer(f)
        writer.writerow(['subject', 'classical_spectral', 'classical_nonspectral',
                         'quantum_spectral', 'quantum_nonspectral'])
        for subj in all_subjects:
            writer.writerow([
                subj,
                get_val(classical_spectral, subj, 'auc'),
                get_val(classical_nonspectral, subj, 'auc'),
                get_val(quantum_spectral, subj, 'auc'),
                get_val(quantum_nonspectral, subj, 'auc'),
            ])

    # Accuracy CSV
    acc_comment = (
        "# classical_spectral: XGBoost 5-bag on band power (5 bands x 8ch = 40 features) | "
        "classical_nonspectral: XGBoost 5-bag on PLV coherence (28 channel pairs, 4-13Hz) | "
        "quantum_spectral: analysis_results/quantum_loso_v2/quantum_results.json (V2 band-power encoding) | "
        "quantum_nonspectral: analysis_results/quantum_loso_v3/quantum_theta_alpha_results.json (V3 PLV 4-13Hz)\n"
    )

    acc_path = OUTPUT_DIR / 'QDNU_factorial_Accuracy.csv'
    with open(acc_path, 'w', newline='') as f:
        f.write(acc_comment)
        writer = csv.writer(f)
        writer.writerow(['subject', 'classical_spectral', 'classical_nonspectral',
                         'quantum_spectral', 'quantum_nonspectral'])
        for subj in all_subjects:
            writer.writerow([
                subj,
                get_val(classical_spectral, subj, 'accuracy'),
                get_val(classical_nonspectral, subj, 'accuracy'),
                get_val(quantum_spectral, subj, 'accuracy'),
                get_val(quantum_nonspectral, subj, 'accuracy'),
            ])

    # Save full results as JSON for reference
    full_results = {
        'classical_spectral': classical_spectral,
        'classical_nonspectral': classical_nonspectral,
        'quantum_spectral_agg': quantum_spectral['overall'],
        'quantum_nonspectral_agg': quantum_nonspectral['overall'],
    }
    json_path = OUTPUT_DIR / 'results' / 'encoding_analysis' / 'factorial_results.json'
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(full_results, f, indent=2)

    logger.info(f"\nSaved: {auc_path}")
    logger.info(f"Saved: {acc_path}")
    logger.info(f"Saved: {json_path}")


if __name__ == '__main__':
    main()
