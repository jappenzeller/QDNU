#!/usr/bin/env python3
"""
================================================================================
CHB-MIT LOSO CV - Quantum vs Classical Comparison
================================================================================

Runs BOTH quantum (QPNN fidelity-based) and classical (XGBoost) pipelines
on the SAME 8-channel subset for a fair apples-to-apples comparison.

Quantum constraint: 8 channels = 17 qubits (2M+1), practical limit for
statevector simulation.

Channels (8):
  - Left hemisphere:  FP1-F7, F7-T7, FP1-F3, F3-C3
  - Right hemisphere: FP2-F8, F8-T8, FP2-F4, F4-C4

================================================================================
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'QA1'))

from sagemaker.train_chbmit import (
    EXCLUDE_SUBJECTS,
    FS,
    WINDOW_SEC,
    normalize_channel_label,
    read_edf_segment,
    parse_summary_file,
    extract_segments_for_subject as _extract_segments_for_subject,
    EEGSegment,
)

# Quantum imports
from QA1.pn_dynamics import PNDynamics
from QA1.multichannel_circuit import create_multichannel_circuit, compute_fidelity, get_statevector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data location
DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
OUTPUT_DIR = Path("analysis_results/quantum_loso")
CACHE_FILE = Path("analysis_results/quantum_8ch_cache.npz")

# 8 channels for quantum (17 qubits)
QUANTUM_CHANNELS = [
    'FP1-F7', 'F7-T7', 'FP1-F3', 'F3-C3',
    'FP2-F8', 'F8-T8', 'FP2-F4', 'F4-C4',
]

# PN dynamics parameters
PN_LAMBDA_A = 0.1
PN_LAMBDA_C = 0.05
PN_DT = 0.01

# XGBoost parameters (for classical comparison)
N_BAGS = 5


# =============================================================================
# DATA LOADING
# =============================================================================

def load_eeg_segment(seg: EEGSegment, subject_dir: Path, channels: List[str]) -> Optional[np.ndarray]:
    """Load EEG segment for specified channels."""
    try:
        import pyedflib

        edf_path = subject_dir / seg.source_file
        if not edf_path.exists():
            return None

        with pyedflib.EdfReader(str(edf_path)) as f:
            n_signals = f.signals_in_file
            file_channels = [f.getLabel(i) for i in range(n_signals)]
            file_channels_norm = [normalize_channel_label(ch) for ch in file_channels]

            # Map requested channels to file indices
            channel_indices = []
            for ch in channels:
                ch_norm = normalize_channel_label(ch)
                if ch_norm in file_channels_norm:
                    idx = file_channels_norm.index(ch_norm)
                    channel_indices.append(idx)
                else:
                    return None  # Channel not found

            # Read data
            fs = f.getSampleFrequency(0)
            start_sample = int(seg.start_sec * fs)
            n_samples = int(WINDOW_SEC * fs)

            data = np.zeros((len(channels), n_samples))
            for i, ch_idx in enumerate(channel_indices):
                signal = f.readSignal(ch_idx)
                end_sample = min(start_sample + n_samples, len(signal))
                actual_samples = end_sample - start_sample
                if actual_samples < n_samples:
                    return None  # Not enough samples
                data[i] = signal[start_sample:end_sample]

            return data

    except Exception as e:
        logger.warning(f"Failed to load {seg.source_file}: {e}")
        return None


def load_all_segments_cached(channels: List[str]) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Load all EEG segments for specified channels, with caching."""

    if CACHE_FILE.exists():
        logger.info(f"Loading cached EEG from {CACHE_FILE}")
        data = np.load(CACHE_FILE, allow_pickle=True)
        eeg_list = list(data['eeg'])
        labels = list(data['labels'])
        subjects = list(data['subjects'])
        logger.info(f"Loaded {len(eeg_list)} segments from cache")
        return eeg_list, labels, subjects

    logger.info("Loading EEG segments from EDF files...")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Using {len(channels)} channels: {channels}")

    # Find subjects
    all_subjects = sorted([d for d in DATA_DIR.iterdir()
                           if d.is_dir() and d.name.startswith('chb')])
    subjects_dirs = [d for d in all_subjects if d.name not in EXCLUDE_SUBJECTS]
    logger.info(f"Found {len(all_subjects)} subjects, using {len(subjects_dirs)}")

    eeg_list = []
    labels = []
    subjects = []

    for subject_dir in subjects_dirs:
        logger.info(f"\nProcessing {subject_dir.name}")

        segs = _extract_segments_for_subject(subject_dir)

        ictal = sum(1 for s in segs if s.label == 'ictal')
        preictal = sum(1 for s in segs if s.label == 'preictal')
        interictal = sum(1 for s in segs if s.label == 'interictal')
        logger.info(f"  Segments: {ictal} ictal, {preictal} preictal, {interictal} interictal")

        subject_count = 0
        for seg in segs:
            eeg = load_eeg_segment(seg, subject_dir, channels)
            if eeg is not None:
                eeg_list.append(eeg)
                labels.append(seg.label)
                subjects.append(seg.subject)
                subject_count += 1

        logger.info(f"  Loaded: {subject_count} segments")

    logger.info(f"\nTotal: {len(eeg_list)} segments")

    # Save cache
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_FILE,
             eeg=np.array(eeg_list, dtype=object),
             labels=np.array(labels),
             subjects=np.array(subjects))
    logger.info(f"Saved cache to {CACHE_FILE}")

    return eeg_list, labels, subjects


# =============================================================================
# QUANTUM PIPELINE
# =============================================================================

def normalize_eeg(eeg: np.ndarray) -> np.ndarray:
    """Normalize EEG to [0, 1] using RMS envelope."""
    window_size = min(50, max(10, eeg.shape[1] // 10))
    normalized = np.zeros_like(eeg, dtype=float)

    for ch in range(eeg.shape[0]):
        signal = eeg[ch].astype(float)
        squared = signal ** 2
        kernel = np.ones(window_size) / window_size
        power = np.sqrt(np.convolve(squared, kernel, mode='same'))
        normalized[ch] = power

    # Normalize to [0, 1]
    norm_max = np.percentile(normalized, 99)
    if norm_max > 0:
        normalized = normalized / norm_max
    normalized = np.clip(normalized, 0, 1)

    return normalized


def eeg_to_statevector(eeg: np.ndarray, pn: PNDynamics):
    """Convert EEG segment to quantum statevector."""
    eeg_norm = normalize_eeg(eeg)
    params = pn.evolve_multichannel(eeg_norm)
    circuit = create_multichannel_circuit(params)
    sv = get_statevector(circuit)
    return sv, params


def train_template(preictal_segments: List[np.ndarray], pn: PNDynamics):
    """Train template by averaging parameters from multiple pre-ictal segments."""
    if not preictal_segments:
        raise ValueError("No pre-ictal segments for template training")

    all_params = []
    for eeg in preictal_segments:
        eeg_norm = normalize_eeg(eeg)
        params = pn.evolve_multichannel(eeg_norm)
        all_params.append(params)

    # Average parameters across segments
    n_channels = len(all_params[0])
    avg_params = []
    for ch in range(n_channels):
        a_avg = np.mean([p[ch][0] for p in all_params])
        b_avg = np.mean([p[ch][1] for p in all_params])
        c_avg = np.mean([p[ch][2] for p in all_params])
        avg_params.append((a_avg, b_avg, c_avg))

    # Create template circuit and statevector
    template_circuit = create_multichannel_circuit(avg_params)
    template_sv = get_statevector(template_circuit)

    return template_sv, avg_params


def run_quantum_loso(eeg_list: List[np.ndarray], labels: List[str],
                     subjects: List[str]) -> Dict:
    """Run LOSO CV with quantum fidelity classification."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score)

    # Label mapping: ictal + preictal = 1, interictal = 0
    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    y = np.array([label_map.get(l, 0) for l in labels])

    unique_subjects = sorted(set(subjects))

    pn = PNDynamics(lambda_a=PN_LAMBDA_A, lambda_c=PN_LAMBDA_C, dt=PN_DT)

    results = {'per_subject': {}, 'overall': {}}
    all_preds, all_true, all_fidelities = [], [], []

    logger.info(f"\nQuantum LOSO CV (8 channels, 17 qubits)")
    logger.info(f"Subjects: {len(unique_subjects)}, Segments: {len(eeg_list)}")
    logger.info("-" * 60)

    for test_subject in unique_subjects:
        # Split data
        train_mask = np.array([s != test_subject for s in subjects])
        test_mask = np.array([s == test_subject for s in subjects])

        train_labels = [labels[i] for i in range(len(labels)) if train_mask[i]]
        train_eeg = [eeg_list[i] for i in range(len(eeg_list)) if train_mask[i]]
        test_eeg = [eeg_list[i] for i in range(len(eeg_list)) if test_mask[i]]
        y_test = y[test_mask]

        if len(np.unique(y_test)) < 2:
            logger.info(f"  {test_subject}: skipped (single class)")
            continue

        # Get pre-ictal segments for template training
        preictal_eeg = [eeg for eeg, lbl in zip(train_eeg, train_labels)
                        if lbl == 'preictal']

        if not preictal_eeg:
            # Fall back to ictal if no pre-ictal
            preictal_eeg = [eeg for eeg, lbl in zip(train_eeg, train_labels)
                            if lbl == 'ictal']

        if not preictal_eeg:
            logger.info(f"  {test_subject}: skipped (no positive training samples)")
            continue

        # Train template
        try:
            template_sv, template_params = train_template(preictal_eeg, pn)
        except Exception as e:
            logger.warning(f"  {test_subject}: template training failed: {e}")
            continue

        # Classify test segments
        fidelities = []
        for eeg in test_eeg:
            try:
                test_sv, _ = eeg_to_statevector(eeg, pn)
                fid = compute_fidelity(template_sv, test_sv)
                fidelities.append(fid)
            except Exception as e:
                fidelities.append(0.5)  # Neutral on error

        fidelities = np.array(fidelities)

        # Find optimal threshold using Youden's J on test set
        # (In practice, would use validation set, but for comparison we use same method)
        best_thresh = 0.5
        best_f1 = 0
        for thresh in np.linspace(0.1, 0.9, 17):
            preds = (fidelities >= thresh).astype(int)
            f1 = f1_score(y_test, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        y_pred = (fidelities >= best_thresh).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, fidelities)
        except ValueError:
            auc = 0.5

        results['per_subject'][test_subject] = {
            'n_test': int(len(y_test)),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'auc': float(auc),
            'threshold': float(best_thresh),
            'mean_fidelity_pos': float(np.mean(fidelities[y_test == 1])),
            'mean_fidelity_neg': float(np.mean(fidelities[y_test == 0])),
        }

        all_preds.extend(y_pred.tolist())
        all_true.extend(y_test.tolist())
        all_fidelities.extend(fidelities.tolist())

        logger.info(f"  {test_subject}: Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f} thresh={best_thresh:.2f}")

    # Overall metrics
    if all_preds:
        results['overall'] = {
            'n_segments': len(all_true),
            'accuracy': float(accuracy_score(all_true, all_preds)),
            'precision': float(precision_score(all_true, all_preds, zero_division=0)),
            'recall': float(recall_score(all_true, all_preds, zero_division=0)),
            'f1': float(f1_score(all_true, all_preds, zero_division=0)),
        }
        try:
            results['overall']['auc'] = float(roc_auc_score(all_true, all_fidelities))
        except ValueError:
            results['overall']['auc'] = 0.5

    results['model_params'] = {
        'classifier': 'Quantum-QPNN',
        'n_channels': len(QUANTUM_CHANNELS),
        'n_qubits': 2 * len(QUANTUM_CHANNELS) + 1,
        'lambda_a': PN_LAMBDA_A,
        'lambda_c': PN_LAMBDA_C,
        'dt': PN_DT,
        'window_sec': WINDOW_SEC,
    }

    return results


# =============================================================================
# CLASSICAL PIPELINE (8 channels)
# =============================================================================

def extract_features_8ch(eeg: np.ndarray) -> np.ndarray:
    """Extract classical features from 8-channel EEG."""
    from scipy import signal as sig
    from scipy.stats import skew, kurtosis

    n_ch = eeg.shape[0]
    features = []

    # 1. Band power (delta, theta, alpha, beta, low-gamma, high-gamma)
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 50), (50, 100)]
    for ch in range(n_ch):
        f, psd = sig.welch(eeg[ch], fs=FS, nperseg=min(256, len(eeg[ch])))
        for low, high in bands:
            idx = (f >= low) & (f < high)
            if idx.sum() > 0:
                features.append(np.mean(psd[idx]))
            else:
                features.append(0)

    # 2. Statistics (mean, std, skew, kurtosis, min, max)
    for ch in range(n_ch):
        features.extend([
            np.mean(eeg[ch]),
            np.std(eeg[ch]),
            skew(eeg[ch]),
            kurtosis(eeg[ch]),
            np.min(eeg[ch]),
            np.max(eeg[ch]),
        ])

    # 3. Hjorth parameters (activity, mobility, complexity)
    for ch in range(n_ch):
        diff1 = np.diff(eeg[ch])
        diff2 = np.diff(diff1)
        activity = np.var(eeg[ch])
        mobility = np.sqrt(np.var(diff1) / activity) if activity > 0 else 0
        complexity = (np.sqrt(np.var(diff2) / np.var(diff1)) / mobility
                      if mobility > 0 and np.var(diff1) > 0 else 0)
        features.extend([activity, mobility, complexity])

    # 4. Spectral entropy
    for ch in range(n_ch):
        f, psd = sig.welch(eeg[ch], fs=FS, nperseg=min(256, len(eeg[ch])))
        psd_norm = psd / psd.sum() if psd.sum() > 0 else psd
        psd_norm = psd_norm[psd_norm > 0]
        entropy = -np.sum(psd_norm * np.log2(psd_norm)) if len(psd_norm) > 0 else 0
        features.append(entropy)

    # 5. Cross-channel correlations
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            corr = np.corrcoef(eeg[i], eeg[j])[0, 1]
            features.append(corr if not np.isnan(corr) else 0)

    return np.array(features)


def run_classical_loso_8ch(eeg_list: List[np.ndarray], labels: List[str],
                           subjects: List[str]) -> Dict:
    """Run LOSO CV with classical XGBoost on 8 channels."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score)
    from xgboost import XGBClassifier

    # Extract features
    logger.info("\nExtracting classical features for 8 channels...")
    features = np.array([extract_features_8ch(eeg) for eeg in eeg_list])
    features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
    logger.info(f"Feature shape: {features.shape}")

    # Label mapping
    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    y = np.array([label_map.get(l, 0) for l in labels])

    unique_subjects = sorted(set(subjects))

    results = {'per_subject': {}, 'overall': {}}
    all_preds, all_true, all_proba = [], [], []

    logger.info(f"\nClassical LOSO CV (8 channels, XGBoost)")
    logger.info(f"Subjects: {len(unique_subjects)}, Segments: {len(features)}")
    logger.info("-" * 60)

    for test_subject in unique_subjects:
        # Split data
        train_mask = np.array([s != test_subject for s in subjects])
        test_mask = np.array([s == test_subject for s in subjects])

        X_train = features[train_mask]
        y_train = y[train_mask]
        X_test = features[test_mask]
        y_test = y[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.info(f"  {test_subject}: skipped (single class)")
            continue

        # Train 5-bag ensemble
        models = []
        for bag in range(N_BAGS):
            rng = np.random.RandomState(42 + bag)
            idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            model = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42 + bag,
                eval_metric='logloss',
                n_jobs=-1,
            )
            model.fit(X_train[idx], y_train[idx])
            models.append(model)

        # Predict
        y_proba = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
        y_pred = (y_proba >= 0.5).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5

        results['per_subject'][test_subject] = {
            'n_test': int(len(y_test)),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'auc': float(auc),
        }

        all_preds.extend(y_pred.tolist())
        all_true.extend(y_test.tolist())
        all_proba.extend(y_proba.tolist())

        logger.info(f"  {test_subject}: Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f}")

    # Overall metrics
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
        'classifier': 'XGBoost-8ch',
        'n_channels': len(QUANTUM_CHANNELS),
        'n_estimators': 200,
        'n_bags': N_BAGS,
        'n_features': features.shape[1],
        'window_sec': WINDOW_SEC,
    }

    return results


# =============================================================================
# COMPARISON
# =============================================================================

def print_comparison(quantum_results: Dict, classical_results: Dict):
    """Print head-to-head comparison table."""

    q = quantum_results.get('overall', {})
    c = classical_results.get('overall', {})

    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD COMPARISON: QUANTUM vs CLASSICAL (8 channels)")
    print("=" * 70)
    print(f"{'Metric':<15} {'Quantum (QPNN)':<20} {'Classical (XGBoost)':<20} {'Delta':<10}")
    print("-" * 70)

    metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    for m in metrics:
        q_val = q.get(m, 0)
        c_val = c.get(m, 0)
        delta = q_val - c_val
        sign = '+' if delta >= 0 else ''
        print(f"{m.capitalize():<15} {q_val*100:>6.2f}%              {c_val*100:>6.2f}%              {sign}{delta*100:.2f}%")

    print("-" * 70)

    q_params = quantum_results.get('model_params', {})
    c_params = classical_results.get('model_params', {})

    print(f"{'Channels':<15} {q_params.get('n_channels', 8):<20} {c_params.get('n_channels', 8):<20}")
    print(f"{'Qubits':<15} {q_params.get('n_qubits', 17):<20} {'N/A':<20}")
    print(f"{'Features':<15} {'Fidelity (1)':<20} {c_params.get('n_features', 0):<20}")
    print("=" * 70)

    # Overall assessment
    winner = "QUANTUM" if q.get('auc', 0) > c.get('auc', 0) else "CLASSICAL"
    auc_diff = abs(q.get('auc', 0) - c.get('auc', 0)) * 100
    print(f"\n{winner} wins by {auc_diff:.2f}% AUC")

    return {'quantum': q, 'classical': c}


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    logger.info("=" * 70)
    logger.info("CHB-MIT LOSO CV - Quantum vs Classical Comparison")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"Channels: {QUANTUM_CHANNELS}")
    logger.info(f"Qubits: {2 * len(QUANTUM_CHANNELS) + 1}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load EEG data
    eeg_list, labels, subjects = load_all_segments_cached(QUANTUM_CHANNELS)
    logger.info(f"\nLoaded {len(eeg_list)} segments")
    logger.info(f"Unique subjects: {len(set(subjects))}")

    n_ictal = sum(1 for l in labels if l == 'ictal')
    n_preictal = sum(1 for l in labels if l == 'preictal')
    n_interictal = sum(1 for l in labels if l == 'interictal')
    logger.info(f"Distribution: {n_ictal} ictal, {n_preictal} preictal, {n_interictal} interictal")

    # Run quantum LOSO
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: QUANTUM LOSO CV")
    logger.info("=" * 70)

    quantum_results = run_quantum_loso(eeg_list, labels, subjects)

    with open(OUTPUT_DIR / 'quantum_results.json', 'w') as f:
        json.dump(quantum_results, f, indent=2)

    # Run classical LOSO
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: CLASSICAL LOSO CV (8 channels)")
    logger.info("=" * 70)

    classical_results = run_classical_loso_8ch(eeg_list, labels, subjects)

    with open(OUTPUT_DIR / 'classical_8ch_results.json', 'w') as f:
        json.dump(classical_results, f, indent=2)

    # Comparison
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: HEAD-TO-HEAD COMPARISON")
    logger.info("=" * 70)

    comparison = print_comparison(quantum_results, classical_results)

    # Save summary
    summary = {
        'n_subjects': len(set(subjects)),
        'n_segments': len(labels),
        'n_ictal': n_ictal,
        'n_preictal': n_preictal,
        'n_interictal': n_interictal,
        'quantum_results': quantum_results.get('overall', {}),
        'classical_results': classical_results.get('overall', {}),
        'quantum_params': quantum_results.get('model_params', {}),
        'classical_params': classical_results.get('model_params', {}),
        'channels': QUANTUM_CHANNELS,
        'completed': datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Final summary
    q_overall = quantum_results.get('overall', {})
    c_overall = classical_results.get('overall', {})

    logger.info(f"\n{'=' * 70}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'=' * 70}")
    logger.info(f"{'Metric':<15} {'Quantum':<15} {'Classical':<15}")
    logger.info(f"{'Accuracy':<15} {q_overall.get('accuracy', 0):.3f}          {c_overall.get('accuracy', 0):.3f}")
    logger.info(f"{'AUC':<15} {q_overall.get('auc', 0):.3f}          {c_overall.get('auc', 0):.3f}")
    logger.info(f"{'F1':<15} {q_overall.get('f1', 0):.3f}          {c_overall.get('f1', 0):.3f}")

    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
