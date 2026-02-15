#!/usr/bin/env python3
"""
================================================================================
CHB-MIT LOSO CV - Quantum vs Classical Comparison V2
================================================================================

Fixes from V1:
1. Band-power PNState extraction (replaces saturating PN dynamics integration)
2. Dual template classification (ictal + interictal templates, relative fidelity)
3. Full V2 feature set for classical comparison (~808 features vs 156)

Quantum: 8 channels = 17 qubits, fidelity-based classification
Classical: 8 channels, XGBoost with full V2 Kaggle-winner features

================================================================================
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from scipy.signal import welch
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'QA1'))

from sagemaker.train_chbmit import (
    EXCLUDE_SUBJECTS,
    FS,
    WINDOW_SEC,
    normalize_channel_label,
    extract_segments_for_subject as _extract_segments_for_subject,
    EEGSegment,
    # Full V2 feature extraction
    extract_band_power,
    extract_fine_spectral,
    extract_fft_magnitudes,
    extract_statistics,
    extract_hjorth,
    extract_spectral_entropy,
    extract_ar_error,
    extract_zero_crossings,
    extract_nonlinear_energy,
    extract_rms,
    extract_correlation_features,
    extract_correlation_eigenvalues,
    extract_freq_correlation,
)

# Quantum imports
from QA1.multichannel_circuit import create_multichannel_circuit, get_statevector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
OUTPUT_DIR = Path("analysis_results/quantum_loso_v2")
CACHE_FILE = Path("analysis_results/quantum_8ch_cache.npz")

# 8 channels for quantum (17 qubits)
QUANTUM_CHANNELS = [
    'FP1-F7', 'F7-T7', 'FP1-F3', 'F3-C3',
    'FP2-F8', 'F8-T8', 'FP2-F4', 'F4-C4',
]

# XGBoost parameters
N_BAGS = 5


# =============================================================================
# FIX 1: BAND-POWER BASED PN PARAMETERS
# =============================================================================

def extract_pn_params_bandpower(eeg_segment: np.ndarray, fs: float = 256.0) -> List[Tuple[float, float, float]]:
    """
    Extract (a, b, c) per channel using band power ratios.

    This replaces the saturating PN dynamics integration with spectral analysis.

    Args:
        eeg_segment: (n_channels, n_samples) array
        fs: sampling frequency

    Returns:
        List of (a, b, c) tuples, one per channel
        - a: excitatory [0.05, 0.95] - high frequency power
        - b: phase [0, 2*pi] - log ratio mapped to radians for circuit
        - c: inhibitory [0.05, 0.95] - low frequency power
    """
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, min(100, fs / 2 - 1)),
    }

    n_channels = eeg_segment.shape[0]
    params_list = []

    for ch in range(n_channels):
        signal = eeg_segment[ch]
        nperseg = min(len(signal), int(4 * fs))
        if nperseg < 16:
            nperseg = len(signal)

        freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

        power = {}
        for name, (lo, hi) in bands.items():
            idx = (freqs >= lo) & (freqs <= hi)
            power[name] = np.trapezoid(psd[idx], freqs[idx]) if np.any(idx) else 0.0

        total = sum(power.values())
        if total == 0:
            params_list.append((0.5, np.pi, 0.5))  # Neutral state
            continue

        rel = {name: val / total for name, val in power.items()}

        # a = excitatory (high frequency: beta + gamma)
        a = np.clip(rel['beta'] + rel['gamma'], 0.05, 0.95)

        # c = inhibitory (low frequency: delta + theta)
        c = np.clip(rel['delta'] + rel['theta'], 0.05, 0.95)

        # b = phase (log ratio mapped to [0, 2*pi] for quantum circuit)
        high = power['beta'] + power['gamma']
        low = power['delta'] + power['theta']
        ratio = high / (low + 1e-10)
        log_ratio = np.log10(ratio + 1e-10)
        b_01 = 1.0 / (1.0 + np.exp(-2.0 * log_ratio))  # sigmoid -> [0, 1]
        b = b_01 * 2 * np.pi  # Scale to [0, 2*pi] for phase gates

        params_list.append((a, b, c))

    return params_list


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

            channel_indices = []
            for ch in channels:
                ch_norm = normalize_channel_label(ch)
                if ch_norm in file_channels_norm:
                    idx = file_channels_norm.index(ch_norm)
                    channel_indices.append(idx)
                else:
                    return None

            fs = f.getSampleFrequency(0)
            start_sample = int(seg.start_sec * fs)
            n_samples = int(WINDOW_SEC * fs)

            data = np.zeros((len(channels), n_samples))
            for i, ch_idx in enumerate(channel_indices):
                signal = f.readSignal(ch_idx)
                end_sample = min(start_sample + n_samples, len(signal))
                actual_samples = end_sample - start_sample
                if actual_samples < n_samples:
                    return None
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
    all_subjects = sorted([d for d in DATA_DIR.iterdir()
                           if d.is_dir() and d.name.startswith('chb')])
    subjects_dirs = [d for d in all_subjects if d.name not in EXCLUDE_SUBJECTS]

    eeg_list = []
    labels = []
    subjects = []

    for subject_dir in subjects_dirs:
        logger.info(f"Processing {subject_dir.name}")
        segs = _extract_segments_for_subject(subject_dir)

        for seg in segs:
            eeg = load_eeg_segment(seg, subject_dir, channels)
            if eeg is not None:
                eeg_list.append(eeg)
                labels.append(seg.label)
                subjects.append(seg.subject)

    logger.info(f"Total: {len(eeg_list)} segments")

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_FILE, eeg=np.array(eeg_list, dtype=object),
             labels=np.array(labels), subjects=np.array(subjects))
    return eeg_list, labels, subjects


# =============================================================================
# FIX 2: DUAL TEMPLATE CLASSIFICATION
# =============================================================================

def build_template(eeg_segments: List[np.ndarray], fs: float = 256.0):
    """
    Build averaged template circuit from multiple EEG segments.

    Uses circular mean for phase parameter (b) since it's in radians.

    Returns:
        (circuit, statevector, avg_params) or (None, None, None) if no segments
    """
    if not eeg_segments:
        return None, None, None

    all_params = []
    for eeg in eeg_segments:
        params = extract_pn_params_bandpower(eeg, fs)
        all_params.append(params)

    n_channels = len(all_params[0])
    avg_params = []

    for ch in range(n_channels):
        # Linear mean for a and c
        avg_a = np.mean([p[ch][0] for p in all_params])
        avg_c = np.mean([p[ch][2] for p in all_params])

        # Circular mean for b (phase in radians)
        sins = np.mean([np.sin(p[ch][1]) for p in all_params])
        coss = np.mean([np.cos(p[ch][1]) for p in all_params])
        avg_b = np.arctan2(sins, coss) % (2 * np.pi)

        avg_params.append((avg_a, avg_b, avg_c))

    circuit = create_multichannel_circuit(avg_params)
    sv = get_statevector(circuit)

    return circuit, sv, avg_params


def compute_fidelity(sv1, sv2) -> float:
    """Compute fidelity |<psi1|psi2>|^2"""
    return abs(sv1.inner(sv2)) ** 2


def print_parameter_diagnostics(eeg_list: List[np.ndarray], labels: List[str],
                                 subjects: List[str], sample_subjects: List[str] = None):
    """Print parameter diagnostics for sample subjects."""

    if sample_subjects is None:
        unique_subjects = sorted(set(subjects))
        sample_subjects = unique_subjects[:2]  # First 2 subjects

    print("\n" + "=" * 70)
    print("PARAMETER DIAGNOSTICS (Band-Power PNState)")
    print("=" * 70)

    for subj in sample_subjects:
        print(f"\n=== {subj} ===")

        # Get segments for this subject
        ictal_params = []
        inter_params = []

        for eeg, lbl, s in zip(eeg_list, labels, subjects):
            if s == subj:
                params = extract_pn_params_bandpower(eeg, fs=FS)
                if lbl in ('ictal', 'preictal'):
                    ictal_params.append(params)
                else:
                    inter_params.append(params)

        if ictal_params:
            n_ch = len(ictal_params[0])
            a_vals = [p[ch][0] for p in ictal_params for ch in range(n_ch)]
            b_vals = [p[ch][1] for p in ictal_params for ch in range(n_ch)]
            c_vals = [p[ch][2] for p in ictal_params for ch in range(n_ch)]
            print(f"  Ictal segments (N={len(ictal_params)}):")
            print(f"    a: mean={np.mean(a_vals):.3f} std={np.std(a_vals):.3f} range=[{np.min(a_vals):.3f}, {np.max(a_vals):.3f}]")
            print(f"    b: mean={np.mean(b_vals):.3f} std={np.std(b_vals):.3f} range=[{np.min(b_vals):.3f}, {np.max(b_vals):.3f}] (radians)")
            print(f"    c: mean={np.mean(c_vals):.3f} std={np.std(c_vals):.3f} range=[{np.min(c_vals):.3f}, {np.max(c_vals):.3f}]")
            ictal_a_mean = np.mean(a_vals)
            ictal_c_mean = np.mean(c_vals)
        else:
            ictal_a_mean = ictal_c_mean = 0.5
            print("  Ictal segments: None")

        if inter_params:
            n_ch = len(inter_params[0])
            a_vals = [p[ch][0] for p in inter_params for ch in range(n_ch)]
            b_vals = [p[ch][1] for p in inter_params for ch in range(n_ch)]
            c_vals = [p[ch][2] for p in inter_params for ch in range(n_ch)]
            print(f"  Interictal segments (N={len(inter_params)}):")
            print(f"    a: mean={np.mean(a_vals):.3f} std={np.std(a_vals):.3f} range=[{np.min(a_vals):.3f}, {np.max(a_vals):.3f}]")
            print(f"    b: mean={np.mean(b_vals):.3f} std={np.std(b_vals):.3f} range=[{np.min(b_vals):.3f}, {np.max(b_vals):.3f}] (radians)")
            print(f"    c: mean={np.mean(c_vals):.3f} std={np.std(c_vals):.3f} range=[{np.min(c_vals):.3f}, {np.max(c_vals):.3f}]")
            inter_a_mean = np.mean(a_vals)
            inter_c_mean = np.mean(c_vals)
        else:
            inter_a_mean = inter_c_mean = 0.5
            print("  Interictal segments: None")

        # Separation
        delta_a = ictal_a_mean - inter_a_mean
        delta_c = ictal_c_mean - inter_c_mean
        a_check = "higher during seizure" if delta_a > 0 else "lower during seizure"
        c_check = "lower during seizure" if delta_c < 0 else "higher during seizure"
        print(f"  Parameter separation (ictal - interictal):")
        print(f"    Delta_a: {delta_a:+.3f} (excitatory {a_check})")
        print(f"    Delta_c: {delta_c:+.3f} (inhibitory {c_check})")


def run_quantum_loso_v2(eeg_list: List[np.ndarray], labels: List[str],
                        subjects: List[str]) -> Dict:
    """
    Run LOSO CV with dual-template quantum fidelity classification.

    For each fold:
    1. Build ictal template from training ictal+preictal segments
    2. Build interictal template from training interictal segments
    3. Classify: score = fidelity_ictal - fidelity_interictal
    4. predict = 1 if score > 0 else 0
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score)

    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    y = np.array([label_map.get(l, 0) for l in labels])

    unique_subjects = sorted(set(subjects))

    results = {'per_subject': {}, 'overall': {}}
    all_preds, all_true, all_scores = [], [], []
    diagnostics = []

    logger.info(f"\nQuantum LOSO CV V2 (Dual Template, Band-Power PNState)")
    logger.info(f"Subjects: {len(unique_subjects)}, Segments: {len(eeg_list)}")
    logger.info("-" * 60)

    for test_subject in unique_subjects:
        train_mask = np.array([s != test_subject for s in subjects])
        test_mask = np.array([s == test_subject for s in subjects])

        train_eeg = [eeg_list[i] for i in range(len(eeg_list)) if train_mask[i]]
        train_labels = [labels[i] for i in range(len(labels)) if train_mask[i]]
        test_eeg = [eeg_list[i] for i in range(len(eeg_list)) if test_mask[i]]
        y_test = y[test_mask]

        if len(np.unique(y_test)) < 2:
            logger.info(f"  {test_subject}: skipped (single class)")
            continue

        # Build dual templates
        ictal_segs = [eeg for eeg, lbl in zip(train_eeg, train_labels) if lbl in ('ictal', 'preictal')]
        inter_segs = [eeg for eeg, lbl in zip(train_eeg, train_labels) if lbl == 'interictal']

        if not ictal_segs or not inter_segs:
            logger.info(f"  {test_subject}: skipped (missing template class)")
            continue

        _, ictal_sv, ictal_params = build_template(ictal_segs, fs=FS)
        _, inter_sv, inter_params = build_template(inter_segs, fs=FS)

        # Classify test segments
        scores = []
        fid_ictal_list = []
        fid_inter_list = []

        for eeg in test_eeg:
            test_params = extract_pn_params_bandpower(eeg, fs=FS)
            test_circuit = create_multichannel_circuit(test_params)
            test_sv = get_statevector(test_circuit)

            fid_ictal = compute_fidelity(ictal_sv, test_sv)
            fid_inter = compute_fidelity(inter_sv, test_sv)

            fid_ictal_list.append(fid_ictal)
            fid_inter_list.append(fid_inter)
            scores.append(fid_ictal - fid_inter)

        scores = np.array(scores)
        fid_ictal_arr = np.array(fid_ictal_list)
        fid_inter_arr = np.array(fid_inter_list)

        y_pred = (scores > 0).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, scores)
        except ValueError:
            auc = 0.5

        # Diagnostic metrics
        pos_mask = y_test == 1
        neg_mask = y_test == 0

        mean_fid_ictal_pos = np.mean(fid_ictal_arr[pos_mask]) if pos_mask.any() else 0
        mean_fid_ictal_neg = np.mean(fid_ictal_arr[neg_mask]) if neg_mask.any() else 0
        mean_fid_inter_pos = np.mean(fid_inter_arr[pos_mask]) if pos_mask.any() else 0
        mean_fid_inter_neg = np.mean(fid_inter_arr[neg_mask]) if neg_mask.any() else 0
        mean_score_pos = np.mean(scores[pos_mask]) if pos_mask.any() else 0
        mean_score_neg = np.mean(scores[neg_mask]) if neg_mask.any() else 0
        separation = mean_score_pos - mean_score_neg
        direction = "Correct" if separation > 0 else "Inverted"

        results['per_subject'][test_subject] = {
            'n_test': int(len(y_test)),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'auc': float(auc),
            'mean_fid_ictal_pos': float(mean_fid_ictal_pos),
            'mean_fid_ictal_neg': float(mean_fid_ictal_neg),
            'mean_fid_inter_pos': float(mean_fid_inter_pos),
            'mean_fid_inter_neg': float(mean_fid_inter_neg),
            'mean_score_pos': float(mean_score_pos),
            'mean_score_neg': float(mean_score_neg),
            'separation': float(separation),
            'direction': direction,
        }

        diagnostics.append({
            'subject': test_subject,
            'score_pos': mean_score_pos,
            'score_neg': mean_score_neg,
            'separation': separation,
            'direction': direction,
            'auc': auc,
        })

        all_preds.extend(y_pred.tolist())
        all_true.extend(y_test.tolist())
        all_scores.extend(scores.tolist())

        logger.info(f"  {test_subject}: Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f} Sep={separation:+.3f} ({direction})")

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
            results['overall']['auc'] = float(roc_auc_score(all_true, all_scores))
        except ValueError:
            results['overall']['auc'] = 0.5

    results['model_params'] = {
        'classifier': 'Quantum-QPNN-DualTemplate',
        'n_channels': len(QUANTUM_CHANNELS),
        'n_qubits': 2 * len(QUANTUM_CHANNELS) + 1,
        'pn_method': 'band_power',
        'window_sec': WINDOW_SEC,
    }

    # Print fidelity diagnostics
    print("\n" + "=" * 80)
    print("FIDELITY DIAGNOSTICS")
    print("=" * 80)
    print(f"{'Subject':<10} {'Score_pos':>10} {'Score_neg':>10} {'Separation':>12} {'Direction':>10} {'AUC':>8}")
    print("-" * 80)
    correct_count = 0
    for d in diagnostics:
        print(f"{d['subject']:<10} {d['score_pos']:>+10.4f} {d['score_neg']:>+10.4f} {d['separation']:>+12.4f} {d['direction']:>10} {d['auc']:>8.3f}")
        if d['direction'] == 'Correct':
            correct_count += 1
    print("-" * 80)
    print(f"Correctly oriented: {correct_count}/{len(diagnostics)} subjects")

    return results


# =============================================================================
# FIX 3: FULL V2 CLASSICAL FEATURES
# =============================================================================

def extract_all_features_v2(data: np.ndarray, fs: int) -> np.ndarray:
    """
    Extract full Kaggle winner feature set (V2).

    For 8 channels: ~808 features
    """
    # Ensure float64 for FFT operations
    data = np.asarray(data, dtype=np.float64)

    features = []
    n_ch = data.shape[0]

    # Spectral features
    features.append(extract_band_power(data, fs))           # n_ch * 6 = 48
    features.append(extract_fine_spectral(data, fs))        # n_ch * 20 = 160

    fmax = min(47.0, fs / 2.0 - 1)
    if fmax > 1.0:
        features.append(extract_fft_magnitudes(data, fs, 1.0, fmax))  # n_ch * 46 = 368

    # Statistical features
    features.append(extract_statistics(data))               # n_ch * 6 = 48
    features.append(extract_hjorth(data))                   # n_ch * 3 = 24
    features.append(extract_spectral_entropy(data, fs))     # n_ch = 8
    features.append(extract_ar_error(data, max_order=6))    # n_ch * 6 = 48
    features.append(extract_zero_crossings(data))           # n_ch * 3 = 24
    features.append(extract_nonlinear_energy(data))         # n_ch = 8
    features.append(extract_rms(data))                      # n_ch = 8

    # Cross-channel features
    if n_ch > 1:
        features.append(extract_correlation_features(data))      # n_ch*(n_ch-1)/2 = 28
        features.append(extract_correlation_eigenvalues(data))   # n_ch = 8
        fmax = min(47.0, fs / 2.0 - 1)
        if fmax > 1.0:
            features.append(extract_freq_correlation(data, fs, 1.0, fmax))

    result = np.concatenate(features)
    result = np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0)
    return result


def run_classical_loso_v2(eeg_list: List[np.ndarray], labels: List[str],
                          subjects: List[str]) -> Dict:
    """Run LOSO CV with classical XGBoost using full V2 features."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score)
    from xgboost import XGBClassifier

    logger.info("\nExtracting full V2 classical features for 8 channels...")
    features = np.array([extract_all_features_v2(eeg, FS) for eeg in eeg_list])
    features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
    logger.info(f"Feature shape: {features.shape}")

    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    y = np.array([label_map.get(l, 0) for l in labels])

    unique_subjects = sorted(set(subjects))

    results = {'per_subject': {}, 'overall': {}}
    all_preds, all_true, all_proba = [], [], []

    logger.info(f"\nClassical LOSO CV V2 (8 channels, {features.shape[1]} features)")
    logger.info(f"Subjects: {len(unique_subjects)}, Segments: {len(features)}")
    logger.info("-" * 60)

    for test_subject in unique_subjects:
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
                n_estimators=200, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                reg_alpha=0.1, reg_lambda=1.0, random_state=42 + bag,
                eval_metric='logloss', n_jobs=-1,
            )
            model.fit(X_train[idx], y_train[idx])
            models.append(model)

        y_proba = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
        y_pred = (y_proba >= 0.5).astype(int)

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
        'classifier': 'XGBoost-8ch-V2',
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

    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD COMPARISON V2: QUANTUM vs CLASSICAL (8 channels)")
    print("=" * 80)
    print(f"{'Metric':<15} {'Quantum (Dual-Tmpl)':<22} {'Classical (V2 Feat)':<22} {'Delta':<10}")
    print("-" * 80)

    metrics = ['accuracy', 'auc', 'f1', 'precision', 'recall']
    for m in metrics:
        q_val = q.get(m, 0)
        c_val = c.get(m, 0)
        delta = q_val - c_val
        sign = '+' if delta >= 0 else ''
        print(f"{m.capitalize():<15} {q_val*100:>6.2f}%                {c_val*100:>6.2f}%                {sign}{delta*100:.2f}%")

    print("-" * 80)

    q_params = quantum_results.get('model_params', {})
    c_params = classical_results.get('model_params', {})

    print(f"{'Channels':<15} {q_params.get('n_channels', 8):<22} {c_params.get('n_channels', 8):<22}")
    print(f"{'Qubits':<15} {q_params.get('n_qubits', 17):<22} {'N/A':<22}")
    print(f"{'Features':<15} {'Dual Fidelity':<22} {c_params.get('n_features', 0):<22}")
    print(f"{'PN Method':<15} {q_params.get('pn_method', 'band_power'):<22} {'N/A':<22}")
    print("=" * 80)

    winner = "QUANTUM" if q.get('auc', 0) > c.get('auc', 0) else "CLASSICAL"
    auc_diff = abs(q.get('auc', 0) - c.get('auc', 0)) * 100
    print(f"\n{winner} wins by {auc_diff:.2f}% AUC")

    return {'quantum': q, 'classical': c}


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("CHB-MIT LOSO CV V2 - Quantum vs Classical Comparison")
    logger.info("=" * 70)
    logger.info(f"Fixes: Band-power PNState, Dual template, Full V2 features")
    logger.info(f"Started: {datetime.now()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load EEG data
    eeg_list, labels, subjects = load_all_segments_cached(QUANTUM_CHANNELS)
    logger.info(f"\nLoaded {len(eeg_list)} segments, {len(set(subjects))} subjects")

    n_ictal = sum(1 for l in labels if l == 'ictal')
    n_preictal = sum(1 for l in labels if l == 'preictal')
    n_interictal = sum(1 for l in labels if l == 'interictal')
    logger.info(f"Distribution: {n_ictal} ictal, {n_preictal} preictal, {n_interictal} interictal")

    # Parameter diagnostics
    print_parameter_diagnostics(eeg_list, labels, subjects)

    # Quantum LOSO V2
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: QUANTUM LOSO CV V2 (Dual Template)")
    logger.info("=" * 70)

    quantum_results = run_quantum_loso_v2(eeg_list, labels, subjects)

    with open(OUTPUT_DIR / 'quantum_results.json', 'w') as f:
        json.dump(quantum_results, f, indent=2)

    # Classical LOSO V2
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: CLASSICAL LOSO CV V2 (Full Features)")
    logger.info("=" * 70)

    classical_results = run_classical_loso_v2(eeg_list, labels, subjects)

    with open(OUTPUT_DIR / 'classical_results.json', 'w') as f:
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
        'fixes': ['band_power_pnstate', 'dual_template', 'full_v2_features'],
        'completed': datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Final summary
    q_overall = quantum_results.get('overall', {})
    c_overall = classical_results.get('overall', {})

    logger.info(f"\n{'=' * 70}")
    logger.info("FINAL RESULTS V2")
    logger.info(f"{'=' * 70}")
    logger.info(f"{'Metric':<15} {'Quantum':<15} {'Classical':<15}")
    logger.info(f"{'Accuracy':<15} {q_overall.get('accuracy', 0):.3f}          {c_overall.get('accuracy', 0):.3f}")
    logger.info(f"{'AUC':<15} {q_overall.get('auc', 0):.3f}          {c_overall.get('auc', 0):.3f}")
    logger.info(f"{'F1':<15} {q_overall.get('f1', 0):.3f}          {c_overall.get('f1', 0):.3f}")

    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
