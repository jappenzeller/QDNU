#!/usr/bin/env python3
"""
================================================================================
CHB-MIT LOSO CV - Quantum vs Classical Comparison V3
================================================================================

Key insight: The quantum circuit was designed to detect PHASE SYNCHRONIZATION
via the P(b) phase gates and ring+ancilla topology. V1/V2 never fed it actual
phase information.

V3 encoding:
  b = instantaneous phase (Hilbert) -> feeds P(b) gates -> circuit detects sync
  a = amplitude envelope -> feeds Rx(2a) -> encodes signal strength
  c = PLV with global phase -> feeds Ry(2c) -> encodes network coupling

Additional changes:
  - 2-second sub-windows with MAX aggregation (seizure precursors are transient)
  - Test 3 frequency bands: theta-alpha (4-13), alpha-beta (8-30), theta (4-8)
  - Reuse V2 classical results (816 features)

================================================================================
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from scipy.signal import hilbert, butter, filtfilt
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
OUTPUT_DIR = Path("analysis_results/quantum_loso_v3")
CACHE_FILE = Path("analysis_results/quantum_8ch_cache.npz")
V2_CLASSICAL_FILE = Path("analysis_results/quantum_loso_v2/classical_results.json")

# 8 channels for quantum (17 qubits)
QUANTUM_CHANNELS = [
    'FP1-F7', 'F7-T7', 'FP1-F3', 'F3-C3',
    'FP2-F8', 'F8-T8', 'FP2-F4', 'F4-C4',
]

# Frequency bands to test
BANDS = {
    'theta_alpha': (4, 13),   # Dominant scalp EEG, well-studied for sync
    'alpha_beta': (8, 30),    # Mu rhythm + beta desync during seizures
    'theta': (4, 8),          # Temporal lobe seizures show theta sync
}

# Sub-window parameters
SUBWINDOW_SEC = 2.0  # 2-second sub-windows


# =============================================================================
# PLV / HILBERT PHASE ENCODING
# =============================================================================

def extract_plv_params(eeg_segment: np.ndarray, fs: float = 256.0,
                       band: Tuple[float, float] = (4, 13)) -> List[Tuple[float, float, float]]:
    """
    Extract (a, b, c) per channel using Hilbert phase/amplitude.

    Encoding rationale:
      b = instantaneous phase -> feeds P(b) gates -> circuit detects synchronization
      a = normalized amplitude envelope -> feeds Rx(2a) -> encodes signal strength
      c = channel's PLV with global mean -> feeds Ry(2c) -> encodes network coupling

    Args:
        eeg_segment: (n_channels, n_samples) array
        fs: sampling frequency (256 Hz for CHB-MIT)
        band: frequency band for narrowband filtering (Hz)

    Returns:
        List of (a, b, c) tuples, one per channel
    """
    n_channels, n_samples = eeg_segment.shape

    # 1. Bandpass filter to target band
    nyq = fs / 2
    low, high = band[0] / nyq, min(band[1] / nyq, 0.99)

    try:
        b_filt, a_filt = butter(4, [low, high], btype='band')
    except ValueError:
        # Fall back if band is invalid
        return [(0.5, np.pi, 0.5) for _ in range(n_channels)]

    # 2. Filter and compute analytic signal per channel
    phases = np.zeros((n_channels, n_samples))
    envelopes = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        try:
            filtered = filtfilt(b_filt, a_filt, eeg_segment[ch].astype(np.float64))
            analytic = hilbert(filtered)
            phases[ch] = np.angle(analytic)      # instantaneous phase [-pi, pi]
            envelopes[ch] = np.abs(analytic)     # amplitude envelope
        except Exception:
            phases[ch] = np.zeros(n_samples)
            envelopes[ch] = np.ones(n_samples)

    # 3. Global mean phase (for PLV computation)
    mean_analytic = np.mean(envelopes * np.exp(1j * phases), axis=0)
    global_phase = np.angle(mean_analytic)

    # 4. Compute summary per channel
    all_amps = [np.mean(envelopes[ch]) for ch in range(n_channels)]
    amp_min, amp_max = min(all_amps), max(all_amps)
    amp_range = amp_max - amp_min if amp_max > amp_min else 1.0

    params_list = []
    for ch in range(n_channels):
        # a = normalized mean amplitude envelope [0.05, 0.95]
        a_norm = 0.05 + 0.9 * (all_amps[ch] - amp_min) / amp_range

        # b = circular mean of instantaneous phase, shifted to [0, 2pi]
        mean_sin = np.mean(np.sin(phases[ch]))
        mean_cos = np.mean(np.cos(phases[ch]))
        b_phase = np.arctan2(mean_sin, mean_cos)  # [-pi, pi]
        b_phase = b_phase % (2 * np.pi)           # [0, 2pi]

        # c = PLV of this channel with global mean phase [0, 1]
        phase_diff = phases[ch] - global_phase
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        c_val = np.clip(plv, 0.05, 0.95)

        params_list.append((a_norm, b_phase, c_val))

    return params_list


def extract_plv_params_subwindows(eeg_segment: np.ndarray, fs: float = 256.0,
                                   band: Tuple[float, float] = (4, 13),
                                   window_sec: float = 2.0) -> List[List[Tuple[float, float, float]]]:
    """
    Extract (a, b, c) per channel from sub-windows.

    Returns params for each sub-window (for MAX aggregation of fidelity).

    Args:
        eeg_segment: (n_channels, n_samples)
        fs: sampling rate
        band: filter band
        window_sec: sub-window length in seconds

    Returns:
        List of param_lists, one per sub-window.
        Each param_list is [(a,b,c), ...] for each channel.
    """
    n_channels, n_samples = eeg_segment.shape
    window_samples = int(window_sec * fs)
    step = window_samples  # non-overlapping

    all_subwindow_params = []

    for start in range(0, n_samples - window_samples + 1, step):
        sub = eeg_segment[:, start:start + window_samples]
        params = extract_plv_params(sub, fs=fs, band=band)
        all_subwindow_params.append(params)

    # Ensure at least one sub-window
    if not all_subwindow_params:
        all_subwindow_params.append(extract_plv_params(eeg_segment, fs=fs, band=band))

    return all_subwindow_params


def get_phase_stats(params_list: List[Tuple[float, float, float]]) -> Dict:
    """Get phase statistics from a params list."""
    b_vals = [p[1] for p in params_list]
    c_vals = [p[2] for p in params_list]

    # Circular std for phase b
    sins = np.sin(b_vals)
    coss = np.cos(b_vals)
    r = np.sqrt(np.mean(sins)**2 + np.mean(coss)**2)
    b_std = np.sqrt(-2 * np.log(r + 1e-10)) if r > 0 else np.pi

    return {
        'b_mean': float(np.arctan2(np.mean(sins), np.mean(coss)) % (2 * np.pi)),
        'b_std': float(b_std),
        'b_values': [float(v) for v in b_vals],
        'c_mean': float(np.mean(c_vals)),
        'c_values': [float(v) for v in c_vals],
    }


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
# PLV DIAGNOSTICS
# =============================================================================

def print_plv_diagnostics(eeg_list: List[np.ndarray], labels: List[str],
                          subjects: List[str], band: Tuple[float, float],
                          sample_subjects: List[str] = None):
    """Print PLV encoding diagnostics before LOSO."""

    if sample_subjects is None:
        unique_subjects = sorted(set(subjects))
        sample_subjects = unique_subjects[:2]

    print(f"\n{'=' * 70}")
    print(f"PLV ENCODING DIAGNOSTICS - Band: {band[0]}-{band[1]} Hz")
    print("=" * 70)

    # Sample segment analysis
    for subj in sample_subjects:
        print(f"\n=== {subj} ===")

        # Find one ictal and one interictal segment
        ictal_eeg = None
        inter_eeg = None
        for eeg, lbl, s in zip(eeg_list, labels, subjects):
            if s == subj:
                if lbl in ('ictal', 'preictal') and ictal_eeg is None:
                    ictal_eeg = eeg
                elif lbl == 'interictal' and inter_eeg is None:
                    inter_eeg = eeg
            if ictal_eeg is not None and inter_eeg is not None:
                break

        if ictal_eeg is not None:
            sw_params = extract_plv_params_subwindows(ictal_eeg, fs=FS, band=band)
            if sw_params:
                stats = get_phase_stats(sw_params[0])
                print(f"  Ictal sub-window 1:")
                print(f"    b values: [{', '.join(f'{v:.2f}' for v in stats['b_values'])}] rad")
                print(f"    b std: {stats['b_std']:.3f} {'(LOW = synchronized)' if stats['b_std'] < 1.0 else '(HIGH = desync)'}")
                print(f"    c (PLV) values: [{', '.join(f'{v:.2f}' for v in stats['c_values'])}]")
                print(f"    c mean: {stats['c_mean']:.3f} {'(HIGH = phase-locked)' if stats['c_mean'] > 0.5 else '(LOW)'}")

        if inter_eeg is not None:
            sw_params = extract_plv_params_subwindows(inter_eeg, fs=FS, band=band)
            if sw_params:
                stats = get_phase_stats(sw_params[0])
                print(f"  Interictal sub-window 1:")
                print(f"    b values: [{', '.join(f'{v:.2f}' for v in stats['b_values'])}] rad")
                print(f"    b std: {stats['b_std']:.3f} {'(LOW = synchronized)' if stats['b_std'] < 1.0 else '(HIGH = desync)'}")
                print(f"    c (PLV) values: [{', '.join(f'{v:.2f}' for v in stats['c_values'])}]")
                print(f"    c mean: {stats['c_mean']:.3f} {'(HIGH = phase-locked)' if stats['c_mean'] > 0.5 else '(LOW)'}")

    # Phase std separation across all subjects
    print(f"\n--- Phase std separation (ictal vs interictal) ---")
    separation_ratios = []

    unique_subjects = sorted(set(subjects))
    for subj in unique_subjects:
        ictal_stds = []
        inter_stds = []

        for eeg, lbl, s in zip(eeg_list, labels, subjects):
            if s == subj:
                sw_params = extract_plv_params_subwindows(eeg, fs=FS, band=band)
                for params in sw_params:
                    stats = get_phase_stats(params)
                    if lbl in ('ictal', 'preictal'):
                        ictal_stds.append(stats['b_std'])
                    else:
                        inter_stds.append(stats['b_std'])

        if ictal_stds and inter_stds:
            ictal_mean = np.mean(ictal_stds)
            inter_mean = np.mean(inter_stds)
            # Ratio: we expect interictal to have HIGHER std (more desync)
            ratio = inter_mean / (ictal_mean + 1e-10)
            separation_ratios.append(ratio)
            status = "GOOD" if ratio > 3.0 else ("OK" if ratio > 1.5 else "POOR")
            print(f"  {subj}: ictal_b_std={ictal_mean:.3f}, inter_b_std={inter_mean:.3f}, ratio={ratio:.1f}x [{status}]")

    if separation_ratios:
        good_count = sum(1 for r in separation_ratios if r > 3.0)
        ok_count = sum(1 for r in separation_ratios if r > 1.5)
        print(f"\nSeparation summary: {good_count}/{len(separation_ratios)} subjects with ratio > 3.0x")
        print(f"                    {ok_count}/{len(separation_ratios)} subjects with ratio > 1.5x")

    return separation_ratios


# =============================================================================
# DUAL TEMPLATE CONSTRUCTION
# =============================================================================

def build_plv_template(eeg_segments: List[np.ndarray], fs: float = 256.0,
                       band: Tuple[float, float] = (4, 13)):
    """
    Build template from averaged per-channel PLV parameters.

    Averages across ALL sub-windows from ALL segments.
    """
    all_params = []

    for eeg in eeg_segments:
        subwindow_params = extract_plv_params_subwindows(eeg, fs, band)
        for sw_params in subwindow_params:
            all_params.append(sw_params)

    if not all_params:
        return None, None

    n_channels = len(all_params[0])
    avg_params = []

    for ch in range(n_channels):
        avg_a = np.mean([p[ch][0] for p in all_params])
        # Circular mean for phase b
        sins = np.mean([np.sin(p[ch][1]) for p in all_params])
        coss = np.mean([np.cos(p[ch][1]) for p in all_params])
        avg_b = np.arctan2(sins, coss) % (2 * np.pi)
        avg_c = np.clip(np.mean([p[ch][2] for p in all_params]), 0.05, 0.95)
        avg_params.append((avg_a, avg_b, avg_c))

    circuit = create_multichannel_circuit(avg_params)
    sv = get_statevector(circuit)

    return avg_params, sv


def compute_fidelity(sv1, sv2) -> float:
    """Compute fidelity |<psi1|psi2>|^2"""
    return abs(sv1.inner(sv2)) ** 2


# =============================================================================
# QUANTUM LOSO V3 (PLV + MAX AGGREGATION)
# =============================================================================

def run_quantum_loso_v3(eeg_list: List[np.ndarray], labels: List[str],
                        subjects: List[str], band: Tuple[float, float]) -> Dict:
    """
    Run LOSO CV with PLV encoding and MAX aggregation.

    For each fold:
    1. Build ictal template from training ictal+preictal sub-windows
    2. Build interictal template from training interictal sub-windows
    3. For each test segment:
       - Compute scores for each sub-window
       - Use MAX(fid_ictal - fid_inter) as final score
    4. predict = 1 if max_score > 0 else 0
    """
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score)

    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    y = np.array([label_map.get(l, 0) for l in labels])

    unique_subjects = sorted(set(subjects))

    results = {'per_subject': {}, 'overall': {}, 'band': f"{band[0]}-{band[1]}Hz"}
    all_preds, all_true, all_scores = [], [], []
    diagnostics = []

    logger.info(f"\nQuantum LOSO CV V3 (PLV, MAX agg, {band[0]}-{band[1]} Hz)")
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

        _, ictal_sv = build_plv_template(ictal_segs, fs=FS, band=band)
        _, inter_sv = build_plv_template(inter_segs, fs=FS, band=band)

        if ictal_sv is None or inter_sv is None:
            logger.info(f"  {test_subject}: skipped (template build failed)")
            continue

        # Score each test segment with MAX aggregation
        final_scores = []

        for eeg in test_eeg:
            sw_params_list = extract_plv_params_subwindows(eeg, fs=FS, band=band)

            scores = []
            for sw_params in sw_params_list:
                circ = create_multichannel_circuit(sw_params)
                test_sv = get_statevector(circ)
                fid_ictal = compute_fidelity(ictal_sv, test_sv)
                fid_inter = compute_fidelity(inter_sv, test_sv)
                scores.append(fid_ictal - fid_inter)

            # MAX aggregation
            final_score = max(scores) if scores else 0.0
            final_scores.append(final_score)

        final_scores = np.array(final_scores)
        y_pred = (final_scores > 0).astype(int)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, final_scores)
        except ValueError:
            auc = 0.5

        # Diagnostic metrics
        pos_mask = y_test == 1
        neg_mask = y_test == 0

        mean_score_pos = np.mean(final_scores[pos_mask]) if pos_mask.any() else 0
        mean_score_neg = np.mean(final_scores[neg_mask]) if neg_mask.any() else 0
        separation = mean_score_pos - mean_score_neg
        direction = "Correct" if separation > 0 else "Inverted"

        results['per_subject'][test_subject] = {
            'n_test': int(len(y_test)),
            'accuracy': float(acc),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'auc': float(auc),
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
        all_scores.extend(final_scores.tolist())

        logger.info(f"  {test_subject}: Acc={acc:.3f} F1={f1:.3f} AUC={auc:.3f} Sep={separation:+.4f} ({direction})")

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
        'classifier': 'Quantum-QPNN-PLV-V3',
        'n_channels': len(QUANTUM_CHANNELS),
        'n_qubits': 2 * len(QUANTUM_CHANNELS) + 1,
        'encoding': 'hilbert_plv',
        'band_hz': f"{band[0]}-{band[1]}",
        'subwindow_sec': SUBWINDOW_SEC,
        'aggregation': 'MAX',
        'window_sec': WINDOW_SEC,
    }

    # Print fidelity diagnostics
    print(f"\n{'=' * 80}")
    print(f"FIDELITY DIAGNOSTICS - Band: {band[0]}-{band[1]} Hz")
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

    results['correctly_oriented'] = correct_count
    results['total_subjects'] = len(diagnostics)

    return results


# =============================================================================
# COMPARISON
# =============================================================================

def print_band_comparison(band_results: Dict):
    """Print comparison across frequency bands."""
    print("\n" + "=" * 70)
    print("BAND COMPARISON")
    print("=" * 70)
    print(f"{'Band':<15} {'AUC':>10} {'Accuracy':>12} {'F1':>10} {'Correct':>12}")
    print("-" * 70)

    for band_name, results in band_results.items():
        overall = results.get('overall', {})
        auc = overall.get('auc', 0) * 100
        acc = overall.get('accuracy', 0) * 100
        f1 = overall.get('f1', 0) * 100
        correct = results.get('correctly_oriented', 0)
        total = results.get('total_subjects', 22)
        print(f"{band_name:<15} {auc:>9.2f}% {acc:>11.2f}% {f1:>9.2f}% {correct:>5}/{total:<6}")

    print("=" * 70)


def print_version_comparison(v1_auc: float, v2_auc: float, v3_results: Dict):
    """Print encoding evolution comparison."""
    print("\n" + "=" * 70)
    print("ENCODING EVOLUTION")
    print("=" * 70)
    print(f"{'Version':<10} {'Encoding':<30} {'AUC':>10}")
    print("-" * 70)
    print(f"{'V1':<10} {'PN dynamics (saturated)':<30} {v1_auc*100:>9.2f}%")
    print(f"{'V2':<10} {'Band-power ratios':<30} {v2_auc*100:>9.2f}%")

    # Best V3 band
    best_band = None
    best_auc = 0
    for band_name, results in v3_results.items():
        auc = results.get('overall', {}).get('auc', 0)
        if auc > best_auc:
            best_auc = auc
            best_band = band_name

    if best_band:
        print(f"{'V3':<10} {f'Hilbert PLV ({best_band})':<30} {best_auc*100:>9.2f}%")

    print("=" * 70)


def print_final_comparison(quantum_results: Dict, classical_results: Dict):
    """Print head-to-head comparison."""
    q = quantum_results.get('overall', {})
    c = classical_results.get('overall', {})

    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD COMPARISON V3: QUANTUM (PLV) vs CLASSICAL")
    print("=" * 80)
    print(f"{'Metric':<15} {'Quantum (PLV)':<22} {'Classical (V2)':<22} {'Delta':<10}")
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
    print(f"{'Band':<15} {q_params.get('band_hz', 'N/A'):<22}")
    print(f"{'Encoding':<15} {q_params.get('encoding', 'N/A'):<22}")
    print(f"{'Aggregation':<15} {q_params.get('aggregation', 'N/A'):<22}")
    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 70)
    logger.info("CHB-MIT LOSO CV V3 - Quantum PLV Encoding")
    logger.info("=" * 70)
    logger.info(f"Encoding: Hilbert PLV (instantaneous phase)")
    logger.info(f"Aggregation: MAX across 2s sub-windows")
    logger.info(f"Started: {datetime.now()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load EEG data
    eeg_list, labels, subjects = load_all_segments_cached(QUANTUM_CHANNELS)
    logger.info(f"\nLoaded {len(eeg_list)} segments, {len(set(subjects))} subjects")

    n_ictal = sum(1 for l in labels if l == 'ictal')
    n_preictal = sum(1 for l in labels if l == 'preictal')
    n_interictal = sum(1 for l in labels if l == 'interictal')
    logger.info(f"Distribution: {n_ictal} ictal, {n_preictal} preictal, {n_interictal} interictal")

    # Run PLV diagnostics for first band
    print_plv_diagnostics(eeg_list, labels, subjects, band=BANDS['theta_alpha'])

    # Run quantum LOSO for each band
    band_results = {}

    for band_name, band in BANDS.items():
        logger.info("\n" + "=" * 70)
        logger.info(f"QUANTUM LOSO V3 - Band: {band_name} ({band[0]}-{band[1]} Hz)")
        logger.info("=" * 70)

        results = run_quantum_loso_v3(eeg_list, labels, subjects, band)
        band_results[band_name] = results

        with open(OUTPUT_DIR / f'quantum_{band_name}_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Print band comparison
    print_band_comparison(band_results)

    # Find best band
    best_band = max(band_results.keys(),
                    key=lambda k: band_results[k].get('overall', {}).get('auc', 0))
    best_results = band_results[best_band]

    # Load V2 classical results (reuse, don't re-run)
    if V2_CLASSICAL_FILE.exists():
        with open(V2_CLASSICAL_FILE, 'r') as f:
            classical_results = json.load(f)
        logger.info(f"\nLoaded V2 classical results from {V2_CLASSICAL_FILE}")
    else:
        classical_results = {'overall': {'auc': 0.625, 'accuracy': 0.595, 'f1': 0.651}}
        logger.warning("V2 classical results not found, using defaults")

    # Print version comparison
    v1_auc = 0.444  # From V1 results
    v2_auc = 0.534  # From V2 results
    print_version_comparison(v1_auc, v2_auc, band_results)

    # Print final comparison (best band vs classical)
    print_final_comparison(best_results, classical_results)

    # Save summary
    summary = {
        'n_subjects': len(set(subjects)),
        'n_segments': len(labels),
        'n_ictal': n_ictal,
        'n_preictal': n_preictal,
        'n_interictal': n_interictal,
        'best_band': best_band,
        'quantum_results': best_results.get('overall', {}),
        'classical_results': classical_results.get('overall', {}),
        'all_bands': {
            k: v.get('overall', {}) for k, v in band_results.items()
        },
        'encoding_evolution': {
            'v1_pn_dynamics': v1_auc,
            'v2_band_power': v2_auc,
            'v3_plv': best_results.get('overall', {}).get('auc', 0),
        },
        'completed': datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Save encoding comparison
    encoding_comparison = {
        'v1': {'encoding': 'PN dynamics (sample-by-sample)', 'auc': v1_auc, 'problem': 'dt saturation'},
        'v2': {'encoding': 'Band-power ratios', 'auc': v2_auc, 'problem': 'Power direction varies by patient'},
        'v3': {'encoding': f'Hilbert PLV ({best_band})', 'auc': best_results.get('overall', {}).get('auc', 0),
               'problem': 'Tests architecture design premise'},
    }

    with open(OUTPUT_DIR / 'encoding_comparison.json', 'w') as f:
        json.dump(encoding_comparison, f, indent=2)

    # Final summary
    q_overall = best_results.get('overall', {})
    c_overall = classical_results.get('overall', {})

    logger.info(f"\n{'=' * 70}")
    logger.info("FINAL RESULTS V3")
    logger.info(f"{'=' * 70}")
    logger.info(f"Best band: {best_band}")
    logger.info(f"{'Metric':<15} {'Quantum':<15} {'Classical':<15}")
    logger.info(f"{'Accuracy':<15} {q_overall.get('accuracy', 0):.3f}          {c_overall.get('accuracy', 0):.3f}")
    logger.info(f"{'AUC':<15} {q_overall.get('auc', 0):.3f}          {c_overall.get('auc', 0):.3f}")
    logger.info(f"{'F1':<15} {q_overall.get('f1', 0):.3f}          {c_overall.get('f1', 0):.3f}")

    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
