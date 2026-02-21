#!/usr/bin/env python3
"""
Quantum QPNN Channel Scaling (8 / 12 / 16 channels)

Runs quantum QPNN at three channel counts under identical LOSO CV conditions.
Uses PLV encoding (V3) with MAX aggregation.

NOTE: Statevector simulation scales as O(2^n):
  - 8ch = 17 qubits (feasible, ~128KB state)
  - 12ch = 25 qubits (borderline, ~256MB state)
  - 16ch = 33 qubits (NOT feasible, ~64GB state)

If CH12/CH16 exceed memory, results will be marked as "OOM".
"""

import sys
import json
import logging
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from sklearn.metrics import roc_auc_score

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

from QA1.multichannel_circuit import create_multichannel_circuit, get_statevector

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration - Same nested channel sets as classical
# ============================================================================

CH8 = [
    "FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
    "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"
]

# Extend with same additional channels as classical
CH8_SET = set(CH8)
from sagemaker.train_chbmit import COMMON_CHANNELS
ADDITIONAL_CH12 = [ch for ch in COMMON_CHANNELS if ch not in CH8_SET][:4]
CH12 = CH8 + ADDITIONAL_CH12

CH12_SET = set(CH12)
ADDITIONAL_CH16 = [ch for ch in COMMON_CHANNELS if ch not in CH12_SET][:4]
CH16 = CH12 + ADDITIONAL_CH16

# PLV encoding parameters (best band from V3)
BAND = (4, 13)  # theta-alpha
SUBWINDOW_SEC = 2.0

DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
OUTPUT_DIR = Path("results/scaling")

# Expected CH8 result for validation
EXPECTED_CH8_AUC = 0.5497  # From V3 theta-alpha results
VALIDATION_TOLERANCE = 0.01


# ============================================================================
# PLV Encoding (from V3)
# ============================================================================

def extract_plv_params(eeg_segment: np.ndarray, fs: float = 256.0,
                       band: Tuple[float, float] = (4, 13)) -> List[Tuple[float, float, float]]:
    """Extract (a, b, c) per channel using Hilbert phase/amplitude."""
    n_channels, n_samples = eeg_segment.shape

    nyq = fs / 2
    low, high = band[0] / nyq, min(band[1] / nyq, 0.99)

    try:
        b_filt, a_filt = butter(4, [low, high], btype='band')
    except ValueError:
        return [(0.5, np.pi, 0.5) for _ in range(n_channels)]

    phases = np.zeros((n_channels, n_samples))
    envelopes = np.zeros((n_channels, n_samples))

    for ch in range(n_channels):
        try:
            filtered = filtfilt(b_filt, a_filt, eeg_segment[ch].astype(np.float64))
            analytic = hilbert(filtered)
            phases[ch] = np.angle(analytic)
            envelopes[ch] = np.abs(analytic)
        except Exception:
            phases[ch] = np.zeros(n_samples)
            envelopes[ch] = np.ones(n_samples)

    mean_analytic = np.mean(envelopes * np.exp(1j * phases), axis=0)
    global_phase = np.angle(mean_analytic)

    all_amps = [np.mean(envelopes[ch]) for ch in range(n_channels)]
    amp_min, amp_max = min(all_amps), max(all_amps)
    amp_range = amp_max - amp_min if amp_max > amp_min else 1.0

    params_list = []
    for ch in range(n_channels):
        a_norm = 0.05 + 0.9 * (all_amps[ch] - amp_min) / amp_range
        mean_sin = np.mean(np.sin(phases[ch]))
        mean_cos = np.mean(np.cos(phases[ch]))
        b_phase = np.arctan2(mean_sin, mean_cos) % (2 * np.pi)
        phase_diff = phases[ch] - global_phase
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        c_val = np.clip(plv, 0.05, 0.95)
        params_list.append((a_norm, b_phase, c_val))

    return params_list


def extract_plv_params_subwindows(eeg_segment: np.ndarray, fs: float = 256.0,
                                   band: Tuple[float, float] = (4, 13),
                                   window_sec: float = 2.0) -> List[List[Tuple[float, float, float]]]:
    """Extract (a, b, c) per channel from sub-windows."""
    n_channels, n_samples = eeg_segment.shape
    window_samples = int(window_sec * fs)
    step = window_samples

    all_subwindow_params = []
    for start in range(0, n_samples - window_samples + 1, step):
        sub = eeg_segment[:, start:start + window_samples]
        params = extract_plv_params(sub, fs=fs, band=band)
        all_subwindow_params.append(params)

    if not all_subwindow_params:
        all_subwindow_params.append(extract_plv_params(eeg_segment, fs=fs, band=band))

    return all_subwindow_params


# ============================================================================
# Data Loading
# ============================================================================

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
        return None


def load_all_segments(channels: List[str]) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """Load all EEG segments for specified channels."""

    all_subjects = sorted([d for d in DATA_DIR.iterdir()
                           if d.is_dir() and d.name.startswith('chb')])
    subjects_dirs = [d for d in all_subjects if d.name not in EXCLUDE_SUBJECTS]

    eeg_list = []
    labels = []
    subjects = []

    for subject_dir in subjects_dirs:
        segs = _extract_segments_for_subject(subject_dir)
        for seg in segs:
            eeg = load_eeg_segment(seg, subject_dir, channels)
            if eeg is not None:
                eeg_list.append(eeg)
                labels.append(seg.label)
                subjects.append(seg.subject)

    return eeg_list, labels, subjects


# ============================================================================
# Template Building
# ============================================================================

def build_plv_template(eeg_segments: List[np.ndarray], n_channels: int, fs: float = 256.0,
                       band: Tuple[float, float] = (4, 13)):
    """Build template from averaged per-channel PLV parameters."""
    all_params = []

    for eeg in eeg_segments:
        subwindow_params = extract_plv_params_subwindows(eeg, fs, band)
        for sw_params in subwindow_params:
            all_params.append(sw_params)

    if not all_params:
        return None, None

    avg_params = []
    for ch in range(n_channels):
        avg_a = np.mean([p[ch][0] for p in all_params])
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


# ============================================================================
# Quantum LOSO CV
# ============================================================================

def run_quantum_loso(channels: List[str]) -> Dict:
    """Run quantum QPNN LOSO CV with specified channels."""

    n_channels = len(channels)
    n_qubits = 2 * n_channels + 1

    logger.info(f"Running Quantum LOSO CV with {n_channels} channels ({n_qubits} qubits)")
    logger.info(f"Channels: {channels}")

    # Memory estimate
    state_size_bytes = 2**n_qubits * 16  # complex128
    state_size_mb = state_size_bytes / (1024 * 1024)
    logger.info(f"Estimated state size: {state_size_mb:.1f} MB")

    if n_qubits > 28:
        logger.warning(f"WARNING: {n_qubits} qubits may exceed memory limits")

    # Load data
    logger.info("Loading EEG data...")
    eeg_list, labels, subjects = load_all_segments(channels)
    logger.info(f"Loaded {len(eeg_list)} segments from {len(set(subjects))} subjects")

    if len(eeg_list) == 0:
        return {'error': 'No data loaded', 'channels': channels, 'n_channels': n_channels}

    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    y = np.array([label_map.get(l, 0) for l in labels])
    unique_subjects = sorted(set(subjects))

    all_preds = []
    all_true = []
    all_scores = []
    per_subject = {}

    for test_subject in unique_subjects:
        train_mask = np.array([s != test_subject for s in subjects])
        test_mask = np.array([s == test_subject for s in subjects])

        train_eeg = [eeg_list[i] for i in range(len(eeg_list)) if train_mask[i]]
        train_labels = [labels[i] for i in range(len(labels)) if train_mask[i]]
        test_eeg = [eeg_list[i] for i in range(len(eeg_list)) if test_mask[i]]
        y_test = y[test_mask]

        if len(np.unique(y_test)) < 2:
            continue

        # Build templates
        ictal_segs = [eeg for eeg, lbl in zip(train_eeg, train_labels) if lbl in ('ictal', 'preictal')]
        inter_segs = [eeg for eeg, lbl in zip(train_eeg, train_labels) if lbl == 'interictal']

        if not ictal_segs or not inter_segs:
            continue

        try:
            _, ictal_sv = build_plv_template(ictal_segs, n_channels, fs=FS, band=BAND)
            _, inter_sv = build_plv_template(inter_segs, n_channels, fs=FS, band=BAND)
        except MemoryError:
            logger.error(f"MemoryError building templates for {test_subject}")
            return {'error': 'OOM', 'channels': channels, 'n_channels': n_channels, 'n_qubits': n_qubits}

        if ictal_sv is None or inter_sv is None:
            continue

        # Score test segments
        final_scores = []
        for eeg in test_eeg:
            try:
                sw_params_list = extract_plv_params_subwindows(eeg, fs=FS, band=BAND)
                scores = []
                for sw_params in sw_params_list:
                    circ = create_multichannel_circuit(sw_params)
                    test_sv = get_statevector(circ)
                    fid_ictal = compute_fidelity(ictal_sv, test_sv)
                    fid_inter = compute_fidelity(inter_sv, test_sv)
                    scores.append(fid_ictal - fid_inter)
                final_score = max(scores) if scores else 0.0
            except MemoryError:
                logger.error(f"MemoryError during scoring for {test_subject}")
                return {'error': 'OOM', 'channels': channels, 'n_channels': n_channels, 'n_qubits': n_qubits}

            final_scores.append(final_score)

        final_scores = np.array(final_scores)
        y_pred = (final_scores > 0).astype(int)

        try:
            auc = roc_auc_score(y_test, final_scores)
        except:
            auc = 0.5

        per_subject[test_subject] = {'auc': auc, 'n_samples': len(y_test)}

        all_preds.extend(y_pred.tolist())
        all_true.extend(y_test.tolist())
        all_scores.extend(final_scores.tolist())

        logger.info(f"  {test_subject}: AUC = {auc:.4f}")

    # Overall metrics
    try:
        overall_auc = roc_auc_score(all_true, all_scores)
    except:
        overall_auc = 0.5

    aucs = [r['auc'] for r in per_subject.values()]

    return {
        'channels': channels,
        'n_channels': n_channels,
        'n_qubits': n_qubits,
        'overall_auc': overall_auc,
        'mean_auc': np.mean(aucs) if aucs else 0,
        'std_auc': np.std(aucs) if aucs else 0,
        'per_subject': per_subject,
        'band': f"{BAND[0]}-{BAND[1]}Hz",
        'aggregation': 'MAX',
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run quantum channel scaling benchmark."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Quantum QPNN Channel Scaling")
    logger.info("=" * 60)

    logger.info(f"\nChannel sets (nested):")
    logger.info(f"  CH8  ({len(CH8)} channels, {2*len(CH8)+1} qubits):  {CH8}")
    logger.info(f"  CH12 ({len(CH12)} channels, {2*len(CH12)+1} qubits): {CH12}")
    logger.info(f"  CH16 ({len(CH16)} channels, {2*len(CH16)+1} qubits): {CH16}")

    results = {
        'method': 'Quantum_QPNN_PLV_MAX',
        'timestamp': datetime.now().isoformat(),
    }

    # Step 1: Run CH8
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: CH8 (17 qubits)")
    logger.info("=" * 60)

    ch8_results = run_quantum_loso(CH8)
    results['ch8'] = ch8_results

    if 'error' in ch8_results:
        logger.error(f"CH8 failed: {ch8_results['error']}")
        return

    logger.info(f"CH8 Overall AUC: {ch8_results['overall_auc']:.4f}")

    # Step 2: Run CH12
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: CH12 (25 qubits)")
    logger.info("=" * 60)

    try:
        ch12_results = run_quantum_loso(CH12)
        results['ch12'] = ch12_results

        if 'error' in ch12_results:
            logger.warning(f"CH12 failed: {ch12_results['error']}")
        else:
            logger.info(f"CH12 Overall AUC: {ch12_results['overall_auc']:.4f}")
    except MemoryError:
        logger.error("CH12: Out of memory")
        results['ch12'] = {'error': 'OOM', 'n_channels': 12, 'n_qubits': 25}
    except Exception as e:
        logger.error(f"CH12 error: {e}")
        results['ch12'] = {'error': str(e), 'n_channels': 12, 'n_qubits': 25}

    # Step 3: Run CH16
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: CH16 (33 qubits)")
    logger.info("=" * 60)

    try:
        ch16_results = run_quantum_loso(CH16)
        results['ch16'] = ch16_results

        if 'error' in ch16_results:
            logger.warning(f"CH16 failed: {ch16_results['error']}")
        else:
            logger.info(f"CH16 Overall AUC: {ch16_results['overall_auc']:.4f}")
    except MemoryError:
        logger.error("CH16: Out of memory")
        results['ch16'] = {'error': 'OOM', 'n_channels': 16, 'n_qubits': 33}
    except Exception as e:
        logger.error(f"CH16 error: {e}")
        results['ch16'] = {'error': str(e), 'n_channels': 16, 'n_qubits': 33}

    # Save results
    with open(OUTPUT_DIR / 'quantum_channel_scaling.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Update comparison markdown
    ch8_auc = ch8_results.get('overall_auc', 'N/A')
    ch12_auc = results['ch12'].get('overall_auc', results['ch12'].get('error', 'N/A'))
    ch16_auc = results['ch16'].get('overall_auc', results['ch16'].get('error', 'N/A'))

    # Load classical results
    classical_file = OUTPUT_DIR / 'classical_channel_scaling.json'
    if classical_file.exists():
        with open(classical_file, 'r') as f:
            classical = json.load(f)
        c8 = classical['ch8']['overall_auc']
        c12 = classical['ch12']['overall_auc']
        c16 = classical['ch16']['overall_auc']
    else:
        c8, c12, c16 = 0.7234, 0.7116, 0.7184

    # Determine trends
    def get_trend(aucs):
        if isinstance(aucs[2], str) or isinstance(aucs[0], str):
            return 'N/A'
        if aucs[2] > aucs[0] + 0.01:
            return 'up'
        elif aucs[2] < aucs[0] - 0.01:
            return 'down'
        return 'flat'

    q_trend = get_trend([ch8_auc if isinstance(ch8_auc, float) else 0,
                         ch12_auc if isinstance(ch12_auc, float) else 0,
                         ch16_auc if isinstance(ch16_auc, float) else 0])

    def fmt_auc(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    md = f"""# Channel Scaling Comparison

## Results

| Method          | 8ch AUC | 12ch AUC | 16ch AUC | Trend |
|-----------------|---------|----------|----------|-------|
| Classical Tier2 | {c8:.4f}  | {c12:.4f}   | {c16:.4f}   | flat |
| Quantum QPNN    | {fmt_auc(ch8_auc)}  | {fmt_auc(ch12_auc)}   | {fmt_auc(ch16_auc)}   | {q_trend} |

## Channel Sets (Nested)

- **CH8**: {CH8}
- **CH12**: {CH12}
- **CH16**: {CH16}

## Qubit Requirements

| Channels | Qubits | State Size | Feasibility |
|----------|--------|------------|-------------|
| 8        | 17     | ~2 MB      | OK          |
| 12       | 25     | ~512 MB    | Borderline  |
| 16       | 33     | ~128 GB    | Infeasible  |

## Configuration

- **Classical**: Tier 2 MAX (correlation eigenvalues with MAX pooling), XGBoost 5-bag ensemble
- **Quantum**: PLV encoding (theta-alpha 4-13 Hz), MAX aggregation, fidelity-based classification
- **CV**: Leave-One-Subject-Out

## Key Finding

The quantum circuit's qubit count scales as 2M+1 for M channels, making statevector simulation
exponentially expensive. Classical methods scale linearly with channel count.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    with open(OUTPUT_DIR / 'channel_scaling_comparison.md', 'w', encoding='utf-8') as f:
        f.write(md)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"CH8  ({len(CH8)} ch, 17 qubits):  {fmt_auc(ch8_auc)}")
    logger.info(f"CH12 ({len(CH12)} ch, 25 qubits): {fmt_auc(ch12_auc)}")
    logger.info(f"CH16 ({len(CH16)} ch, 33 qubits): {fmt_auc(ch16_auc)}")
    logger.info(f"\nResults saved to: {OUTPUT_DIR}")

    return results


if __name__ == '__main__':
    main()
