#!/usr/bin/env python3
"""
================================================================================
PROMPT 008 — Simulation Scaling Curve: V2 band_power at 4/8/12/16 Channels
================================================================================

Establishes the ideal quantum performance curve using V2 band_power encoding.
Statevector simulation at 4, 8, 12, and 16 channels under LOSO CV.

This is the theoretical ceiling — what quantum would achieve on perfect hardware.

Memory scaling: O(2^(2M+1)) complex numbers
- CH4  =  9 qubits = 2^9  = 512 complex = ~8 KB
- CH8  = 17 qubits = 2^17 = 131K complex = ~2 MB
- CH12 = 25 qubits = 2^25 = 33M complex = ~512 MB
- CH16 = 33 qubits = 2^33 = 8.6B complex = ~128 GB

================================================================================
"""

import os
import sys
import json
import time
import psutil
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from scipy.signal import welch
import logging

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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
OUTPUT_DIR = Path("results/projection")
CLASSICAL_SCALING = Path("results/scaling/classical_channel_scaling.json")

# Priority subjects for testing (from PROMPT 006)
PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']

# Timeout per subject (seconds) - to detect infeasibility
TIMEOUT_PER_SUBJECT = 600  # 10 minutes


# =============================================================================
# V2 BAND-POWER ENCODING (exact copy from run_quantum_loso_v2.py)
# =============================================================================

def extract_pn_params_bandpower(eeg_segment: np.ndarray, fs: float = 256.0) -> List[Tuple[float, float, float]]:
    """
    Extract (a, b, c) per channel using band power ratios.

    This is the V2 band_power encoding that produced AUC 0.534 in simulation.
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
            params_list.append((0.5, np.pi, 0.5))
            continue

        rel = {name: val / total for name, val in power.items()}

        # a = excitatory (high frequency: beta + gamma)
        a = np.clip(rel['beta'] + rel['gamma'], 0.05, 0.95)

        # c = inhibitory (low frequency: delta + theta)
        c = np.clip(rel['delta'] + rel['theta'], 0.05, 0.95)

        # b = phase (log ratio mapped to [0, 2*pi])
        high = power['beta'] + power['gamma']
        low = power['delta'] + power['theta']
        ratio = high / (low + 1e-10)
        log_ratio = np.log10(ratio + 1e-10)
        b_01 = 1.0 / (1.0 + np.exp(-2.0 * log_ratio))
        b = b_01 * 2 * np.pi

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


def load_segments_for_channels(channels: List[str], subjects: List[str] = None):
    """Load EEG segments for specified channels and subjects."""
    all_subjects = sorted([d for d in DATA_DIR.iterdir()
                           if d.is_dir() and d.name.startswith('chb')])
    subjects_dirs = [d for d in all_subjects if d.name not in EXCLUDE_SUBJECTS]

    if subjects:
        subjects_dirs = [d for d in subjects_dirs if d.name in subjects]

    eeg_list = []
    labels = []
    subject_ids = []

    for subject_dir in subjects_dirs:
        segs = _extract_segments_for_subject(subject_dir)

        for seg in segs:
            eeg = load_eeg_segment(seg, subject_dir, channels)
            if eeg is not None:
                eeg_list.append(eeg)
                labels.append(seg.label)
                subject_ids.append(seg.subject)

    return eeg_list, labels, subject_ids


# =============================================================================
# QUANTUM SIMULATION (V2 DUAL TEMPLATE)
# =============================================================================

def build_template(eeg_segments: List[np.ndarray], fs: float = 256.0):
    """Build averaged template circuit from multiple EEG segments."""
    if not eeg_segments:
        return None, None

    all_params = []
    for eeg in eeg_segments:
        params = extract_pn_params_bandpower(eeg, fs)
        all_params.append(params)

    n_channels = len(all_params[0])
    avg_params = []

    for ch in range(n_channels):
        avg_a = np.mean([p[ch][0] for p in all_params])
        avg_c = np.mean([p[ch][2] for p in all_params])
        sins = np.mean([np.sin(p[ch][1]) for p in all_params])
        coss = np.mean([np.cos(p[ch][1]) for p in all_params])
        avg_b = np.arctan2(sins, coss) % (2 * np.pi)
        avg_params.append((avg_a, avg_b, avg_c))

    circuit = create_multichannel_circuit(avg_params)
    sv = get_statevector(circuit)

    return circuit, sv


def compute_fidelity(sv1, sv2) -> float:
    """Compute fidelity |<psi1|psi2>|^2"""
    return abs(sv1.inner(sv2)) ** 2


def run_quantum_loso_v2(eeg_list: List[np.ndarray], labels: List[str],
                         subjects: List[str], n_channels: int) -> Dict:
    """
    Run LOSO CV with V2 band_power dual-template quantum classification.
    Returns per-subject and overall AUC.
    """
    from sklearn.metrics import roc_auc_score

    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    y = np.array([label_map.get(l, 0) for l in labels])

    unique_subjects = sorted(set(subjects))

    results = {'per_subject': {}}
    all_scores = []
    all_true = []

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
        ictal_segs = [eeg for eeg, lbl in zip(train_eeg, train_labels)
                      if lbl in ('ictal', 'preictal')]
        inter_segs = [eeg for eeg, lbl in zip(train_eeg, train_labels)
                      if lbl == 'interictal']

        if not ictal_segs or not inter_segs:
            logger.info(f"  {test_subject}: skipped (missing template class)")
            continue

        _, ictal_sv = build_template(ictal_segs, fs=FS)
        _, inter_sv = build_template(inter_segs, fs=FS)

        # Classify test segments
        scores = []
        for eeg in test_eeg:
            test_params = extract_pn_params_bandpower(eeg, fs=FS)
            test_circuit = create_multichannel_circuit(test_params)
            test_sv = get_statevector(test_circuit)

            fid_ictal = compute_fidelity(ictal_sv, test_sv)
            fid_inter = compute_fidelity(inter_sv, test_sv)
            scores.append(fid_ictal - fid_inter)

        scores = np.array(scores)

        try:
            auc = roc_auc_score(y_test, scores)
        except ValueError:
            auc = 0.5

        results['per_subject'][test_subject] = {
            'auc': float(auc),
            'n_samples': int(len(y_test)),
            'n_seizure': int(y_test.sum())
        }

        all_scores.extend(scores.tolist())
        all_true.extend(y_test.tolist())

        logger.info(f"  {test_subject}: AUC={auc:.4f} (n={len(y_test)})")

    # Overall AUC
    if all_scores:
        try:
            results['overall_auc'] = float(roc_auc_score(all_true, all_scores))
        except ValueError:
            results['overall_auc'] = 0.5
    else:
        results['overall_auc'] = None

    return results


def estimate_memory_gb(n_qubits: int) -> float:
    """Estimate memory required for statevector simulation in GB."""
    # 2^n complex128 numbers = 2^n * 16 bytes
    return (2 ** n_qubits * 16) / (1024 ** 3)


def check_feasibility(n_channels: int) -> Tuple[bool, str]:
    """Check if channel count is feasible for statevector simulation."""
    n_qubits = 2 * n_channels + 1
    mem_gb = estimate_memory_gb(n_qubits)
    available_gb = psutil.virtual_memory().available / (1024 ** 3)

    if mem_gb > available_gb * 0.8:  # Leave 20% margin
        return False, f"state_vector_size_gb: {mem_gb:.2f}, available_gb: {available_gb:.2f}"
    return True, ""


def run_channel_scaling(channels_dict: Dict) -> Dict:
    """Run V2 band_power simulation at all channel counts."""
    results = {
        'encoding': 'V2_band_power',
        'backend': 'statevector_simulation',
        'timestamp': datetime.now().isoformat()
    }

    for ch_key in ['ch4', 'ch8', 'ch12', 'ch16']:
        if ch_key not in channels_dict:
            # Derive CH4 from CH8 if missing
            if ch_key == 'ch4' and 'ch8' in channels_dict:
                channels = channels_dict['ch8']['channels'][:4]
            else:
                logger.warning(f"{ch_key} not found in channel config, skipping")
                continue
        else:
            channels = channels_dict[ch_key]['channels']

        n_channels = len(channels)
        n_qubits = 2 * n_channels + 1

        logger.info(f"\n{'='*60}")
        logger.info(f"Running {ch_key.upper()}: {n_channels} channels, {n_qubits} qubits")
        logger.info(f"Channels: {channels}")
        logger.info(f"{'='*60}")

        # Check feasibility
        feasible, failure_reason = check_feasibility(n_channels)
        mem_gb = estimate_memory_gb(n_qubits)

        if not feasible:
            logger.warning(f"{ch_key}: INFEASIBLE - {failure_reason}")
            results[ch_key] = {
                'n_qubits': n_qubits,
                'channels': channels,
                'overall_auc': None,
                'per_subject': {},
                'feasible': False,
                'failure_reason': failure_reason,
                'estimated_memory_gb': mem_gb
            }
            continue

        logger.info(f"Estimated memory: {mem_gb:.2f} GB")

        # Load data
        logger.info("Loading EEG segments...")
        start_time = time.time()

        try:
            # For CH12/CH16, use priority subjects only to speed up
            if n_channels > 8:
                logger.info(f"Using priority subjects only for {ch_key}")
                eeg_list, labels, subject_ids = load_segments_for_channels(
                    channels, subjects=PRIORITY_SUBJECTS
                )
            else:
                eeg_list, labels, subject_ids = load_segments_for_channels(channels)

            logger.info(f"Loaded {len(eeg_list)} segments from {len(set(subject_ids))} subjects")

            if len(eeg_list) == 0:
                logger.warning(f"{ch_key}: No segments loaded")
                results[ch_key] = {
                    'n_qubits': n_qubits,
                    'channels': channels,
                    'overall_auc': None,
                    'per_subject': {},
                    'feasible': False,
                    'failure_reason': 'no_segments_loaded',
                    'estimated_memory_gb': mem_gb
                }
                continue

            # Run quantum LOSO
            logger.info("Running V2 band_power LOSO CV...")
            quantum_results = run_quantum_loso_v2(eeg_list, labels, subject_ids, n_channels)

            elapsed = time.time() - start_time

            results[ch_key] = {
                'n_qubits': n_qubits,
                'channels': channels,
                'overall_auc': quantum_results['overall_auc'],
                'per_subject': quantum_results['per_subject'],
                'feasible': True,
                'runtime_sec': elapsed,
                'estimated_memory_gb': mem_gb
            }

            logger.info(f"{ch_key}: Overall AUC = {quantum_results['overall_auc']:.4f} ({elapsed:.1f}s)")

        except MemoryError as e:
            logger.error(f"{ch_key}: MemoryError - {e}")
            results[ch_key] = {
                'n_qubits': n_qubits,
                'channels': channels,
                'overall_auc': None,
                'per_subject': {},
                'feasible': False,
                'failure_reason': f'MemoryError: {str(e)}',
                'estimated_memory_gb': mem_gb
            }
        except Exception as e:
            logger.error(f"{ch_key}: Error - {e}")
            results[ch_key] = {
                'n_qubits': n_qubits,
                'channels': channels,
                'overall_auc': None,
                'per_subject': {},
                'feasible': False,
                'failure_reason': str(e),
                'estimated_memory_gb': mem_gb
            }

    return results


def main():
    logger.info("=" * 70)
    logger.info("PROMPT 008: Simulation Scaling Curve - V2 band_power")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load channel sets from classical scaling
    logger.info(f"\nLoading channel sets from {CLASSICAL_SCALING}")
    with open(CLASSICAL_SCALING) as f:
        classical_data = json.load(f)

    # Build channel dictionary
    channels_dict = {}

    # CH4: derive from CH8 (first 4 channels)
    if 'ch8' in classical_data:
        ch8_channels = classical_data['ch8']['channels']
        channels_dict['ch4'] = {'channels': ch8_channels[:4]}
        logger.info(f"CH4 derived from CH8: {channels_dict['ch4']['channels']}")

    # CH8, CH12, CH16 from classical scaling
    for key in ['ch8', 'ch12', 'ch16']:
        if key in classical_data:
            channels_dict[key] = {'channels': classical_data[key]['channels']}
            logger.info(f"{key.upper()}: {channels_dict[key]['channels']}")

    # Run scaling experiment
    results = run_channel_scaling(channels_dict)

    # Save results
    output_file = OUTPUT_DIR / 'simulation_scaling.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SIMULATION SCALING SUMMARY (V2 band_power)")
    print("=" * 70)
    print(f"{'Channels':<10} {'Qubits':<10} {'Feasible':<10} {'AUC':<10} {'Memory (GB)':<12}")
    print("-" * 70)

    for ch_key in ['ch4', 'ch8', 'ch12', 'ch16']:
        if ch_key in results:
            r = results[ch_key]
            n_ch = len(r.get('channels', []))
            feasible = 'Yes' if r.get('feasible', False) else 'No'
            auc = f"{r['overall_auc']:.4f}" if r.get('overall_auc') else 'N/A'
            mem = f"{r.get('estimated_memory_gb', 0):.2f}"
            print(f"{n_ch:<10} {r.get('n_qubits', 'N/A'):<10} {feasible:<10} {auc:<10} {mem:<12}")

    print("=" * 70)

    logger.info(f"\nCompleted: {datetime.now()}")


if __name__ == '__main__':
    main()
