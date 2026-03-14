#!/usr/bin/env python3
"""
================================================================================
PROMPT 019 - Investigate Whether `b` (Mean Phase) is Noise in the A-Gate Encoding
================================================================================

Hypothesis: `b` (mean instantaneous phase over a 20-second window) carries no
state-discriminative signal and acts as per-window noise that adds variance to
Z without contributing to ictal/interictal separation.

Tasks:
1. Distribution Analysis of `b` by State
2. Statevector Simulation - Fixed `b` vs Computed `b`
3. Sensitivity Analysis - Perturbation of `b`
4. Per-Channel `b` Variance Analysis

Output:
- results/encoding_analysis/b_parameter_investigation.json
- results/encoding_analysis/b_parameter_report.md
- results/encoding_analysis/b_distributions.png
- results/encoding_analysis/b_sweep.png

Author: Claude Code
Date: 2026-03-08
================================================================================
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import circmean, circvar, mannwhitneyu
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

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

from QA1.multichannel_circuit import create_multichannel_circuit

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "encoding_analysis"
DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")

# Classical scaling results (for channel list)
CLASSICAL_RESULTS = PROJECT_ROOT / "results" / "scaling" / "classical_channel_scaling.json"

# PLV encoding parameters - SAME as PROMPT 006
BAND = (4, 13)  # theta-alpha

# 20-second window
WINDOW_SECONDS = 20.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * FS)  # 5120 samples at 256 Hz

# Priority subjects
PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']

# B sweep values for Task 3
B_SWEEP_VALUES = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]


# =============================================================================
# PLV ENCODING - IDENTICAL TO PROMPT 006/017
# =============================================================================

def extract_plv_params(eeg_segment: np.ndarray, fs: float = 256.0,
                       band: Tuple[float, float] = (4, 13)) -> List[Tuple[float, float, float]]:
    """
    Extract (a, b, c) per channel using Hilbert phase/amplitude.

    IDENTICAL implementation to PROMPT 006 (quantum_heron_validation.py).
    """
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


# =============================================================================
# DATA LOADING
# =============================================================================

def load_channels() -> List[str]:
    """Load CH8 channel list from classical scaling results."""
    if CLASSICAL_RESULTS.exists():
        with open(CLASSICAL_RESULTS, 'r') as f:
            data = json.load(f)
        return data['ch8']['channels']
    else:
        return ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3", "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]


def load_eeg_segment(seg: 'EEGSegment', subject_dir: Path, channels: List[str]) -> Optional[np.ndarray]:
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
            n_samples = int(WINDOW_SEC * fs)  # Full 30s segment

            data = np.zeros((len(channels), n_samples))
            for i, ch_idx in enumerate(channel_indices):
                signal = f.readSignal(ch_idx)
                end_sample = min(start_sample + n_samples, len(signal))
                actual_samples = end_sample - start_sample
                if actual_samples < n_samples:
                    return None
                data[i] = signal[start_sample:end_sample]

            return data

    except Exception:
        return None


def load_subject_data(subject_id: str, channels: List[str]) -> Tuple[List[np.ndarray], List[str]]:
    """Load all segments for a subject."""
    subject_dir = DATA_DIR / subject_id
    if not subject_dir.exists():
        return [], []

    segs = _extract_segments_for_subject(subject_dir)

    eeg_list = []
    labels = []

    for seg in segs:
        eeg = load_eeg_segment(seg, subject_dir, channels)
        if eeg is not None:
            eeg_list.append(eeg)
            labels.append(seg.label)

    return eeg_list, labels


# =============================================================================
# STATEVECTOR SIMULATION
# =============================================================================

def run_statevector_simulation(circuit) -> Dict[str, float]:
    """Run circuit on statevector simulator and return state probabilities."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    simulator = AerSimulator(method='statevector')

    # Need to save statevector before measurement
    circuit_no_meas = circuit.remove_final_measurements(inplace=False)
    circuit_no_meas.save_statevector()

    transpiled = transpile(circuit_no_meas, simulator)
    result = simulator.run(transpiled, shots=1).result()
    statevector = result.get_statevector()

    # Convert to probabilities (squared amplitudes)
    probs = {}
    n_qubits = circuit_no_meas.num_qubits
    for i, amp in enumerate(statevector.data):
        if abs(amp) > 1e-10:
            bitstring = format(i, f'0{n_qubits}b')
            probs[bitstring] = abs(amp) ** 2

    return probs


def statevector_fidelity(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute fidelity between two probability distributions."""
    all_keys = set(p.keys()) | set(q.keys())

    sqrt_sum = 0.0
    for key in all_keys:
        p_prob = p.get(key, 0)
        q_prob = q.get(key, 0)
        sqrt_sum += np.sqrt(p_prob * q_prob)

    return sqrt_sum ** 2


def create_circuit_with_fixed_b(params_list: List[Tuple[float, float, float]],
                                 fixed_b: float) -> 'QuantumCircuit':
    """Create circuit with b parameter replaced by a fixed value."""
    modified_params = [(a, fixed_b, c) for (a, _, c) in params_list]
    return create_multichannel_circuit(modified_params)


# =============================================================================
# TASK 1: Distribution Analysis of `b` by State
# =============================================================================

def task1_distribution_analysis(channels: List[str]) -> Dict:
    """
    Analyze the distribution of `b` parameter by seizure state.

    Computes:
    - Circular mean and variance per state
    - Mann-Whitney U test between states
    - AUC of `b` alone as a classifier
    """
    logger.info("\n" + "=" * 60)
    logger.info("TASK 1: Distribution Analysis of `b` by State")
    logger.info("=" * 60)

    results = {}
    all_b_ictal = []
    all_b_interictal = []

    for subject_id in PRIORITY_SUBJECTS:
        logger.info(f"\nProcessing {subject_id}...")

        eeg_list, labels = load_subject_data(subject_id, channels)
        if not eeg_list:
            logger.warning(f"  No data found for {subject_id}")
            continue

        # Extract b values for each window
        b_ictal = []
        b_interictal = []

        for eeg, label in zip(eeg_list, labels):
            eeg_20s = eeg[:, :WINDOW_SAMPLES]
            if eeg_20s.shape[1] < WINDOW_SAMPLES:
                continue

            params = extract_plv_params(eeg_20s, fs=FS, band=BAND)

            # Average b across channels (circular mean)
            b_values = [p[1] for p in params]
            sins = np.mean([np.sin(b) for b in b_values])
            coss = np.mean([np.cos(b) for b in b_values])
            avg_b = np.arctan2(sins, coss) % (2 * np.pi)

            if label in ('ictal', 'preictal'):
                b_ictal.append(avg_b)
            else:
                b_interictal.append(avg_b)

        if not b_ictal or not b_interictal:
            logger.warning(f"  Missing class for {subject_id}")
            continue

        # Store for pooled analysis
        all_b_ictal.extend(b_ictal)
        all_b_interictal.extend(b_interictal)

        # Circular statistics
        ictal_circmean = float(circmean(b_ictal, high=2*np.pi, low=0))
        ictal_circvar = float(circvar(b_ictal, high=2*np.pi, low=0))
        interictal_circmean = float(circmean(b_interictal, high=2*np.pi, low=0))
        interictal_circvar = float(circvar(b_interictal, high=2*np.pi, low=0))

        # Mann-Whitney U test (on raw values - works for circular data)
        try:
            stat, p_value = mannwhitneyu(b_ictal, b_interictal, alternative='two-sided')
            mannwhitney_p = float(p_value)
        except Exception:
            mannwhitney_p = 1.0

        # AUC of b alone as classifier
        try:
            y_true = [1] * len(b_ictal) + [0] * len(b_interictal)
            y_scores = b_ictal + b_interictal
            auc = roc_auc_score(y_true, y_scores)
            # Handle inverted polarity
            b_alone_auc = float(max(auc, 1.0 - auc))
        except Exception:
            b_alone_auc = 0.5

        results[subject_id] = {
            'ictal_circmean': ictal_circmean,
            'ictal_circvar': ictal_circvar,
            'interictal_circmean': interictal_circmean,
            'interictal_circvar': interictal_circvar,
            'mannwhitney_p': mannwhitney_p,
            'b_alone_auc': b_alone_auc,
            'n_ictal': len(b_ictal),
            'n_interictal': len(b_interictal)
        }

        logger.info(f"  Ictal: mean={ictal_circmean:.3f}, var={ictal_circvar:.3f}")
        logger.info(f"  Interictal: mean={interictal_circmean:.3f}, var={interictal_circvar:.3f}")
        logger.info(f"  Mann-Whitney p={mannwhitney_p:.4f}, b-alone AUC={b_alone_auc:.4f}")

    # Pooled statistics
    if all_b_ictal and all_b_interictal:
        try:
            y_true_pooled = [1] * len(all_b_ictal) + [0] * len(all_b_interictal)
            y_scores_pooled = all_b_ictal + all_b_interictal
            auc_pooled = roc_auc_score(y_true_pooled, y_scores_pooled)
            pooled_auc = float(max(auc_pooled, 1.0 - auc_pooled))
        except Exception:
            pooled_auc = 0.5

        results['pooled'] = {
            'ictal_circmean': float(circmean(all_b_ictal, high=2*np.pi, low=0)),
            'ictal_circvar': float(circvar(all_b_ictal, high=2*np.pi, low=0)),
            'interictal_circmean': float(circmean(all_b_interictal, high=2*np.pi, low=0)),
            'interictal_circvar': float(circvar(all_b_interictal, high=2*np.pi, low=0)),
            'b_alone_auc': pooled_auc,
            'n_ictal': len(all_b_ictal),
            'n_interictal': len(all_b_interictal)
        }

        logger.info(f"\nPooled: b-alone AUC = {pooled_auc:.4f}")

    return {
        'per_subject': results,
        'b_ictal_all': all_b_ictal,
        'b_interictal_all': all_b_interictal
    }


def plot_b_distributions(b_ictal: List[float], b_interictal: List[float],
                         per_subject: Dict, output_path: Path):
    """Create circular histogram of b distributions."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), subplot_kw=dict(projection='polar'))
    axes = axes.flatten()

    # Plot per-subject
    subjects = [s for s in PRIORITY_SUBJECTS if s in per_subject]

    for idx, subject_id in enumerate(subjects):
        if idx >= 7:
            break
        ax = axes[idx]

        # Get subject-specific data (we need to reload)
        # For simplicity, just show the circular mean and variance
        stats = per_subject[subject_id]

        # Draw arrows for circular means
        ax.arrow(stats['ictal_circmean'], 0, 0, 0.8,
                head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.7)
        ax.arrow(stats['interictal_circmean'], 0, 0, 0.8,
                head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.7)

        ax.set_title(f"{subject_id}\nb-AUC={stats['b_alone_auc']:.3f}")
        ax.set_ylim(0, 1)

    # Pooled distribution in last panel
    ax = axes[7]
    bins = np.linspace(0, 2*np.pi, 17)

    ax.hist(b_ictal, bins=bins, alpha=0.5, color='red', label='Ictal')
    ax.hist(b_interictal, bins=bins, alpha=0.5, color='blue', label='Interictal')
    ax.set_title('Pooled Distribution')
    ax.legend(loc='upper right')

    plt.suptitle('Distribution of b (Mean Phase) by State\nArrows show circular mean direction',
                 fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved b distributions plot to {output_path}")


# =============================================================================
# TASK 2: Statevector Simulation - Fixed `b` vs Computed `b`
# =============================================================================

def task2_simulation_comparison(channels: List[str]) -> Dict:
    """
    Run statevector simulation comparing:
    - Condition A: Computed `b` (current encoding)
    - Condition B: Fixed `b` = pi
    - Condition C: Fixed `b` = 0
    """
    logger.info("\n" + "=" * 60)
    logger.info("TASK 2: Statevector Simulation - Fixed b vs Computed b")
    logger.info("=" * 60)

    n_channels = len(channels)
    n_qubits = 2 * n_channels + 1

    logger.info(f"Channels: {n_channels}, Qubits: {n_qubits}")

    # Load all subject data
    subject_data = {}
    for subj in PRIORITY_SUBJECTS:
        eeg_list, labels = load_subject_data(subj, channels)
        if len(eeg_list) > 0:
            label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
            y = np.array([label_map.get(l, 0) for l in labels])
            if len(np.unique(y)) == 2:
                subject_data[subj] = (eeg_list, labels, y)
                logger.info(f"  {subj}: {len(eeg_list)} segments, {int(sum(y))} seizure")

    if not subject_data:
        return {'error': 'No valid subjects found'}

    # Run LOSO CV for each condition
    conditions = {
        'condition_a_computed_b': None,  # Use computed b
        'condition_b_fixed_pi': np.pi,   # Fixed b = pi
        'condition_c_fixed_zero': 0.0    # Fixed b = 0
    }

    results = {}

    for condition_name, fixed_b_value in conditions.items():
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {condition_name}")
        logger.info(f"{'='*50}")

        per_subject = {}
        all_y_true = []
        all_scores = []

        for test_subject in subject_data.keys():
            logger.info(f"\n  Test subject: {test_subject}")

            # Training data - extract 20s windows
            train_ictal = []
            train_inter = []
            for subj, (eeg_list, labels, _) in subject_data.items():
                if subj != test_subject:
                    for eeg, lbl in zip(eeg_list, labels):
                        eeg_20s = eeg[:, :WINDOW_SAMPLES]
                        if eeg_20s.shape[1] >= WINDOW_SAMPLES:
                            if lbl in ('ictal', 'preictal'):
                                train_ictal.append(eeg_20s)
                            else:
                                train_inter.append(eeg_20s)

            if not train_ictal or not train_inter:
                logger.warning(f"    Skipped: missing template class")
                continue

            # Extract PLV params from 20s windows
            ictal_params_all = []
            for eeg in train_ictal[:50]:
                params = extract_plv_params(eeg, fs=FS, band=BAND)
                ictal_params_all.append(params)

            inter_params_all = []
            for eeg in train_inter[:50]:
                params = extract_plv_params(eeg, fs=FS, band=BAND)
                inter_params_all.append(params)

            # Average template params
            def average_params(params_list):
                avg = []
                for ch in range(n_channels):
                    avg_a = np.mean([p[ch][0] for p in params_list])
                    sins = np.mean([np.sin(p[ch][1]) for p in params_list])
                    coss = np.mean([np.cos(p[ch][1]) for p in params_list])
                    avg_b = np.arctan2(sins, coss) % (2 * np.pi)
                    avg_c = np.clip(np.mean([p[ch][2] for p in params_list]), 0.05, 0.95)
                    avg.append((avg_a, avg_b, avg_c))
                return avg

            ictal_avg = average_params(ictal_params_all)
            inter_avg = average_params(inter_params_all)

            # Create template circuits (with or without fixed b)
            if fixed_b_value is not None:
                ictal_circuit = create_circuit_with_fixed_b(ictal_avg, fixed_b_value)
                inter_circuit = create_circuit_with_fixed_b(inter_avg, fixed_b_value)
            else:
                ictal_circuit = create_multichannel_circuit(ictal_avg)
                inter_circuit = create_multichannel_circuit(inter_avg)

            ictal_template_probs = run_statevector_simulation(ictal_circuit)
            inter_template_probs = run_statevector_simulation(inter_circuit)

            # Test data
            test_eeg, test_labels, y_test = subject_data[test_subject]

            scores = []
            valid_indices = []

            for idx, eeg in enumerate(test_eeg):
                eeg_20s = eeg[:, :WINDOW_SAMPLES]
                if eeg_20s.shape[1] >= WINDOW_SAMPLES:
                    params = extract_plv_params(eeg_20s, fs=FS, band=BAND)

                    if fixed_b_value is not None:
                        test_circuit = create_circuit_with_fixed_b(params, fixed_b_value)
                    else:
                        test_circuit = create_multichannel_circuit(params)

                    test_probs = run_statevector_simulation(test_circuit)

                    fid_ictal = statevector_fidelity(test_probs, ictal_template_probs)
                    fid_inter = statevector_fidelity(test_probs, inter_template_probs)
                    score = fid_ictal - fid_inter
                    scores.append(score)
                    valid_indices.append(idx)

            y_test_valid = y_test[valid_indices]
            scores = np.array(scores)

            # Compute AUC
            try:
                auc = roc_auc_score(y_test_valid, scores)
            except:
                auc = 0.5

            # Determine polarity
            calibrated_auc = max(auc, 1.0 - auc)
            polarity = 'inverted' if auc < 0.5 else 'standard'

            per_subject[test_subject] = {
                'raw_auc': float(auc),
                'calibrated_auc': float(calibrated_auc),
                'polarity': polarity,
                'n_samples': len(y_test_valid),
                'n_seizure': int(sum(y_test_valid))
            }

            all_y_true.extend(y_test_valid.tolist())
            all_scores.extend(scores.tolist())

            logger.info(f"    raw AUC = {auc:.4f}, calibrated = {calibrated_auc:.4f}")

        # Overall AUCs
        try:
            overall_raw_auc = roc_auc_score(all_y_true, all_scores)
        except:
            overall_raw_auc = 0.5

        total_samples = sum(ps['n_samples'] for ps in per_subject.values())
        overall_calibrated_auc = sum(
            ps['calibrated_auc'] * ps['n_samples'] for ps in per_subject.values()
        ) / total_samples if total_samples > 0 else 0.5

        results[condition_name] = {
            'mean_raw_auc': float(overall_raw_auc),
            'mean_calibrated_auc': float(overall_calibrated_auc),
            'per_subject': per_subject
        }

        logger.info(f"\n  {condition_name}: mean raw AUC = {overall_raw_auc:.4f}, "
                   f"mean calibrated = {overall_calibrated_auc:.4f}")

    return results


# =============================================================================
# TASK 3: Sensitivity Analysis - Perturbation of `b`
# =============================================================================

def task3_sensitivity_analysis(channels: List[str], subject_id: str = 'chb01') -> Dict:
    """
    Run simulation across a sweep of fixed `b` values for a single subject.
    """
    logger.info("\n" + "=" * 60)
    logger.info(f"TASK 3: Sensitivity Analysis - b Sweep for {subject_id}")
    logger.info("=" * 60)

    n_channels = len(channels)

    # Load subject data
    eeg_list, labels = load_subject_data(subject_id, channels)
    if not eeg_list:
        return {'error': f'No data found for {subject_id}'}

    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    y = np.array([label_map.get(l, 0) for l in labels])

    # Load other subjects for training
    train_data = {}
    for subj in PRIORITY_SUBJECTS:
        if subj != subject_id:
            eeg_list_train, labels_train = load_subject_data(subj, channels)
            if eeg_list_train:
                y_train = np.array([label_map.get(l, 0) for l in labels_train])
                if len(np.unique(y_train)) == 2:
                    train_data[subj] = (eeg_list_train, labels_train, y_train)

    # Build training templates
    train_ictal = []
    train_inter = []
    for subj, (eeg_l, labels_l, _) in train_data.items():
        for eeg, lbl in zip(eeg_l, labels_l):
            eeg_20s = eeg[:, :WINDOW_SAMPLES]
            if eeg_20s.shape[1] >= WINDOW_SAMPLES:
                if lbl in ('ictal', 'preictal'):
                    train_ictal.append(eeg_20s)
                else:
                    train_inter.append(eeg_20s)

    if not train_ictal or not train_inter:
        return {'error': 'Missing training data'}

    # Extract PLV params
    ictal_params_all = [extract_plv_params(eeg, fs=FS, band=BAND) for eeg in train_ictal[:50]]
    inter_params_all = [extract_plv_params(eeg, fs=FS, band=BAND) for eeg in train_inter[:50]]

    def average_params(params_list):
        avg = []
        for ch in range(n_channels):
            avg_a = np.mean([p[ch][0] for p in params_list])
            sins = np.mean([np.sin(p[ch][1]) for p in params_list])
            coss = np.mean([np.cos(p[ch][1]) for p in params_list])
            avg_b = np.arctan2(sins, coss) % (2 * np.pi)
            avg_c = np.clip(np.mean([p[ch][2] for p in params_list]), 0.05, 0.95)
            avg.append((avg_a, avg_b, avg_c))
        return avg

    ictal_avg = average_params(ictal_params_all)
    inter_avg = average_params(inter_params_all)

    # Extract test params
    test_params_list = []
    test_indices = []
    for idx, eeg in enumerate(eeg_list):
        eeg_20s = eeg[:, :WINDOW_SAMPLES]
        if eeg_20s.shape[1] >= WINDOW_SAMPLES:
            params = extract_plv_params(eeg_20s, fs=FS, band=BAND)
            test_params_list.append(params)
            test_indices.append(idx)

    y_test = y[test_indices]

    # Sweep b values
    calibrated_aucs = []
    raw_aucs = []

    for fixed_b in B_SWEEP_VALUES:
        logger.info(f"  Testing b = {fixed_b:.4f} ({fixed_b/np.pi:.2f}*pi)")

        # Create templates
        ictal_circuit = create_circuit_with_fixed_b(ictal_avg, fixed_b)
        inter_circuit = create_circuit_with_fixed_b(inter_avg, fixed_b)

        ictal_template_probs = run_statevector_simulation(ictal_circuit)
        inter_template_probs = run_statevector_simulation(inter_circuit)

        # Test
        scores = []
        for params in test_params_list:
            test_circuit = create_circuit_with_fixed_b(params, fixed_b)
            test_probs = run_statevector_simulation(test_circuit)

            fid_ictal = statevector_fidelity(test_probs, ictal_template_probs)
            fid_inter = statevector_fidelity(test_probs, inter_template_probs)
            scores.append(fid_ictal - fid_inter)

        try:
            auc = roc_auc_score(y_test, scores)
            calibrated_auc = max(auc, 1.0 - auc)
        except:
            auc = 0.5
            calibrated_auc = 0.5

        raw_aucs.append(float(auc))
        calibrated_aucs.append(float(calibrated_auc))

        logger.info(f"    raw AUC = {auc:.4f}, calibrated = {calibrated_auc:.4f}")

    # Analysis
    auc_std = np.std(calibrated_aucs)
    auc_range = max(calibrated_aucs) - min(calibrated_aucs)
    best_idx = np.argmax(calibrated_aucs)
    best_b = B_SWEEP_VALUES[best_idx]

    return {
        'subject': subject_id,
        'b_values': [float(b) for b in B_SWEEP_VALUES],
        'b_values_pi': [float(b / np.pi) for b in B_SWEEP_VALUES],
        'raw_aucs': raw_aucs,
        'calibrated_aucs': calibrated_aucs,
        'auc_std': float(auc_std),
        'auc_range': float(auc_range),
        'best_b': float(best_b),
        'best_b_pi': float(best_b / np.pi),
        'best_auc': float(calibrated_aucs[best_idx]),
        'is_flat': bool(auc_range < 0.02)  # Threshold for "flat"
    }


def plot_b_sweep(sweep_results: Dict, output_path: Path):
    """Plot AUC vs fixed b value."""
    fig, ax = plt.subplots(figsize=(10, 6))

    b_values_pi = sweep_results['b_values_pi']
    calibrated_aucs = sweep_results['calibrated_aucs']

    ax.plot(b_values_pi, calibrated_aucs, 'o-', markersize=10, linewidth=2)
    ax.axhline(y=0.5, color='r', linestyle='--', label='Chance level')
    ax.axhline(y=np.mean(calibrated_aucs), color='g', linestyle=':',
               label=f'Mean AUC = {np.mean(calibrated_aucs):.3f}')

    ax.set_xlabel('Fixed b value (multiples of pi)', fontsize=12)
    ax.set_ylabel('Calibrated AUC', fontsize=12)
    ax.set_title(f'Sensitivity Analysis: AUC vs Fixed b Value\n'
                f'Subject: {sweep_results["subject"]}, Range: {sweep_results["auc_range"]:.4f}',
                fontsize=14)
    ax.set_xticks(b_values_pi)
    ax.set_xticklabels([f'{v:.2f}' for v in b_values_pi])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved b sweep plot to {output_path}")


# =============================================================================
# TASK 4: Per-Channel `b` Variance Analysis
# =============================================================================

def task4_variance_analysis(channels: List[str]) -> Dict:
    """
    Compute variance of `b` per channel across states and subjects.
    """
    logger.info("\n" + "=" * 60)
    logger.info("TASK 4: Per-Channel b Variance Analysis")
    logger.info("=" * 60)

    # Collect b values per channel
    per_channel_ictal = {ch: [] for ch in channels}
    per_channel_interictal = {ch: [] for ch in channels}
    chb01_per_channel = {ch: [] for ch in channels}

    for subject_id in PRIORITY_SUBJECTS:
        logger.info(f"Processing {subject_id}...")

        eeg_list, labels = load_subject_data(subject_id, channels)

        for eeg, label in zip(eeg_list, labels):
            eeg_20s = eeg[:, :WINDOW_SAMPLES]
            if eeg_20s.shape[1] < WINDOW_SAMPLES:
                continue

            params = extract_plv_params(eeg_20s, fs=FS, band=BAND)

            for ch_idx, ch in enumerate(channels):
                b_val = params[ch_idx][1]

                if label in ('ictal', 'preictal'):
                    per_channel_ictal[ch].append(b_val)
                else:
                    per_channel_interictal[ch].append(b_val)

                if subject_id == 'chb01':
                    chb01_per_channel[ch].append(b_val)

    # Compute variances
    results = {}
    for ch in channels:
        ictal_var = float(circvar(per_channel_ictal[ch], high=2*np.pi, low=0)) if per_channel_ictal[ch] else None
        interictal_var = float(circvar(per_channel_interictal[ch], high=2*np.pi, low=0)) if per_channel_interictal[ch] else None
        chb01_var = float(circvar(chb01_per_channel[ch], high=2*np.pi, low=0)) if chb01_per_channel[ch] else None

        results[ch] = {
            'ictal_circvar': ictal_var,
            'interictal_circvar': interictal_var,
            'within_chb01_circvar': chb01_var,
            'n_ictal': len(per_channel_ictal[ch]),
            'n_interictal': len(per_channel_interictal[ch]),
            'n_chb01': len(chb01_per_channel[ch])
        }

        logger.info(f"  {ch}: ictal_var={ictal_var:.4f}, interictal_var={interictal_var:.4f}")

    # Summary statistics
    mean_ictal_var = np.mean([r['ictal_circvar'] for r in results.values() if r['ictal_circvar'] is not None])
    mean_interictal_var = np.mean([r['interictal_circvar'] for r in results.values() if r['interictal_circvar'] is not None])

    return {
        'per_channel': results,
        'summary': {
            'mean_ictal_circvar': float(mean_ictal_var),
            'mean_interictal_circvar': float(mean_interictal_var),
            'high_variance': bool(mean_ictal_var > 0.7 or mean_interictal_var > 0.7)
        }
    }


# =============================================================================
# CONCLUSION LOGIC
# =============================================================================

def determine_conclusion(task1_results: Dict, task2_results: Dict,
                         task3_results: Dict, task4_results: Dict) -> Dict:
    """
    Apply decision gates to determine conclusion and recommendation.
    """
    # Check task 1: b AUC near 0.5?
    pooled_b_auc = task1_results.get('per_subject', {}).get('pooled', {}).get('b_alone_auc', 0.5)
    b_has_no_signal = bool(abs(pooled_b_auc - 0.5) < 0.05)

    # Check task 2: fixed-pi AUC vs computed-b AUC
    computed_b_auc = task2_results.get('condition_a_computed_b', {}).get('mean_calibrated_auc', 0.5)
    fixed_pi_auc = task2_results.get('condition_b_fixed_pi', {}).get('mean_calibrated_auc', 0.5)
    fixed_zero_auc = task2_results.get('condition_c_fixed_zero', {}).get('mean_calibrated_auc', 0.5)

    pi_better_or_equal = bool(fixed_pi_auc >= computed_b_auc - 0.01)
    pi_significantly_worse = bool(fixed_pi_auc < computed_b_auc - 0.02)

    # Check task 3: AUC flat across sweep?
    sweep_flat = bool(task3_results.get('is_flat', False))

    # Check task 4: high variance?
    high_variance = bool(task4_results.get('summary', {}).get('high_variance', False))

    # Decision logic
    if b_has_no_signal and pi_better_or_equal:
        conclusion = "noise_confirmed"
        recommended_action = "Replace b with pi in all subsequent experiments"
    elif b_has_no_signal and pi_significantly_worse:
        conclusion = "ambiguous"
        recommended_action = "Investigate mechanism - b may interact constructively by accident"
    elif pooled_b_auc > 0.55:
        conclusion = "signal_present"
        recommended_action = "Keep computed b and document"
    elif sweep_flat:
        conclusion = "noise_confirmed"
        recommended_action = "b is inert; circuit is effectively 2-parameter"
    else:
        conclusion = "ambiguous"
        recommended_action = "Further investigation needed"

    return {
        'conclusion': conclusion,
        'recommended_action': recommended_action,
        'decision_factors': {
            'pooled_b_auc': pooled_b_auc,
            'b_has_no_signal': b_has_no_signal,
            'computed_b_auc': computed_b_auc,
            'fixed_pi_auc': fixed_pi_auc,
            'fixed_zero_auc': fixed_zero_auc,
            'pi_better_or_equal': pi_better_or_equal,
            'sweep_flat': sweep_flat,
            'high_variance': high_variance
        }
    }


def generate_markdown_report(results: Dict, output_path: Path):
    """Generate markdown report."""

    conclusion = results['conclusion']

    report = f"""# PROMPT 019: b Parameter Investigation Report

**Generated:** {results['timestamp']}
**Hypothesis:** `b` (mean instantaneous phase) is noise in the A-Gate encoding

## Summary

**Conclusion:** `{conclusion['conclusion']}`
**Recommended Action:** {conclusion['recommended_action']}

---

## Task 1: Distribution Analysis of `b` by State

### Per-Subject Results

| Subject | Ictal Mean | Ictal Var | Interictal Mean | Interictal Var | Mann-Whitney p | b-alone AUC |
|---------|------------|-----------|-----------------|----------------|----------------|-------------|
"""

    for subject, stats in results['task1_distribution']['per_subject'].items():
        if subject == 'pooled':
            continue
        report += f"| {subject} | {stats['ictal_circmean']:.3f} | {stats['ictal_circvar']:.3f} | "
        report += f"{stats['interictal_circmean']:.3f} | {stats['interictal_circvar']:.3f} | "
        report += f"{stats.get('mannwhitney_p', 'N/A'):.4f} | {stats['b_alone_auc']:.4f} |\n"

    pooled = results['task1_distribution']['per_subject'].get('pooled', {})
    report += f"\n**Pooled b-alone AUC:** {pooled.get('b_alone_auc', 'N/A')}\n"

    report += """
### Interpretation

- If b AUC ≈ 0.5: `b` carries no discriminative signal
- High circular variance indicates near-uniform distribution
- See `b_distributions.png` for visualization

---

## Task 2: Statevector Simulation Comparison

### 3-Condition AUC Results

| Condition | Mean Raw AUC | Mean Calibrated AUC |
|-----------|--------------|---------------------|
"""

    for cond_name, cond_data in results['task2_simulation'].items():
        if 'error' in cond_data:
            continue
        report += f"| {cond_name} | {cond_data['mean_raw_auc']:.4f} | {cond_data['mean_calibrated_auc']:.4f} |\n"

    report += """
### Per-Subject Breakdown

| Subject | Computed b | Fixed π | Fixed 0 |
|---------|------------|---------|---------|
"""

    for subject in PRIORITY_SUBJECTS:
        row = f"| {subject} |"
        for cond in ['condition_a_computed_b', 'condition_b_fixed_pi', 'condition_c_fixed_zero']:
            if subject in results['task2_simulation'].get(cond, {}).get('per_subject', {}):
                auc = results['task2_simulation'][cond]['per_subject'][subject]['calibrated_auc']
                row += f" {auc:.4f} |"
            else:
                row += " N/A |"
        report += row + "\n"

    report += f"""
---

## Task 3: Sensitivity Analysis (b Sweep)

**Subject:** {results['task3_sweep']['subject']}

| b Value (×π) | Calibrated AUC |
|--------------|----------------|
"""

    for b_pi, auc in zip(results['task3_sweep']['b_values_pi'], results['task3_sweep']['calibrated_aucs']):
        report += f"| {b_pi:.2f} | {auc:.4f} |\n"

    report += f"""
**AUC Range:** {results['task3_sweep']['auc_range']:.4f}
**AUC Std Dev:** {results['task3_sweep']['auc_std']:.4f}
**Best b:** {results['task3_sweep']['best_b_pi']:.2f}π (AUC = {results['task3_sweep']['best_auc']:.4f})
**Flat?** {results['task3_sweep']['is_flat']}

See `b_sweep.png` for visualization.

---

## Task 4: Per-Channel Variance Analysis

| Channel | Ictal Var | Interictal Var | Within chb01 |
|---------|-----------|----------------|--------------|
"""

    for ch, stats in results['task4_variance']['per_channel'].items():
        report += f"| {ch} | {stats['ictal_circvar']:.4f} | {stats['interictal_circvar']:.4f} | "
        report += f"{stats['within_chb01_circvar']:.4f} |\n"

    summary = results['task4_variance']['summary']
    report += f"""
**Mean Ictal Variance:** {summary['mean_ictal_circvar']:.4f}
**Mean Interictal Variance:** {summary['mean_interictal_circvar']:.4f}
**High Variance?** {summary['high_variance']}

---

## Decision Gate Results

| Factor | Value |
|--------|-------|
| Pooled b AUC | {conclusion['decision_factors']['pooled_b_auc']:.4f} |
| b has no signal (< 0.55) | {conclusion['decision_factors']['b_has_no_signal']} |
| Computed b AUC | {conclusion['decision_factors']['computed_b_auc']:.4f} |
| Fixed π AUC | {conclusion['decision_factors']['fixed_pi_auc']:.4f} |
| Fixed 0 AUC | {conclusion['decision_factors']['fixed_zero_auc']:.4f} |
| π ≥ computed | {conclusion['decision_factors']['pi_better_or_equal']} |
| Sweep flat | {conclusion['decision_factors']['sweep_flat']} |
| High variance | {conclusion['decision_factors']['high_variance']} |

---

## Conclusion

**{conclusion['conclusion'].upper()}**

{conclusion['recommended_action']}

### Implications

"""

    if conclusion['conclusion'] == 'noise_confirmed':
        report += """
- **Paper 2 encoding narrative simplifies**: A-Gate encodes amplitude (`a`) and phase synchrony (`c`); `b` removed from model description
- **Within-patient variance reduces**: removing high-variance uninformative rotation should tighten ⟨Z⟩ distribution
- **Hardware run efficiency**: fewer encoding parameters to characterize for ibm_torino session
- **QNFM theoretical clarity**: quantum neural field mapping argument becomes cleaner — circuit is a 2-parameter probe
"""
    elif conclusion['conclusion'] == 'signal_present':
        report += """
- `b` carries state-discriminative information
- Retain computed `b` in the encoding
- Document the signal mechanism for Paper 2
"""
    else:
        report += """
- Results are ambiguous
- Further investigation required before hardware experiments
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"Saved report to {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run complete b parameter investigation."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PROMPT 019 - b Parameter Investigation")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now()}")

    # Load channels
    channels = load_channels()
    logger.info(f"Channels: {channels}")

    # Task 1: Distribution analysis
    task1_results = task1_distribution_analysis(channels)

    # Plot distributions
    if task1_results.get('b_ictal_all') and task1_results.get('b_interictal_all'):
        plot_b_distributions(
            task1_results['b_ictal_all'],
            task1_results['b_interictal_all'],
            task1_results['per_subject'],
            OUTPUT_DIR / 'b_distributions.png'
        )

    # Task 2: Simulation comparison
    task2_results = task2_simulation_comparison(channels)

    # Task 3: Sensitivity analysis
    task3_results = task3_sensitivity_analysis(channels, subject_id='chb01')

    # Plot sweep
    if 'error' not in task3_results:
        plot_b_sweep(task3_results, OUTPUT_DIR / 'b_sweep.png')

    # Task 4: Variance analysis
    task4_results = task4_variance_analysis(channels)

    # Determine conclusion
    conclusion = determine_conclusion(task1_results, task2_results, task3_results, task4_results)

    # Assemble final results
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'prompt': 'PROMPT 019',
        'hypothesis': 'b is noise',
        'task1_distribution': {
            'per_subject': task1_results['per_subject']
        },
        'task2_simulation': task2_results,
        'task3_sweep': task3_results,
        'task4_variance': task4_results,
        'conclusion': conclusion
    }

    # Save JSON
    json_path = OUTPUT_DIR / 'b_parameter_investigation.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)
    logger.info(f"Saved JSON to {json_path}")

    # Generate markdown report
    generate_markdown_report(final_results, OUTPUT_DIR / 'b_parameter_report.md')

    # Summary
    print("\n" + "=" * 70)
    print("PROMPT 019 - b Parameter Investigation Complete")
    print("=" * 70)
    print(f"\nConclusion: {conclusion['conclusion']}")
    print(f"Recommended Action: {conclusion['recommended_action']}")
    print(f"\nKey Metrics:")
    print(f"  Pooled b-alone AUC: {conclusion['decision_factors']['pooled_b_auc']:.4f}")
    print(f"  Computed b AUC: {conclusion['decision_factors']['computed_b_auc']:.4f}")
    print(f"  Fixed π AUC: {conclusion['decision_factors']['fixed_pi_auc']:.4f}")
    print(f"  Fixed 0 AUC: {conclusion['decision_factors']['fixed_zero_auc']:.4f}")
    print(f"  Sweep range: {task3_results.get('auc_range', 'N/A')}")
    print(f"\nOutput files:")
    print(f"  {json_path}")
    print(f"  {OUTPUT_DIR / 'b_parameter_report.md'}")
    print(f"  {OUTPUT_DIR / 'b_distributions.png'}")
    print(f"  {OUTPUT_DIR / 'b_sweep.png'}")

    logger.info(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
