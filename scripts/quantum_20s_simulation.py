#!/usr/bin/env python3
"""
================================================================================
PROMPT 017-SIM - Quantum PLV CH8, 20s Windows, Statevector Simulation
================================================================================

Local simulation preview of PROMPT 017. Runs the same PLV CH8 circuit with
20-second windows using statevector simulation instead of IBM hardware.
No credits needed.

Uses:
1. Predicts what March 22 hardware run should produce
2. Tests whether simulation-hardware inversion persists at 20s windows:
   - At 1.95s: simulation AUC 0.460, hardware calibrated 0.637 (INVERSION)
   - At 20s: if simulation improves, noisy PLV was the root cause
   - If simulation stays low, inversion is fundamental to encoding geometry

Backend: Qiskit Aer StatevectorSimulator
Encoding: PLV_theta_alpha (SAME as PROMPT 006)
Channels: CH8 (same channel set)
Window: 20 seconds

Output: results/window_analysis/quantum_20s_simulation.json

Author: Claude Code
Date: 2026-02-22
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

from QA1.multichannel_circuit import create_multichannel_circuit

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "window_analysis"
DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")

# Classical scaling results (for channel list)
CLASSICAL_RESULTS = PROJECT_ROOT / "results" / "scaling" / "classical_channel_scaling.json"

# PROMPT 011 quantum profiles (for polarity reference)
QUANTUM_PROFILES = PROJECT_ROOT / "results" / "patient_analysis" / "quantum_patient_profiles.json"

# PROMPT 016 Tier 3 polarity results
TIER3_POLARITY = PROJECT_ROOT / "results" / "window_analysis" / "tier3_polarity_calibration.json"

# PLV encoding parameters - SAME as PROMPT 006
BAND = (4, 13)  # theta-alpha

# 20-second window
WINDOW_SECONDS = 20.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * FS)  # 5120 samples at 256 Hz

# Priority subjects - SAME as all prior experiments
PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']

# Reference values from prior experiments
SIM_1_95S_RAW = 0.4597  # From PROMPT 006 simulation baseline
HW_1_95S_RAW = 0.5241
HW_1_95S_CALIBRATED = 0.6365


# =============================================================================
# PLV ENCODING - IDENTICAL TO PROMPT 006
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


def load_hardware_polarities() -> Dict[str, str]:
    """Load per-subject polarity from PROMPT 006 / PROMPT 011 hardware runs."""
    polarities = {}

    if QUANTUM_PROFILES.exists():
        with open(QUANTUM_PROFILES, 'r') as f:
            profiles = json.load(f)

        for subject_id in PRIORITY_SUBJECTS:
            if subject_id in profiles.get('patient_profiles', {}):
                patient = profiles['patient_profiles'][subject_id]
                for run in patient.get('hardware_runs', []):
                    if run.get('encoding') == 'PLV_theta_alpha' and 'HARDWARE' in run.get('backend', ''):
                        polarities[subject_id] = run.get('polarity', 'unknown')
                        break

    return polarities


def load_tier3_polarities() -> Dict[str, str]:
    """Load per-subject Tier 3 polarity from PROMPT 016."""
    polarities = {}

    if TIER3_POLARITY.exists():
        with open(TIER3_POLARITY, 'r') as f:
            data = json.load(f)

        # Use 20s window polarity for comparison
        window_20 = data.get('per_window_size', {}).get('20', {}).get('per_subject', {})
        for subject_id in PRIORITY_SUBJECTS:
            if subject_id in window_20:
                polarities[subject_id] = window_20[subject_id].get('tier3_polarity', 'unknown')

    return polarities


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

def run_statevector_simulation(circuit) -> Dict[str, complex]:
    """Run circuit on statevector simulator and return state amplitudes."""
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
    """
    Compute fidelity between two probability distributions from statevector.

    Uses Bhattacharyya coefficient (equivalent to Hellinger fidelity for probabilities).
    """
    all_keys = set(p.keys()) | set(q.keys())

    sqrt_sum = 0.0
    for key in all_keys:
        p_prob = p.get(key, 0)
        q_prob = q.get(key, 0)
        sqrt_sum += np.sqrt(p_prob * q_prob)

    return sqrt_sum ** 2


# =============================================================================
# LOSO CV WITH STATEVECTOR SIMULATION
# =============================================================================

def run_simulation_loso_20s(channels: List[str], subjects: List[str]) -> Dict:
    """
    Run QPNN LOSO CV with 20-second windows using statevector simulation.

    Key difference from PROMPT 006:
    - Uses first 20s of each segment (not multiple 2s sub-windows)
    - ONE circuit per segment
    - Statevector simulation (no shot noise)
    """
    n_channels = len(channels)
    n_qubits = 2 * n_channels + 1

    logger.info(f"Running Statevector Simulation LOSO CV with 20-second windows")
    logger.info(f"Channels: {n_channels}, Qubits: {n_qubits}")
    logger.info(f"Window: {WINDOW_SECONDS}s ({WINDOW_SAMPLES} samples)")
    logger.info(f"Subjects: {subjects}")

    # Load polarity references
    hw_polarities = load_hardware_polarities()
    tier3_polarities = load_tier3_polarities()
    logger.info(f"Hardware polarities (1.95s): {hw_polarities}")
    logger.info(f"Tier 3 polarities (20s): {tier3_polarities}")

    # Load all subject data
    subject_data = {}
    for subj in subjects:
        eeg_list, labels = load_subject_data(subj, channels)
        if len(eeg_list) > 0:
            label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
            y = np.array([label_map.get(l, 0) for l in labels])
            if len(np.unique(y)) == 2:
                subject_data[subj] = (eeg_list, labels, y)
                logger.info(f"  {subj}: {len(eeg_list)} segments, {int(sum(y))} seizure")

    if not subject_data:
        return {'error': 'No valid subjects found'}

    per_subject = {}
    all_y_true = []
    all_scores = []

    for test_subject in subject_data.keys():
        logger.info(f"\n{'='*60}")
        logger.info(f"Test subject: {test_subject}")
        logger.info(f"{'='*60}")

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
            logger.warning(f"  Skipped: missing template class")
            continue

        logger.info(f"  Building templates: {len(train_ictal)} ictal, {len(train_inter)} interictal")

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

        # Create and simulate template circuits
        ictal_circuit = create_multichannel_circuit(ictal_avg)
        inter_circuit = create_multichannel_circuit(inter_avg)

        logger.info("  Running template circuits on statevector simulator...")
        ictal_template_probs = run_statevector_simulation(ictal_circuit)
        inter_template_probs = run_statevector_simulation(inter_circuit)

        # Test data - extract 20s windows
        test_eeg, test_labels, y_test = subject_data[test_subject]

        scores = []
        valid_indices = []

        logger.info(f"  Processing {len(test_eeg)} test segments...")

        for idx, eeg in enumerate(test_eeg):
            eeg_20s = eeg[:, :WINDOW_SAMPLES]
            if eeg_20s.shape[1] >= WINDOW_SAMPLES:
                params = extract_plv_params(eeg_20s, fs=FS, band=BAND)
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
        polarity = 'inverted' if auc < 0.5 else 'standard'
        calibrated_auc = 1.0 - auc if auc < 0.5 else auc

        hw_polarity = hw_polarities.get(test_subject, 'unknown')
        tier3_polarity = tier3_polarities.get(test_subject, 'unknown')

        per_subject[test_subject] = {
            'raw_auc': float(auc),
            'calibrated_auc': float(calibrated_auc),
            'polarity': polarity,
            'polarity_hw_1.95s': hw_polarity,
            'polarity_tier3': tier3_polarity,
            'matches_hw': polarity == hw_polarity if hw_polarity != 'unknown' else None,
            'matches_tier3': polarity == tier3_polarity if tier3_polarity != 'unknown' else None,
            'n_samples': len(y_test_valid),
            'n_seizure': int(sum(y_test_valid))
        }

        all_y_true.extend(y_test_valid.tolist())
        all_scores.extend(scores.tolist())

        logger.info(f"  {test_subject}: raw AUC = {auc:.4f}, calibrated = {calibrated_auc:.4f}, polarity = {polarity}")

    # Overall AUCs
    try:
        overall_raw_auc = roc_auc_score(all_y_true, all_scores)
    except:
        overall_raw_auc = 0.5

    # Weighted calibrated AUC
    total_samples = sum(ps['n_samples'] for ps in per_subject.values())
    overall_calibrated_auc = sum(
        ps['calibrated_auc'] * ps['n_samples'] for ps in per_subject.values()
    ) / total_samples if total_samples > 0 else 0.5

    n_inverted = sum(1 for ps in per_subject.values() if ps['polarity'] == 'inverted')
    n_matches_hw = sum(1 for ps in per_subject.values() if ps.get('matches_hw'))
    n_matches_tier3 = sum(1 for ps in per_subject.values() if ps.get('matches_tier3'))

    # Compute calibrated 1.95s simulation AUC for comparison
    # (We don't have raw per-subject data, so estimate from raw overall)
    sim_1_95s_calibrated = max(SIM_1_95S_RAW, 1.0 - SIM_1_95S_RAW)

    # Inversion analysis
    window_effect = overall_raw_auc - SIM_1_95S_RAW
    inversion_persists = overall_calibrated_auc < HW_1_95S_CALIBRATED - 0.05

    return {
        'config': {
            'backend': 'statevector_simulation',
            'encoding': 'PLV_theta_alpha',
            'channels': channels,
            'window_seconds': WINDOW_SECONDS,
            'n_qubits': n_qubits
        },
        'overall_raw_auc': float(overall_raw_auc),
        'overall_calibrated_auc': float(overall_calibrated_auc),
        'calibration_lift': float(overall_calibrated_auc - overall_raw_auc),
        'n_inverted': n_inverted,
        'per_subject': per_subject,
        'polarity_concordance': {
            'n_matches_hw_1.95s': n_matches_hw,
            'n_matches_tier3': n_matches_tier3,
            'hw_concordance_rate': n_matches_hw / len(per_subject) if per_subject else 0,
            'tier3_concordance_rate': n_matches_tier3 / len(per_subject) if per_subject else 0
        },
        'inversion_analysis': {
            'sim_1.95s_raw': SIM_1_95S_RAW,
            'sim_1.95s_calibrated': float(sim_1_95s_calibrated),
            'sim_20s_raw': float(overall_raw_auc),
            'sim_20s_calibrated': float(overall_calibrated_auc),
            'hw_1.95s_raw': HW_1_95S_RAW,
            'hw_1.95s_calibrated': HW_1_95S_CALIBRATED,
            'window_effect_on_simulation': float(window_effect),
            'inversion_persists_at_20s': inversion_persists,
            'interpretation': (
                "Window size partially explains inversion" if window_effect > 0.05
                else "Inversion is fundamental to encoding geometry, not window-dependent"
            )
        },
        'aggregation': {
            'note': 'Single sub-window per segment - MAX/MEAN/STD are identical',
            'auc': float(overall_raw_auc)
        },
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run 20-second window statevector simulation."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PROMPT 017-SIM - Quantum PLV CH8, 20s Windows, Statevector Simulation")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"Window size: {WINDOW_SECONDS}s")
    logger.info("Backend: Statevector Simulation (NO hardware credits needed)")

    # Load channels
    channels = load_channels()
    n_qubits = 2 * len(channels) + 1
    logger.info(f"\nChannels: {channels}")
    logger.info(f"Qubits: {n_qubits}")

    # Run LOSO CV
    subjects = PRIORITY_SUBJECTS
    logger.info(f"\nPriority subjects: {subjects}")

    results = run_simulation_loso_20s(channels, subjects)

    # Save results
    output_file = OUTPUT_DIR / 'quantum_20s_simulation.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("Quantum PLV CH8 Simulation - Window Size Comparison")
    print("=" * 70)
    print(f"{'Window':<10} {'Backend':<15} {'Raw AUC':<12} {'Calibrated':<12} {'Inverted':<10}")
    print("-" * 59)
    print(f"{'1.95s':<10} {'simulation':<15} {SIM_1_95S_RAW:<12.4f} {max(SIM_1_95S_RAW, 1-SIM_1_95S_RAW):<12.4f} {'?/7':<10}")
    print(f"{'1.95s':<10} {'hardware':<15} {HW_1_95S_RAW:<12.4f} {HW_1_95S_CALIBRATED:<12.4f} {'3/7':<10}")

    if 'error' not in results:
        print(f"{'20.0s':<10} {'simulation':<15} {results['overall_raw_auc']:<12.4f} {results['overall_calibrated_auc']:<12.4f} {results['n_inverted']}/7")
        print(f"{'20.0s':<10} {'hardware':<15} {'(Mar 22)':<12} {'(Mar 22)':<12} {'?/7':<10}")

        print()
        print(f"Simulation-hardware inversion at 1.95s: YES (hw calibrated > sim calibrated)")

        inv = results['inversion_analysis']
        if inv['window_effect_on_simulation'] > 0.05:
            print(f"Window effect on simulation: +{inv['window_effect_on_simulation']:.4f}")
            print(f"Interpretation: Window size partially explains the inversion")
        else:
            print(f"Window effect on simulation: {inv['window_effect_on_simulation']:+.4f}")
            print(f"Interpretation: Inversion is fundamental to encoding geometry")

        print()
        print(f"Prediction for March 22 hardware (20s):")
        if results['overall_calibrated_auc'] > 0.60:
            print(f"  If simulation improves, hardware should also improve")
            print(f"  Expected hardware calibrated AUC: > {results['overall_calibrated_auc']:.3f}")
        else:
            print(f"  Simulation still low ({results['overall_calibrated_auc']:.3f})")
            print(f"  Hardware may still show inversion (exceed simulation)")

        # Polarity concordance
        print()
        conc = results['polarity_concordance']
        print(f"Polarity concordance with hardware (1.95s): {conc['n_matches_hw_1.95s']}/7 ({conc['hw_concordance_rate']*100:.0f}%)")
        print(f"Polarity concordance with Tier 3 (20s): {conc['n_matches_tier3']}/7 ({conc['tier3_concordance_rate']*100:.0f}%)")

    else:
        print(f"\nError: {results['error']}")

    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
