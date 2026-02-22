#!/usr/bin/env python3
"""
================================================================================
PROMPT 017 - Quantum PLV CH8 with 20-Second Windows on Hardware
================================================================================

Tests whether quantum PLV encoding benefits from longer windows, matching
the improvement seen in classical Tier 2 (0.72 -> 0.87 at 20s windows).

Uses IDENTICAL PLV encoding as PROMPT 006, only changes window size.

Current: PLV CH8, 1.95s windows, calibrated AUC 0.637
Prediction: 20s windows should improve AUC due to more stable PLV estimation.

Backend: IBM Heron (ibm_torino)
Encoding: PLV_theta_alpha (SAME as PROMPT 006)
Channels: CH8 (SAME channel set)
Window: 20 seconds (was ~1.95s in PROMPT 006)

Output: results/window_analysis/quantum_20s_hardware.json

Author: Claude Code
Date: 2026-02-22
Status: READY (waiting for credits, March 22+)
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

# PROMPT 006 results (for comparison)
PROMPT_006_RESULTS = PROJECT_ROOT / "results" / "hardware_validation" / "ch8_heron_loso_results.json"

# PROMPT 011 quantum profiles (for polarity reference)
QUANTUM_PROFILES = PROJECT_ROOT / "results" / "patient_analysis" / "quantum_patient_profiles.json"

# PLV encoding parameters - SAME as PROMPT 006
BAND = (4, 13)  # theta-alpha

# NEW: 20-second window instead of 2s sub-windows
WINDOW_SECONDS = 20.0
WINDOW_SAMPLES = int(WINDOW_SECONDS * FS)  # 5120 samples at 256 Hz

# Hardware configuration - SAME as PROMPT 006
SHOTS = 1024
ERROR_MITIGATION = "none"

# Priority subjects - SAME as PROMPT 006
PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']

# Reference values from prior prompts
QUANTUM_1_95S_RAW = 0.5241
QUANTUM_1_95S_CALIBRATED = 0.6365
TIER2_20S = 0.8726  # from PROMPT 015
TIER3_CALIBRATED_30S = 0.724  # from PROMPT 016


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
        # Fallback to default CH8
        return ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3", "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]


def load_prompt_006_polarities() -> Dict[str, str]:
    """Load per-subject polarity from PROMPT 006 / PROMPT 011."""
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
# HARDWARE EXECUTION
# =============================================================================

def connect_to_ibm():
    """Connect to IBM Quantum and select Heron backend."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    logger.info("Connecting to IBM Quantum...")

    try:
        service = QiskitRuntimeService()
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        raise

    logger.info("Finding Heron backend...")

    try:
        backend = service.backend('ibm_torino')
        logger.info(f"Using backend: {backend.name} ({backend.num_qubits} qubits)")
        return backend
    except:
        pass

    try:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=17
        )
        logger.info(f"Using fallback backend: {backend.name} ({backend.num_qubits} qubits)")
        return backend
    except Exception as e:
        logger.error(f"No suitable backend found: {e}")
        raise


def run_circuits_on_hardware(circuits: List, backend, shots: int = SHOTS) -> List[Dict[str, int]]:
    """Run circuits on IBM hardware using SamplerV2."""
    from qiskit_ibm_runtime import SamplerV2, Batch
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    logger.info(f"Transpiling {len(circuits)} circuits...")

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    transpiled = pm.run(circuits)

    logger.info(f"Submitting to hardware: {backend.name}")

    all_counts = []

    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch)

        job = sampler.run(transpiled, shots=shots)
        job_id = job.job_id()
        logger.info(f"Job ID: {job_id}")

        logger.info("Waiting for hardware results...")
        result = job.result()

        for i in range(len(circuits)):
            pub_result = result[i]
            try:
                counts = pub_result.data.meas.get_counts()
            except:
                try:
                    counts = pub_result.data.c.get_counts()
                except:
                    counts = {}
            all_counts.append(dict(counts))

    return all_counts


def hellinger_fidelity(p: Dict[str, int], q: Dict[str, int]) -> float:
    """Compute Hellinger fidelity between two count distributions."""
    total_p = sum(p.values())
    total_q = sum(q.values())

    if total_p == 0 or total_q == 0:
        return 0.0

    all_keys = set(p.keys()) | set(q.keys())

    sqrt_sum = 0.0
    for key in all_keys:
        p_prob = p.get(key, 0) / total_p
        q_prob = q.get(key, 0) / total_q
        sqrt_sum += np.sqrt(p_prob * q_prob)

    return sqrt_sum ** 2


# =============================================================================
# LOSO CV WITH 20-SECOND WINDOWS
# =============================================================================

def run_hardware_loso_20s(channels: List[str], subjects: List[str], backend) -> Dict:
    """
    Run QPNN LOSO CV on hardware with 20-second windows.

    Key difference from PROMPT 006:
    - Uses first 20s of each segment (not multiple 2s sub-windows)
    - ONE circuit per segment (not ~15)
    - No sub-window aggregation needed
    """
    n_channels = len(channels)
    n_qubits = 2 * n_channels + 1

    logger.info(f"Running Hardware LOSO CV with 20-second windows")
    logger.info(f"Channels: {n_channels}, Qubits: {n_qubits}")
    logger.info(f"Window: {WINDOW_SECONDS}s ({WINDOW_SAMPLES} samples)")
    logger.info(f"Subjects: {subjects}")

    # Load prior polarity data for comparison
    prior_polarities = load_prompt_006_polarities()
    logger.info(f"Prior polarities (1.95s): {prior_polarities}")

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
                    # Take first 20 seconds of segment
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

        # Extract PLV params from 20s windows (ONE set per segment)
        ictal_params_all = []
        for eeg in train_ictal[:50]:  # Limit for efficiency
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

        # Create template circuits
        ictal_circuit = create_multichannel_circuit(ictal_avg)
        ictal_circuit.measure_all()
        inter_circuit = create_multichannel_circuit(inter_avg)
        inter_circuit.measure_all()

        # Run templates on hardware
        logger.info("  Running template circuits on hardware...")
        template_counts = run_circuits_on_hardware(
            [ictal_circuit, inter_circuit], backend, shots=SHOTS
        )
        ictal_template_counts = template_counts[0]
        inter_template_counts = template_counts[1]

        # Test data - extract 20s windows
        test_eeg, test_labels, y_test = subject_data[test_subject]

        # Build test circuits (ONE per segment, not multiple sub-windows)
        test_circuits = []
        valid_indices = []
        for idx, eeg in enumerate(test_eeg):
            eeg_20s = eeg[:, :WINDOW_SAMPLES]
            if eeg_20s.shape[1] >= WINDOW_SAMPLES:
                params = extract_plv_params(eeg_20s, fs=FS, band=BAND)
                circ = create_multichannel_circuit(params)
                circ.measure_all()
                test_circuits.append(circ)
                valid_indices.append(idx)

        y_test_valid = y_test[valid_indices]

        logger.info(f"  Running {len(test_circuits)} test circuits on hardware...")

        # Run test circuits
        test_counts = run_circuits_on_hardware(test_circuits, backend, shots=SHOTS)

        # Compute scores using fidelity
        scores = []
        for counts in test_counts:
            fid_ictal = hellinger_fidelity(counts, ictal_template_counts)
            fid_inter = hellinger_fidelity(counts, inter_template_counts)
            score = fid_ictal - fid_inter
            scores.append(score)

        scores = np.array(scores)

        # Compute AUC
        try:
            auc = roc_auc_score(y_test_valid, scores)
        except:
            auc = 0.5

        # Determine polarity
        polarity = 'inverted' if auc < 0.5 else 'standard'
        calibrated_auc = 1.0 - auc if auc < 0.5 else auc

        prior_polarity = prior_polarities.get(test_subject, 'unknown')
        polarity_changed = polarity != prior_polarity if prior_polarity != 'unknown' else None

        per_subject[test_subject] = {
            'raw_auc': float(auc),
            'calibrated_auc': float(calibrated_auc),
            'polarity': polarity,
            'polarity_1.95s': prior_polarity,
            'polarity_changed': polarity_changed,
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

    # Compute calibrated overall AUC using weighted mean of per-subject calibrated AUCs
    total_samples = sum(ps['n_samples'] for ps in per_subject.values())
    overall_calibrated_auc = sum(
        ps['calibrated_auc'] * ps['n_samples'] for ps in per_subject.values()
    ) / total_samples if total_samples > 0 else 0.5

    n_inverted = sum(1 for ps in per_subject.values() if ps['polarity'] == 'inverted')
    n_polarity_changed = sum(1 for ps in per_subject.values() if ps['polarity_changed'])

    return {
        'config': {
            'backend': backend.name,
            'encoding': 'PLV_theta_alpha',
            'channels': channels,
            'window_seconds': WINDOW_SECONDS,
            'shots': SHOTS,
            'n_qubits': n_qubits
        },
        'overall_raw_auc': float(overall_raw_auc),
        'overall_calibrated_auc': float(overall_calibrated_auc),
        'calibration_lift': float(overall_calibrated_auc - overall_raw_auc),
        'n_inverted': n_inverted,
        'n_polarity_changed': n_polarity_changed,
        'per_subject': per_subject,
        'comparison': {
            'quantum_1.95s_raw': QUANTUM_1_95S_RAW,
            'quantum_1.95s_calibrated': QUANTUM_1_95S_CALIBRATED,
            'quantum_20s_raw': float(overall_raw_auc),
            'quantum_20s_calibrated': float(overall_calibrated_auc),
            'improvement_from_window': float(overall_calibrated_auc - QUANTUM_1_95S_CALIBRATED),
            'tier2_20s': TIER2_20S,
            'tier3_calibrated_30s': TIER3_CALIBRATED_30S,
            'gap_to_tier2': float(TIER2_20S - overall_calibrated_auc),
            'gap_to_tier3_calibrated': float(TIER3_CALIBRATED_30S - overall_calibrated_auc)
        },
        'timestamp': datetime.now().isoformat()
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run 20-second window hardware validation."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PROMPT 017 - Quantum PLV CH8 with 20-Second Windows")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now()}")
    logger.info(f"Window size: {WINDOW_SECONDS}s (was 1.95s in PROMPT 006)")

    # Load channels
    channels = load_channels()
    n_qubits = 2 * len(channels) + 1
    logger.info(f"\nChannels: {channels}")
    logger.info(f"Qubits: {n_qubits}")

    # Connect to IBM
    logger.info("\nConnecting to IBM Quantum...")
    try:
        backend = connect_to_ibm()
    except Exception as e:
        logger.error(f"Could not connect to IBM Quantum: {e}")
        logger.error("Make sure IBM credentials are configured")
        return

    # Run LOSO CV
    subjects = PRIORITY_SUBJECTS
    logger.info(f"\nPriority subjects: {subjects}")

    results = run_hardware_loso_20s(channels, subjects, backend)

    # Save results
    output_file = OUTPUT_DIR / 'quantum_20s_hardware.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Summary
    print("\n" + "=" * 70)
    print("Quantum PLV CH8 - Window Size Comparison")
    print("=" * 70)
    print(f"{'Window':<10} {'Raw AUC':<12} {'Calibrated':<12} {'Inverted':<10}")
    print("-" * 44)
    print(f"{'1.95s':<10} {QUANTUM_1_95S_RAW:<12.4f} {QUANTUM_1_95S_CALIBRATED:<12.4f} {'3/7':<10}")

    if 'error' not in results:
        print(f"{'20.0s':<10} {results['overall_raw_auc']:<12.4f} {results['overall_calibrated_auc']:<12.4f} {results['n_inverted']}/7")
        print()
        print(f"Improvement from 20s windows: {results['comparison']['improvement_from_window']:+.4f}")
        print(f"Gap to Tier 2 (20s): {results['comparison']['gap_to_tier2']:.4f}")
        print(f"Gap to Tier 3 calibrated (30s): {results['comparison']['gap_to_tier3_calibrated']:.4f}")

        # Polarity changes
        changed = [s for s, ps in results['per_subject'].items() if ps.get('polarity_changed')]
        if changed:
            print(f"\nPolarity changed from 1.95s: {', '.join(changed)}")
        else:
            print(f"\nPolarity stable: all subjects have same polarity as 1.95s")
    else:
        print(f"\nError: {results['error']}")

    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
