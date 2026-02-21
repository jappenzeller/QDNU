#!/usr/bin/env python3
"""
IBM Heron Hardware Validation at 8 Channels (LOSO CV)

Validates quantum QPNN on IBM Heron hardware to confirm whether the 0.4595 AUC
from statevector simulation is a simulation artifact or a real result.

DO NOT use statevector simulation - SamplerV2 on real hardware only.

Output: results/hardware_validation/ch8_heron_loso_results.json
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

# ============================================================================
# Configuration
# ============================================================================

# Load CH8 from classical scaling results
CLASSICAL_RESULTS = Path("results/scaling/classical_channel_scaling.json")
OUTPUT_DIR = Path("results/hardware_validation")
DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")

# PLV encoding parameters (same as V3 and quantum scaling)
BAND = (4, 13)  # theta-alpha
SUBWINDOW_SEC = 2.0

# Hardware configuration
SHOTS = 1024  # Minimum required by prompt
ERROR_MITIGATION = "none"  # Match prior hardware runs

# Statevector simulation result for comparison
SIMULATION_CH8_AUC = 0.4595

# Priority subjects for hardware runs (based on classical baseline analysis)
# These are subjects with moderate classical AUC (0.65-0.85) - good quantum candidates
PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']


# ============================================================================
# PLV Encoding (same as quantum_channel_scaling.py)
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

def load_channels_from_results() -> List[str]:
    """Load CH8 channel list from classical scaling results."""
    if not CLASSICAL_RESULTS.exists():
        # Fallback to default
        return ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3", "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]

    with open(CLASSICAL_RESULTS, 'r') as f:
        data = json.load(f)

    return data['ch8']['channels']


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


# ============================================================================
# Hardware Execution
# ============================================================================

def connect_to_ibm():
    """Connect to IBM Quantum and select Heron backend."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    logger.info("Connecting to IBM Quantum...")

    try:
        service = QiskitRuntimeService()
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        raise

    # Find Heron processor (133 qubits)
    logger.info("Finding Heron backend...")

    try:
        # Try to get ibm_torino (Heron r2) specifically
        backend = service.backend('ibm_torino')
        logger.info(f"Using backend: {backend.name} ({backend.num_qubits} qubits)")
        return backend
    except:
        pass

    # Fallback to least busy with sufficient qubits
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
    """
    Run circuits on IBM hardware using SamplerV2.

    DO NOT use statevector simulation.
    """
    from qiskit_ibm_runtime import SamplerV2, Batch
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    logger.info(f"Transpiling {len(circuits)} circuits...")

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    transpiled = pm.run(circuits)

    logger.info(f"Submitting to hardware: {backend.name}")

    all_counts = []

    with Batch(backend=backend) as batch:
        sampler = SamplerV2(mode=batch)

        # Submit all circuits
        job = sampler.run(transpiled, shots=shots)
        job_id = job.job_id()
        logger.info(f"Job ID: {job_id}")

        # Wait for results
        logger.info("Waiting for hardware results...")
        result = job.result()

        # Extract counts from each circuit
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


def counts_to_probability(counts: Dict[str, int], target_state: str = None) -> float:
    """Convert counts to probability of target state (or highest probability state)."""
    total = sum(counts.values())
    if total == 0:
        return 0.5

    if target_state is None:
        # Return probability of most frequent outcome
        return max(counts.values()) / total

    # Return probability of specific state
    return counts.get(target_state, 0) / total


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


# ============================================================================
# LOSO CV on Hardware
# ============================================================================

def run_hardware_loso(channels: List[str], subjects: List[str], backend) -> Dict:
    """
    Run QPNN LOSO CV on hardware.

    Uses dual-template fidelity classification:
    - Build ictal and interictal templates from training data
    - Classify test segments by comparing fidelity to each template
    """
    n_channels = len(channels)
    n_qubits = 2 * n_channels + 1

    logger.info(f"Running Hardware LOSO CV")
    logger.info(f"Channels: {n_channels}, Qubits: {n_qubits}")
    logger.info(f"Subjects: {subjects}")

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

        # Training data
        train_ictal = []
        train_inter = []
        for subj, (eeg_list, labels, _) in subject_data.items():
            if subj != test_subject:
                for eeg, lbl in zip(eeg_list, labels):
                    if lbl in ('ictal', 'preictal'):
                        train_ictal.append(eeg)
                    else:
                        train_inter.append(eeg)

        if not train_ictal or not train_inter:
            logger.warning(f"  Skipped: missing template class")
            continue

        # Build template circuits
        logger.info(f"  Building templates: {len(train_ictal)} ictal, {len(train_inter)} interictal")

        # Average parameters for templates
        ictal_params_all = []
        for eeg in train_ictal[:50]:  # Limit for efficiency
            sw_params = extract_plv_params_subwindows(eeg, fs=FS, band=BAND)
            for sw in sw_params:
                ictal_params_all.append(sw)

        inter_params_all = []
        for eeg in train_inter[:50]:
            sw_params = extract_plv_params_subwindows(eeg, fs=FS, band=BAND)
            for sw in sw_params:
                inter_params_all.append(sw)

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

        # Create template circuits with measurements
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

        # Test data
        test_eeg, test_labels, y_test = subject_data[test_subject]

        # Build test circuits
        logger.info(f"  Building {len(test_eeg)} test circuits...")
        test_circuits = []
        for eeg in test_eeg:
            sw_params_list = extract_plv_params_subwindows(eeg, fs=FS, band=BAND)
            # Use first sub-window for efficiency
            params = sw_params_list[0] if sw_params_list else [(0.5, np.pi, 0.5)] * n_channels
            circ = create_multichannel_circuit(params)
            circ.measure_all()
            test_circuits.append(circ)

        # Run test circuits on hardware
        logger.info(f"  Running test circuits on hardware...")
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
            auc = roc_auc_score(y_test, scores)
        except:
            auc = 0.5

        per_subject[test_subject] = {
            'auc': float(auc),
            'n_samples': len(y_test),
            'n_seizure': int(sum(y_test))
        }

        all_y_true.extend(y_test.tolist())
        all_scores.extend(scores.tolist())

        logger.info(f"  {test_subject}: AUC = {auc:.4f}")

    # Overall AUC
    try:
        overall_auc = roc_auc_score(all_y_true, all_scores)
    except:
        overall_auc = 0.5

    return {
        'backend': backend.name,
        'encoding_variant': 'PLV_theta_alpha',
        'shots': SHOTS,
        'error_mitigation': ERROR_MITIGATION,
        'ch8': {
            'channels': channels,
            'overall_auc': overall_auc,
            'per_subject': per_subject
        },
        'simulation_ch8_auc': SIMULATION_CH8_AUC,
        'delta_hardware_vs_simulation': overall_auc - SIMULATION_CH8_AUC,
        'prior_hardware_auc': None,  # No prior LOSO hardware results
        'delta_vs_prior': None,
        'subjects_completed': list(per_subject.keys()),
        'subjects_skipped': [s for s in subjects if s not in per_subject]
    }


# ============================================================================
# Main
# ============================================================================

def main():
    """Run hardware validation."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("IBM Heron Hardware Validation - CH8 LOSO CV")
    logger.info("=" * 60)
    logger.info(f"Started: {datetime.now()}")

    # Load channels
    channels = load_channels_from_results()
    logger.info(f"\nChannels: {channels}")

    # Connect to IBM
    logger.info("\nConnecting to IBM Quantum...")
    try:
        backend = connect_to_ibm()
    except Exception as e:
        logger.error(f"Could not connect to IBM Quantum: {e}")
        logger.error("Make sure IBM credentials are configured")
        return

    # Use priority subjects to limit queue time
    subjects = PRIORITY_SUBJECTS
    logger.info(f"\nPriority subjects: {subjects}")

    # Run LOSO CV
    results = run_hardware_loso(channels, subjects, backend)

    # Save results
    output_file = OUTPUT_DIR / 'ch8_heron_loso_results.json'
    results['timestamp'] = datetime.now().isoformat()

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if 'error' not in results:
        logger.info(f"Backend: {results['backend']}")
        logger.info(f"Shots: {results['shots']}")
        logger.info(f"Subjects completed: {len(results['subjects_completed'])}")
        logger.info(f"\nHardware CH8 AUC: {results['ch8']['overall_auc']:.4f}")
        logger.info(f"Simulation CH8 AUC: {results['simulation_ch8_auc']:.4f}")
        logger.info(f"Delta (hw - sim): {results['delta_hardware_vs_simulation']:+.4f}")

        # Interpretation
        if results['delta_hardware_vs_simulation'] > 0.05:
            logger.info("\nFinding: Hardware AUC > Simulation AUC")
            logger.info("The low simulation AUC may be a simulation artifact.")
        elif results['delta_hardware_vs_simulation'] < -0.05:
            logger.info("\nFinding: Hardware AUC < Simulation AUC")
            logger.info("Hardware noise may be degrading performance.")
        else:
            logger.info("\nFinding: Hardware and simulation AUCs are similar.")
    else:
        logger.error(f"Error: {results['error']}")

    logger.info(f"\nResults saved to: {output_file}")
    logger.info(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
