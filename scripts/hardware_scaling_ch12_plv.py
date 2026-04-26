#!/usr/bin/env python3
"""
================================================================================
PROMPT 013 - Hardware Scaling: 12 Channels PLV on IBM Heron (Mar 22+)
================================================================================

12-channel hardware run using segment-level PLV encoding.
Scheduled for March 22, 2026+ when IBM Quantum credits reset.

Backend: IBM Heron r2 (ibm_torino) - no simulator
Encoding: PLV_theta_alpha (segment-level, NOT sub-windows)
Channels: 12 (25 qubits)
Shots: 1024
Cost estimate: ~9s/subject (7 subjects = ~63s total)

Output: results/projection/hardware_scaling.json (updates ch12 section)
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
from sklearn.metrics import roc_auc_score
import logging
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'QA1'))

from sagemaker.train_chbmit import (
    FS,
    WINDOW_SEC,
    normalize_channel_label,
    extract_segments_for_subject as _extract_segments_for_subject,
    EEGSegment,
)

from QA1.multichannel_circuit import create_multichannel_circuit

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
EXISTING_SCALING = Path("results/projection/hardware_scaling.json")

# Hardware configuration
SHOTS = 1024
BACKEND_NAME = "ibm_torino"

# PLV encoding parameters
BAND = (4, 13)  # theta-alpha

# Priority subjects (same as PROMPT 006/008)
PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']

# Channel configuration
N_CHANNELS = 12
N_QUBITS = 2 * N_CHANNELS + 1  # = 25


# =============================================================================
# PLV SEGMENT-LEVEL ENCODING (NOT sub-windows - keeps costs low)
# =============================================================================

def extract_plv_params(eeg_segment: np.ndarray, fs: float = 256.0,
                       band: Tuple[float, float] = (4, 13)) -> List[Tuple[float, float, float]]:
    """
    Extract (a, b, c) per channel using Hilbert phase/amplitude.

    This is SEGMENT-LEVEL encoding - uses the entire segment, not sub-windows.
    This keeps hardware costs at ~9s/subject instead of ~228s/subject.
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
    """Load CH12 channel list from classical scaling results."""
    if CLASSICAL_SCALING.exists():
        with open(CLASSICAL_SCALING, 'r') as f:
            data = json.load(f)
        if 'ch12' in data:
            return data['ch12']['channels']

    # Fallback
    return [
        "FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
        "FP2-F8", "F8-T8", "FP2-F4", "F4-C4",
        "C3-P3", "C4-P4", "CZ-PZ", "FZ-CZ"
    ]


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
# IBM HARDWARE EXECUTION
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

    logger.info(f"Looking for backend: {BACKEND_NAME}")

    try:
        backend = service.backend(BACKEND_NAME)
        logger.info(f"Using backend: {backend.name} ({backend.num_qubits} qubits)")
        return backend
    except Exception:
        pass

    # Fallback to least busy with sufficient qubits
    try:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=N_QUBITS
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

    logger.info(f"Transpiling {len(circuits)} circuits for {backend.name}...")

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    transpiled = pm.run(circuits)

    logger.info(f"Submitting to hardware with {shots} shots...")

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
    """Compute Hellinger fidelity between two distributions."""
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
# LOSO CV
# =============================================================================

def run_hardware_loso(channels: List[str], subjects: List[str], backend) -> Dict:
    """Run LOSO CV on IBM hardware for CH12 using segment-level PLV."""

    n_channels = len(channels)
    n_qubits = 2 * n_channels + 1

    logger.info(f"Running CH12 hardware LOSO: {n_channels} channels, {n_qubits} qubits")
    logger.info(f"Encoding: PLV_theta_alpha (segment-level)")
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

        # Build template circuits using SEGMENT-LEVEL PLV
        logger.info(f"  Building templates: {len(train_ictal)} ictal, {len(train_inter)} interictal")

        # Average parameters for templates (segment-level, not sub-windows)
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

        # Test data
        test_eeg, test_labels, y_test = subject_data[test_subject]

        # Build test circuits (segment-level PLV)
        logger.info(f"  Building {len(test_eeg)} test circuits...")
        test_circuits = []
        for eeg in test_eeg:
            params = extract_plv_params(eeg, fs=FS, band=BAND)
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

        polarity = 'inverted' if auc < 0.5 else 'standard'
        per_subject[test_subject] = {
            'auc': float(auc),
            'raw_auc': float(auc),
            'calibrated_auc': float(max(auc, 1.0 - auc)),
            'polarity': polarity,
            'n_samples': len(y_test),
            'n_seizure': int(sum(y_test)),
            'scores': scores.tolist(),
            'y_true': y_test.tolist(),
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
        'overall_auc': overall_auc,
        'per_subject': per_subject,
        'channels': channels,
        'n_qubits': n_qubits
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PROMPT 013 - Hardware Scaling: 12 Channels PLV on IBM Heron")
    print("=" * 70)
    print(f"Scheduled for: March 22, 2026+")
    print(f"Cost estimate: ~63s (9s/subject x 7 subjects)")
    print()

    # Load channels
    channels = load_channels()
    n_qubits = 2 * len(channels) + 1
    logger.info(f"CH12 channels: {channels}")
    logger.info(f"Qubits required: {n_qubits}")

    # Connect to IBM hardware
    try:
        backend = connect_to_ibm()
    except Exception as e:
        logger.error(f"Cannot connect to IBM hardware: {e}")
        logger.info("Script ready - run when credits are available (Mar 22+)")
        return None

    # Run LOSO on hardware
    logger.info("\nRunning CH12 LOSO CV on hardware...")
    result = run_hardware_loso(channels, PRIORITY_SUBJECTS, backend)

    if 'error' in result:
        logger.error(f"Error: {result['error']}")
        return None

    # Load existing results and update
    existing = {}
    if EXISTING_SCALING.exists():
        with open(EXISTING_SCALING, 'r') as f:
            existing = json.load(f)

    # Update CH12 section
    existing['ch12'] = {
        'backend': backend.name,
        'encoding': 'PLV_theta_alpha',
        'mitigation': 'none',
        'shots': SHOTS,
        'n_qubits': n_qubits,
        'channels': channels,
        'overall_auc': result['overall_auc'],
        'per_subject': result['per_subject']
    }
    existing['timestamp'] = datetime.now().isoformat()

    # Update gap table
    sim_ch12_auc = None
    sim_scaling = Path("results/projection/simulation_scaling.json")
    if sim_scaling.exists():
        with open(sim_scaling, 'r') as f:
            sim_data = json.load(f)
        sim_ch12_auc = sim_data.get('ch12', {}).get('overall_auc')

    gap_entry = {
        'channels': 12,
        'qubits': n_qubits,
        'sim_auc': sim_ch12_auc,
        'hw_auc': result['overall_auc'],
        'gap': (sim_ch12_auc - result['overall_auc']) if sim_ch12_auc else None,
        'encoding_match': True
    }

    # Update or append gap table entry
    gap_table = existing.get('gap_table', [])
    ch12_idx = next((i for i, g in enumerate(gap_table) if g.get('channels') == 12), None)
    if ch12_idx is not None:
        gap_table[ch12_idx] = gap_entry
    else:
        gap_table.append(gap_entry)
    gap_table.sort(key=lambda x: x['channels'])
    existing['gap_table'] = gap_table

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "hardware_scaling.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("CH12 HARDWARE RESULTS")
    print("=" * 70)
    print(f"Backend: {backend.name}")
    print(f"Encoding: PLV_theta_alpha (segment-level)")
    print(f"Channels: {len(channels)}, Qubits: {n_qubits}")
    print(f"Overall AUC: {result['overall_auc']:.4f}")

    if sim_ch12_auc:
        gap = sim_ch12_auc - result['overall_auc']
        print(f"Simulation AUC: {sim_ch12_auc:.4f}")
        print(f"Hardware gap: {gap:.4f} ({100*gap/sim_ch12_auc:.1f}%)")

    print(f"\nPer-subject:")
    for subj, data in sorted(result['per_subject'].items()):
        print(f"  {subj}: {data['auc']:.4f}")

    print(f"\nSaved: {output_path}")

    return result


if __name__ == "__main__":
    main()
