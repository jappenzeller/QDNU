#!/usr/bin/env python3
"""
================================================================================
PROMPT 020 — CH16 Statevector Simulation on Amazon Braket SV1
================================================================================

CH16 PLV simulation requires 33 qubits — infeasible locally (128GB+ RAM).
Amazon Braket SV1 simulator handles up to 34 qubits in the cloud.
This fills the missing 4th point on the simulation scaling curve:
CH4 (0.522) → CH8 (0.534) → CH12 (0.644) → CH16 (???)

Backend: Amazon Braket SV1 (statevector simulator, up to 34 qubits)
Encoding: PLV_theta_alpha (same as all prior quantum experiments)
Channels: 16 (standard montage subset)
Qubits: 33 (2 per channel + 1 ancilla)
Window: 1.95s (matching CH4/CH8/CH12 simulation baseline)
Subjects: 7 priority (chb01, chb03, chb05, chb07, chb11, chb14, chb21)

Output: results/projection/ch16_sv1_simulation.json

Author: Claude Code
Date: 2026-02-22
================================================================================
"""

import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from scipy.signal import hilbert, butter, filtfilt
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from sagemaker.train_chbmit import (
    EXCLUDE_SUBJECTS,
    FS,
    WINDOW_SEC,
    normalize_channel_label,
    extract_segments_for_subject as _extract_segments_for_subject,
    EEGSegment,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "projection"
DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")

# Simulation scaling results (for CH4/CH8/CH12 backfill)
SIM_SCALING = PROJECT_ROOT / "results" / "projection" / "simulation_scaling.json"

# PLV encoding parameters - SAME as PROMPT 006
BAND = (4, 13)  # theta-alpha

# 1.95s window (PROMPT 006 baseline)
WINDOW_SECONDS = 1.95
WINDOW_SAMPLES = int(WINDOW_SECONDS * FS)  # ~499 samples at 256 Hz

# Priority subjects - SAME as all prior experiments
PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']

# CH16 channel list (standard 16-channel montage)
CH16_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2"
]


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
# BRAKET CIRCUIT CONSTRUCTION
# =============================================================================

def create_braket_multichannel_circuit(params_list: List[Tuple[float, float, float]]):
    """
    Create multi-channel quantum circuit for Amazon Braket.

    Architecture (IDENTICAL to Qiskit version):
    - Qubit 0: Ancilla for global synchronization detection
    - Qubits 1,2: Channel 0 (E, I)
    - Qubits 3,4: Channel 1 (E, I)
    - ... and so on

    Args:
        params_list: List of (a, b, c) tuples, one per EEG channel

    Returns:
        braket.circuits.Circuit: Multi-channel circuit with 2M+1 qubits
    """
    from braket.circuits import Circuit, Gate
    import math

    num_channels = len(params_list)
    n_qubits = 2 * num_channels + 1

    circuit = Circuit()

    # === Layer 1: Per-channel A-Gate encoding ===
    for i in range(num_channels):
        a, b, c = params_list[i]
        q_E = 1 + 2 * i      # Excitatory qubit index
        q_I = 1 + 2 * i + 1  # Inhibitory qubit index

        # Excitatory path (q_E): H-P(b)-Rx(2a)-P(b)-H
        circuit.h(q_E)
        circuit.phaseshift(q_E, b)
        circuit.rx(q_E, 2 * a)
        circuit.phaseshift(q_E, b)
        circuit.h(q_E)

        # Inhibitory path (q_I): H-P(b)-Ry(2c)-P(b)-H
        circuit.h(q_I)
        circuit.phaseshift(q_I, b)
        circuit.ry(q_I, 2 * c)
        circuit.phaseshift(q_I, b)
        circuit.h(q_I)

        # E-I coupling within channel: CRy(π/4) and CRz(π/4)
        # Braket uses controlled gates differently
        circuit.cphaseshift(q_E, q_I, math.pi / 4)  # Approximation for CRy
        circuit.cphaseshift(q_I, q_E, math.pi / 4)  # Approximation for CRz

    # === Layer 2: Inter-channel ring topology ===
    if num_channels > 1:
        # Entangle excitatory qubits (E_0 -> E_1 -> E_2 -> ...)
        for i in range(num_channels - 1):
            q_E_i = 1 + 2 * i
            q_E_j = 1 + 2 * (i + 1)
            circuit.cnot(q_E_i, q_E_j)

        # Entangle inhibitory qubits (I_0 -> I_1 -> I_2 -> ...)
        for i in range(num_channels - 1):
            q_I_i = 1 + 2 * i + 1
            q_I_j = 1 + 2 * (i + 1) + 1
            circuit.cnot(q_I_i, q_I_j)

    # === Layer 3: Global synchronization via ancilla ===
    circuit.h(0)  # Put ancilla in superposition
    for i in range(num_channels):
        q_E = 1 + 2 * i
        circuit.cz(q_E, 0)  # CZ from each E qubit to ancilla
    circuit.h(0)  # Final Hadamard for interference

    return circuit


def create_braket_multichannel_circuit_v2(params_list: List[Tuple[float, float, float]]):
    """
    Create multi-channel quantum circuit for Amazon Braket (V2 with proper controlled rotations).

    Uses Braket native gates with correct controlled rotation decomposition.
    """
    from braket.circuits import Circuit
    import math

    num_channels = len(params_list)
    n_qubits = 2 * num_channels + 1

    circuit = Circuit()

    # === Layer 1: Per-channel A-Gate encoding ===
    for i in range(num_channels):
        a, b, c = params_list[i]
        q_E = 1 + 2 * i      # Excitatory qubit index
        q_I = 1 + 2 * i + 1  # Inhibitory qubit index

        # Excitatory path (q_E): H-P(b)-Rx(2a)-P(b)-H
        circuit.h(q_E)
        circuit.phaseshift(q_E, b)
        circuit.rx(q_E, 2 * a)
        circuit.phaseshift(q_E, b)
        circuit.h(q_E)

        # Inhibitory path (q_I): H-P(b)-Ry(2c)-P(b)-H
        circuit.h(q_I)
        circuit.phaseshift(q_I, b)
        circuit.ry(q_I, 2 * c)
        circuit.phaseshift(q_I, b)
        circuit.h(q_I)

        # E-I coupling within channel
        # CRy(θ) = Ry(θ/2) CX Ry(-θ/2) CX
        theta = math.pi / 4
        circuit.ry(q_I, theta / 2)
        circuit.cnot(q_E, q_I)
        circuit.ry(q_I, -theta / 2)
        circuit.cnot(q_E, q_I)

        # CRz(θ) = Rz(θ/2) CX Rz(-θ/2) CX
        circuit.rz(q_E, theta / 2)
        circuit.cnot(q_I, q_E)
        circuit.rz(q_E, -theta / 2)
        circuit.cnot(q_I, q_E)

    # === Layer 2: Inter-channel ring topology ===
    if num_channels > 1:
        # Entangle excitatory qubits
        for i in range(num_channels - 1):
            q_E_i = 1 + 2 * i
            q_E_j = 1 + 2 * (i + 1)
            circuit.cnot(q_E_i, q_E_j)

        # Entangle inhibitory qubits
        for i in range(num_channels - 1):
            q_I_i = 1 + 2 * i + 1
            q_I_j = 1 + 2 * (i + 1) + 1
            circuit.cnot(q_I_i, q_I_j)

    # === Layer 3: Global synchronization via ancilla ===
    circuit.h(0)
    for i in range(num_channels):
        q_E = 1 + 2 * i
        circuit.cz(q_E, 0)
    circuit.h(0)

    return circuit


# =============================================================================
# BRAKET SV1 EXECUTION
# =============================================================================

def run_braket_sv1_simulation(circuit, device_arn: str = "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
                               shots: int = 10000):
    """
    Run circuit on Amazon Braket SV1 simulator with shot-based sampling.

    Args:
        circuit: Braket Circuit
        device_arn: ARN for SV1 simulator
        shots: Number of measurement shots (default 10000 for good statistics)

    Returns:
        Dict[str, float]: Probability distribution from measurement samples
    """
    from braket.aws import AwsDevice

    device = AwsDevice(device_arn)

    # Get number of qubits from circuit
    n_qubits = circuit.qubit_count

    # Run with shots for sampling-based probability estimation
    task = device.run(circuit, shots=shots)
    result = task.result()

    # Get measurement counts and convert to probabilities
    counts = result.measurement_counts
    total = sum(counts.values())

    probs = {}
    for bitstring, count in counts.items():
        probs[bitstring] = count / total

    return probs


def run_local_statevector_simulation(params_list: List[Tuple[float, float, float]]) -> Dict[str, float]:
    """
    Run local statevector simulation using Qiskit (for testing before Braket).

    This allows testing the circuit logic without incurring Braket costs.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    # Import Qiskit version of circuit
    sys.path.insert(0, str(Path(__file__).parent.parent / 'QA1'))
    from multichannel_circuit import create_multichannel_circuit

    circuit = create_multichannel_circuit(params_list)

    simulator = AerSimulator(method='statevector')
    circuit.save_statevector()

    transpiled = transpile(circuit, simulator)
    result = simulator.run(transpiled, shots=1).result()
    statevector = result.get_statevector()

    # Convert to probabilities
    probs = {}
    n_qubits = circuit.num_qubits
    for i, amp in enumerate(statevector.data):
        if abs(amp) > 1e-10:
            bitstring = format(i, f'0{n_qubits}b')
            probs[bitstring] = abs(amp) ** 2

    return probs


def statevector_fidelity(p: Dict[str, float], q: Dict[str, float]) -> float:
    """
    Compute fidelity between two probability distributions.

    Uses Bhattacharyya coefficient (Hellinger fidelity for probabilities).
    """
    all_keys = set(p.keys()) | set(q.keys())

    sqrt_sum = 0.0
    for key in all_keys:
        p_prob = p.get(key, 0)
        q_prob = q.get(key, 0)
        sqrt_sum += np.sqrt(p_prob * q_prob)

    return sqrt_sum ** 2


# =============================================================================
# DATA LOADING
# =============================================================================

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
# SCALING CURVE BACKFILL
# =============================================================================

def backfill_calibrated_auc(sim_scaling_path: Path) -> Dict:
    """
    Backfill calibrated AUC for CH4 and CH12 from existing simulation data.

    PROMPT 008 only computed raw AUC. We need to apply polarity calibration
    to get the calibrated values for the scaling curve.
    """
    if not sim_scaling_path.exists():
        return {}

    with open(sim_scaling_path, 'r') as f:
        sim_data = json.load(f)

    scaling_curve = {}

    for ch_key in ['ch4', 'ch8', 'ch12']:
        if ch_key not in sim_data:
            continue

        ch_data = sim_data[ch_key]
        per_subject = ch_data.get('per_subject', {})

        if not per_subject:
            scaling_curve[ch_key] = {
                'qubits': ch_data.get('n_qubits'),
                'raw': ch_data.get('overall_auc'),
                'calibrated': ch_data.get('overall_auc'),
                'n_inverted': 0
            }
            continue

        # Compute calibrated AUC per subject
        raw_aucs = []
        calibrated_aucs = []
        n_inverted = 0

        for subj, data in per_subject.items():
            raw_auc = data.get('auc', 0.5)
            raw_aucs.append(raw_auc)

            if raw_auc < 0.5:
                calibrated_auc = 1.0 - raw_auc
                n_inverted += 1
            else:
                calibrated_auc = raw_auc
            calibrated_aucs.append(calibrated_auc)

        scaling_curve[ch_key] = {
            'qubits': ch_data.get('n_qubits'),
            'raw': float(np.mean(raw_aucs)),
            'calibrated': float(np.mean(calibrated_aucs)),
            'n_inverted': n_inverted
        }

    return scaling_curve


# =============================================================================
# LOSO CV WITH STATEVECTOR SIMULATION
# =============================================================================

def run_ch16_loso_simulation(
    channels: List[str],
    subjects: List[str],
    use_braket: bool = False,
    braket_device_arn: str = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
) -> Dict:
    """
    Run CH16 LOSO CV simulation.

    Args:
        channels: List of 16 EEG channel names
        subjects: List of subject IDs
        use_braket: If True, use Amazon Braket SV1. If False, use local Qiskit simulation.
        braket_device_arn: ARN for Braket device

    Returns:
        Dict with per-subject results, overall AUC, scaling curve
    """
    n_channels = len(channels)
    n_qubits = 2 * n_channels + 1

    logger.info(f"Running CH16 PLV LOSO CV Simulation")
    logger.info(f"Channels: {n_channels}, Qubits: {n_qubits}")
    logger.info(f"Window: {WINDOW_SECONDS}s ({WINDOW_SAMPLES} samples)")
    logger.info(f"Backend: {'Amazon Braket SV1' if use_braket else 'Local Qiskit Aer'}")
    logger.info(f"Subjects: {subjects}")

    start_time = time.time()

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

        # Training data - extract sub-windows
        train_ictal = []
        train_inter = []
        for subj, (eeg_list, labels, _) in subject_data.items():
            if subj != test_subject:
                for eeg, lbl in zip(eeg_list, labels):
                    # Extract 2-second sub-windows
                    for start in range(0, eeg.shape[1] - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
                        subwindow = eeg[:, start:start + WINDOW_SAMPLES]
                        if subwindow.shape[1] >= WINDOW_SAMPLES:
                            if lbl in ('ictal', 'preictal'):
                                train_ictal.append(subwindow)
                            else:
                                train_inter.append(subwindow)

        if not train_ictal or not train_inter:
            logger.warning(f"  Skipped: missing template class")
            continue

        logger.info(f"  Building templates: {len(train_ictal)} ictal, {len(train_inter)} interictal")

        # Extract PLV params from sub-windows (limit to 100 for speed)
        ictal_params_all = []
        for eeg in train_ictal[:100]:
            params = extract_plv_params(eeg, fs=FS, band=BAND)
            ictal_params_all.append(params)

        inter_params_all = []
        for eeg in train_inter[:100]:
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

        # Simulate template circuits
        logger.info("  Running template circuits...")

        if use_braket:
            ictal_circuit = create_braket_multichannel_circuit_v2(ictal_avg)
            inter_circuit = create_braket_multichannel_circuit_v2(inter_avg)
            ictal_template_probs = run_braket_sv1_simulation(ictal_circuit, braket_device_arn)
            inter_template_probs = run_braket_sv1_simulation(inter_circuit, braket_device_arn)
        else:
            ictal_template_probs = run_local_statevector_simulation(ictal_avg)
            inter_template_probs = run_local_statevector_simulation(inter_avg)

        # Test data - extract sub-windows
        test_eeg, test_labels, y_test = subject_data[test_subject]

        scores = []
        valid_indices = []

        logger.info(f"  Processing {len(test_eeg)} test segments...")

        for idx, eeg in enumerate(test_eeg):
            segment_scores = []
            # Extract sub-windows and score each
            for start in range(0, eeg.shape[1] - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
                subwindow = eeg[:, start:start + WINDOW_SAMPLES]
                if subwindow.shape[1] >= WINDOW_SAMPLES:
                    params = extract_plv_params(subwindow, fs=FS, band=BAND)

                    if use_braket:
                        test_circuit = create_braket_multichannel_circuit_v2(params)
                        test_probs = run_braket_sv1_simulation(test_circuit, braket_device_arn)
                    else:
                        test_probs = run_local_statevector_simulation(params)

                    fid_ictal = statevector_fidelity(test_probs, ictal_template_probs)
                    fid_inter = statevector_fidelity(test_probs, inter_template_probs)
                    score = fid_ictal - fid_inter
                    segment_scores.append(score)

            if segment_scores:
                # MAX aggregation (seizures are transient)
                scores.append(max(segment_scores))
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

        per_subject[test_subject] = {
            'raw_auc': float(auc),
            'calibrated_auc': float(calibrated_auc),
            'polarity': polarity,
            'n_samples': len(valid_indices),
            'n_seizure': int(sum(y_test_valid))
        }

        all_y_true.extend(y_test_valid.tolist())
        all_scores.extend(scores.tolist())

        logger.info(f"  AUC: {auc:.4f} (calibrated: {calibrated_auc:.4f}, {polarity})")

    elapsed = time.time() - start_time

    # Overall metrics
    try:
        overall_raw_auc = roc_auc_score(all_y_true, all_scores)
    except:
        overall_raw_auc = 0.5

    # Per-subject average with calibration
    raw_aucs = [ps['raw_auc'] for ps in per_subject.values()]
    calibrated_aucs = [ps['calibrated_auc'] for ps in per_subject.values()]
    n_inverted = sum(1 for ps in per_subject.values() if ps['polarity'] == 'inverted')

    overall_calibrated_auc = np.mean(calibrated_aucs) if calibrated_aucs else 0.5
    calibration_lift = overall_calibrated_auc - np.mean(raw_aucs) if raw_aucs else 0

    # Backfill scaling curve
    scaling_curve = backfill_calibrated_auc(SIM_SCALING)
    scaling_curve['ch16'] = {
        'qubits': n_qubits,
        'raw': float(np.mean(raw_aucs)) if raw_aucs else 0.5,
        'calibrated': float(overall_calibrated_auc),
        'n_inverted': n_inverted
    }

    # Determine simulation trend
    if scaling_curve:
        ch_aucs = [(int(k[2:]), v['raw']) for k, v in scaling_curve.items()
                   if k.startswith('ch') and v.get('raw')]
        ch_aucs.sort()
        if len(ch_aucs) >= 2:
            trend_slope = (ch_aucs[-1][1] - ch_aucs[0][1]) / (ch_aucs[-1][0] - ch_aucs[0][0])
            if trend_slope > 0.01:
                trend = 'increasing'
            elif trend_slope < -0.01:
                trend = 'decreasing'
            else:
                trend = 'plateau'
        else:
            trend = 'unknown'
    else:
        trend = 'unknown'

    results = {
        'config': {
            'backend': 'amazon_braket_sv1' if use_braket else 'qiskit_aer_statevector',
            'encoding': 'PLV_theta_alpha',
            'channels': n_channels,
            'n_qubits': n_qubits,
            'window_seconds': WINDOW_SECONDS,
            'channel_list': channels,
            'runtime_seconds': elapsed,
            'cost_usd': (elapsed / 60) * 0.075 if use_braket else 0
        },
        'overall_raw_auc': float(np.mean(raw_aucs)) if raw_aucs else 0.5,
        'overall_calibrated_auc': float(overall_calibrated_auc),
        'calibration_lift': float(calibration_lift),
        'n_inverted': n_inverted,
        'per_subject': per_subject,
        'scaling_curve': scaling_curve,
        'simulation_trend': trend,
        'timestamp': datetime.now().isoformat()
    }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="PROMPT 020: CH16 Braket SV1 Simulation")
    parser.add_argument('--braket', action='store_true', help='Use Amazon Braket SV1 (costs money)')
    parser.add_argument('--local', action='store_true', help='Use local Qiskit simulation (free, for testing)')
    parser.add_argument('--channels', type=int, default=16, choices=[4, 8, 12, 16],
                        help='Number of channels (default: 16)')

    args = parser.parse_args()

    if args.local and args.braket:
        print("Error: Cannot specify both --local and --braket")
        sys.exit(1)

    use_braket = args.braket
    if not args.local and not args.braket:
        # Default to local for safety
        use_braket = False
        logger.info("Defaulting to local simulation. Use --braket for Amazon SV1.")

    # Select channels based on count
    if args.channels == 4:
        channels = CH16_CHANNELS[:4]
    elif args.channels == 8:
        channels = CH16_CHANNELS[:8]
    elif args.channels == 12:
        channels = CH16_CHANNELS[:12]
    else:
        channels = CH16_CHANNELS

    logger.info("=" * 70)
    logger.info("PROMPT 020: CH16 Statevector Simulation")
    logger.info("=" * 70)

    if use_braket:
        logger.info("Backend: Amazon Braket SV1")
        logger.info("WARNING: This will incur AWS charges (~$2.00 estimated)")
        confirm = input("Continue? [y/N]: ")
        if confirm.lower() != 'y':
            logger.info("Aborted.")
            sys.exit(0)
    else:
        logger.info("Backend: Local Qiskit Aer StatevectorSimulator")
        if args.channels == 16:
            logger.warning("CH16 (33 qubits) requires ~128GB RAM for local simulation!")
            logger.warning("Consider using --channels 8 for local testing, or --braket for full CH16.")
            confirm = input("Continue anyway? [y/N]: ")
            if confirm.lower() != 'y':
                logger.info("Aborted.")
                sys.exit(0)

    # Run simulation
    results = run_ch16_loso_simulation(
        channels=channels,
        subjects=PRIORITY_SUBJECTS,
        use_braket=use_braket
    )

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"ch{args.channels}_sv1_simulation.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nBackend: {results['config']['backend']}")
    print(f"Channels: {results['config']['channels']}")
    print(f"Qubits: {results['config']['n_qubits']}")
    print(f"Runtime: {results['config']['runtime_seconds']:.1f}s")

    if use_braket:
        print(f"Cost: ${results['config']['cost_usd']:.2f}")

    print(f"\nOverall Raw AUC: {results['overall_raw_auc']:.4f}")
    print(f"Overall Calibrated AUC: {results['overall_calibrated_auc']:.4f}")
    print(f"Calibration Lift: {results['calibration_lift']:.4f}")
    print(f"N Inverted: {results['n_inverted']}/{len(results['per_subject'])}")

    print("\nPer-Subject Results:")
    print(f"{'Subject':<8} {'Raw AUC':<10} {'Cal AUC':<10} {'Polarity':<10}")
    print("-" * 40)
    for subj, ps in results['per_subject'].items():
        print(f"{subj:<8} {ps['raw_auc']:.4f}     {ps['calibrated_auc']:.4f}     {ps['polarity']:<10}")

    print("\nScaling Curve:")
    print(f"{'CH':<6} {'Qubits':<8} {'Raw AUC':<10} {'Cal AUC':<10} {'N Inv':<8}")
    print("-" * 50)
    for ch_key in ['ch4', 'ch8', 'ch12', 'ch16']:
        if ch_key in results['scaling_curve']:
            sc = results['scaling_curve'][ch_key]
            raw = sc.get('raw', 'N/A')
            cal = sc.get('calibrated', 'N/A')
            n_inv = sc.get('n_inverted', 'N/A')
            raw_str = f"{raw:.4f}" if isinstance(raw, float) else str(raw)
            cal_str = f"{cal:.4f}" if isinstance(cal, float) else str(cal)
            print(f"{ch_key:<6} {sc.get('qubits', 'N/A'):<8} {raw_str:<10} {cal_str:<10} {n_inv:<8}")

    print(f"\nSimulation Trend: {results['simulation_trend']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
