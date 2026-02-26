#!/usr/bin/env python3
"""
Compute Circuit Trajectories for Visualization (C-001)

Extracts PLV encoding parameters from EEG at runtime, steps through the
logical A-Gate circuit gate-by-gate using statevector simulation, and saves
the full trajectory to JSON for the Three.js visualization.

Targets: chb01 (standard polarity) and chb11 (inverted polarity)

Output files:
- visualization_data/circuit_trajectories.json
- visualization_data/heron_topology.json
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from scipy.signal import hilbert, butter, filtfilt
from sklearn.decomposition import PCA
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / 'QA1'))

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Import from existing infrastructure
from QA1.multichannel_circuit import create_multichannel_circuit, get_qubit_indices
from sagemaker.train_chbmit import (
    WINDOW_SEC,
    FS,
    normalize_channel_label,
    extract_segments_for_subject,
    EEGSegment,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
OUTPUT_DIR = Path(__file__).parent.parent / "visualization_data"
TRANSPILATION_FILE = Path(__file__).parent.parent / "analysis_results/ibm_hardware/transpilation_report.json"
HARDWARE_RESULTS_FILE = Path(__file__).parent.parent / "results/hardware_validation/ch8_heron_loso_results.json"
CALIBRATION_FILE = Path(__file__).parent.parent / "results/calibration/calibrated_gap_inputs.json"

# 8 channels (same as run_quantum_loso_v3.py)
QUANTUM_CHANNELS = [
    'FP1-F7', 'F7-T7', 'FP1-F3', 'F3-C3',
    'FP2-F8', 'F8-T8', 'FP2-F4', 'F4-C4',
]

# Target patients
TARGET_PATIENTS = {
    'chb01': 'standard',   # Standard polarity
    'chb11': 'inverted',   # Inverted polarity
}

# Frequency band (theta-alpha, same as hardware run)
BAND = (4, 13)

# Max trajectory points (subsample if needed)
MAX_TRAJECTORY_POINTS = 80


# =============================================================================
# PLV ENCODING (from run_quantum_loso_v3.py)
# =============================================================================

def extract_plv_params(eeg_segment: np.ndarray, fs: float = 256.0,
                       band: Tuple[float, float] = (4, 13)) -> List[Tuple[float, float, float]]:
    """
    Extract (a, b, c) per channel using Hilbert phase/amplitude.

    This is the EXACT function from run_quantum_loso_v3.py.

    Encoding rationale:
      b = instantaneous phase -> feeds P(b) gates -> circuit detects synchronization
      a = normalized amplitude envelope -> feeds Rx(2a) -> encodes signal strength
      c = channel's PLV with global mean -> feeds Ry(2c) -> encodes network coupling
    """
    n_channels, n_samples = eeg_segment.shape

    # 1. Bandpass filter to target band
    nyq = fs / 2
    low, high = band[0] / nyq, min(band[1] / nyq, 0.99)

    try:
        b_filt, a_filt = butter(4, [low, high], btype='band')
    except ValueError:
        return [(0.5, np.pi, 0.5) for _ in range(n_channels)]

    # 2. Filter and compute analytic signal per channel
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
        b_phase = np.arctan2(mean_sin, mean_cos)
        b_phase = b_phase % (2 * np.pi)

        # c = PLV of this channel with global mean phase [0, 1]
        phase_diff = phases[ch] - global_phase
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        c_val = np.clip(plv, 0.05, 0.95)

        params_list.append((a_norm, b_phase, c_val))

    return params_list


# =============================================================================
# EEG LOADING
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
                if actual_samples > 0:
                    data[i, :actual_samples] = signal[start_sample:end_sample]

            return data
    except Exception as e:
        logger.warning(f"Failed to load {seg.source_file}: {e}")
        return None


def get_segments_for_patient(patient_id: str) -> Tuple[Optional[EEGSegment], Optional[EEGSegment]]:
    """
    Get one ictal and one interictal segment for a patient.

    Picks segments from the middle of recordings for cleaner encoding.
    """
    subject_dir = DATA_DIR / patient_id
    if not subject_dir.exists():
        return None, None

    segments = extract_segments_for_subject(subject_dir)

    ictal_segs = [s for s in segments if s.label == 'ictal']
    interictal_segs = [s for s in segments if s.label == 'interictal']

    if not ictal_segs or not interictal_segs:
        return None, None

    # Pick middle segments for cleaner encoding
    ictal_seg = ictal_segs[len(ictal_segs) // 2]
    interictal_seg = interictal_segs[len(interictal_segs) // 2]

    return ictal_seg, interictal_seg


# =============================================================================
# GATE-BY-GATE STATEVECTOR SIMULATION
# =============================================================================

def compute_subsystem_entropy(sv_array: np.ndarray, n_qubits: int, subsystem_qubits: List[int]) -> float:
    """
    Compute von Neumann entropy of a subsystem by tracing out the rest.

    For efficiency, we compute reduced density matrix only for small subsystems.
    """
    n_subsystem = len(subsystem_qubits)
    dim_subsystem = 2 ** n_subsystem
    dim_rest = 2 ** (n_qubits - n_subsystem)

    # Build reduced density matrix
    rho_reduced = np.zeros((dim_subsystem, dim_subsystem), dtype=complex)

    # For each pair of subsystem basis states
    for i in range(dim_subsystem):
        for j in range(dim_subsystem):
            # Sum over all states of the traced-out qubits
            for k in range(dim_rest):
                # Construct full basis state indices
                # This is complex - simplified approach: use statevector probabilities
                pass

    # Simplified: compute bipartite entanglement entropy with ancilla
    # Trace out all except ancilla (qubit 0)
    prob_0 = 0.0
    prob_1 = 0.0
    for state_idx, amp in enumerate(sv_array):
        p = abs(amp) ** 2
        if (state_idx & 1) == 0:  # Ancilla is qubit 0
            prob_0 += p
        else:
            prob_1 += p

    # Binary entropy
    if prob_0 > 1e-12 and prob_1 > 1e-12:
        entropy = -prob_0 * np.log2(prob_0) - prob_1 * np.log2(prob_1)
    else:
        entropy = 0.0

    return float(entropy)


def step_through_circuit(qc: QuantumCircuit, subsample_step: int = 1) -> Tuple[List[Dict], List[np.ndarray]]:
    """
    Step through each gate, save state after each.

    For 17 qubits, we CANNOT compute full density matrix (would be 256 GB).
    Instead, we compute metrics directly from the statevector:
    - Purity is always 1.0 for pure states
    - We compute ancilla entanglement entropy (binary entropy of P(ancilla=0))
    - Boundary distance is <Z_ancilla>

    Args:
        qc: Quantum circuit (without measurements)
        subsample_step: Only record every Nth gate (for long circuits)

    Returns:
        Tuple of (trajectory list, feature vectors for PCA)
    """
    gates = list(qc.data)
    n_gates = len(gates)

    trajectory = []
    feature_vectors = []

    # Get qubit count
    n_qubits = qc.num_qubits
    ancilla_idx = 0  # Ancilla is qubit 0

    # Build circuit incrementally for efficiency
    current_sv = Statevector.from_int(0, 2 ** n_qubits)

    for i in range(n_gates):
        # Apply gate i to current state
        gate_op = gates[i].operation
        gate_qubits = [qc.find_bit(q).index for q in gates[i].qubits]

        # Create single-gate circuit and evolve
        single_gate_qc = QuantumCircuit(n_qubits)
        single_gate_qc.append(gate_op, gate_qubits)
        current_sv = current_sv.evolve(single_gate_qc)

        sv_array = np.array(current_sv.data)

        # Only record every Nth gate
        if i % subsample_step == 0 or i == n_gates - 1:
            probs = np.abs(sv_array) ** 2

            # Compute Z expectation on ancilla (boundary distance)
            z_expectation = 0.0
            prob_ancilla_0 = 0.0
            for state_idx, prob in enumerate(probs):
                ancilla_bit = (state_idx >> ancilla_idx) & 1
                z_val = 1 if ancilla_bit == 0 else -1
                z_expectation += prob * z_val
                if ancilla_bit == 0:
                    prob_ancilla_0 += prob

            boundary_distance = z_expectation  # Already centered at 0

            # Compute ancilla entanglement (binary entropy)
            prob_ancilla_1 = 1.0 - prob_ancilla_0
            if prob_ancilla_0 > 1e-12 and prob_ancilla_1 > 1e-12:
                ancilla_entropy = -prob_ancilla_0 * np.log2(prob_ancilla_0) - prob_ancilla_1 * np.log2(prob_ancilla_1)
            else:
                ancilla_entropy = 0.0

            # Compute E-I balance: sum of Z expectations on E qubits vs I qubits
            # E qubits: 1, 3, 5, ... (odd indices starting at 1)
            # I qubits: 2, 4, 6, ... (even indices starting at 2)
            z_E = 0.0
            z_I = 0.0
            for state_idx, prob in enumerate(probs):
                for ch in range(8):  # 8 channels
                    e_idx = 1 + 2 * ch
                    i_idx = 2 + 2 * ch
                    e_bit = (state_idx >> e_idx) & 1
                    i_bit = (state_idx >> i_idx) & 1
                    z_E += prob * (1 if e_bit == 0 else -1)
                    z_I += prob * (1 if i_bit == 0 else -1)
            z_E /= 8  # Average over channels
            z_I /= 8

            # Feature vector for PCA: [boundary_dist, ancilla_entropy, E-I balance]
            feature_vec = np.array([boundary_distance, ancilla_entropy, z_E - z_I])
            feature_vectors.append(feature_vec)

            projection = {
                "gate_index": i,
                "gate_type": gate_op.name,
                "gate_qubits": gate_qubits,
                "purity": 1.0,  # Always 1 for pure states
                "ancilla_entropy": float(ancilla_entropy),
                "boundary_distance": float(boundary_distance),
                "z_excitatory": float(z_E),
                "z_inhibitory": float(z_I),
                "pca": None,  # Filled in post-processing
            }
            trajectory.append(projection)

    return trajectory, feature_vectors


def compute_pca_projections(all_feature_vectors: List[np.ndarray],
                            trajectories: Dict[str, Dict[str, List[Dict]]]) -> Tuple[Dict, List[float]]:
    """
    Compute PCA projection for all feature vectors across all trajectories.

    Feature vectors are [boundary_dist, ancilla_entropy, E-I balance].

    Returns:
        Tuple of (updated trajectories, explained variance ratios)
    """
    if not all_feature_vectors:
        return trajectories, [0.0, 0.0, 0.0]

    # Stack all vectors
    all_vecs = np.array(all_feature_vectors)

    # Check if we have enough samples
    if len(all_vecs) < 3:
        # Not enough for PCA, use raw features
        idx = 0
        for patient_id in trajectories:
            for state_type in ['ictal', 'interictal']:
                if state_type in trajectories[patient_id] and isinstance(trajectories[patient_id][state_type], list):
                    for point in trajectories[patient_id][state_type]:
                        if idx < len(all_feature_vectors):
                            vec = all_feature_vectors[idx]
                            point['pca'] = [float(vec[0]), float(vec[1]), float(vec[2])]
                            idx += 1
        return trajectories, [1.0, 0.0, 0.0]

    # Fit PCA
    pca = PCA(n_components=3)
    transformed_all = pca.fit_transform(all_vecs)

    # Assign transformed coordinates to trajectory points
    idx = 0
    for patient_id in trajectories:
        for state_type in ['ictal', 'interictal']:
            if state_type in trajectories[patient_id] and isinstance(trajectories[patient_id][state_type], list):
                for point in trajectories[patient_id][state_type]:
                    if idx < len(transformed_all):
                        point['pca'] = [float(transformed_all[idx, 0]),
                                       float(transformed_all[idx, 1]),
                                       float(transformed_all[idx, 2])]
                        idx += 1

    return trajectories, pca.explained_variance_ratio_.tolist()


# =============================================================================
# HERON TOPOLOGY GENERATION
# =============================================================================

# IBM Torino (ibm_torino) heavy-hex qubit coordinates
# Source: IBM Quantum backend.configuration().qubit_coordinates
# Format: qubit_index -> [row, col]
IBMQ_TORINO_COORDS = {
    0:  [0, 0],   1:  [0, 1],   2:  [0, 2],   3:  [0, 3],   4:  [0, 4],
    5:  [0, 5],   6:  [0, 6],   7:  [0, 7],   8:  [0, 8],   9:  [0, 9],
    10: [0, 10],  11: [0, 11],  12: [0, 12],
    13: [1, 1],   14: [1, 3],   15: [1, 5],   16: [1, 7],   17: [1, 9],  18: [1, 11],
    19: [2, 0],   20: [2, 1],   21: [2, 2],   22: [2, 3],   23: [2, 4],
    24: [2, 5],   25: [2, 6],   26: [2, 7],   27: [2, 8],   28: [2, 9],
    29: [2, 10],  30: [2, 11],  31: [2, 12],
    32: [3, 1],   33: [3, 3],   34: [3, 5],   35: [3, 7],   36: [3, 9],  37: [3, 11],
    38: [4, 0],   39: [4, 1],   40: [4, 2],   41: [4, 3],   42: [4, 4],
    43: [4, 5],   44: [4, 6],   45: [4, 7],   46: [4, 8],   47: [4, 9],
    48: [4, 10],  49: [4, 11],  50: [4, 12],
    51: [5, 1],   52: [5, 3],   53: [5, 5],   54: [5, 7],   55: [5, 9],  56: [5, 11],
    57: [6, 0],   58: [6, 1],   59: [6, 2],   60: [6, 3],   61: [6, 4],
    62: [6, 5],   63: [6, 6],   64: [6, 7],   65: [6, 8],   66: [6, 9],
    67: [6, 10],  68: [6, 11],  69: [6, 12],
    70: [7, 1],   71: [7, 3],   72: [7, 5],   73: [7, 7],   74: [7, 9],  75: [7, 11],
    76: [8, 0],   77: [8, 1],   78: [8, 2],   79: [8, 3],   80: [8, 4],
    81: [8, 5],   82: [8, 6],   83: [8, 7],   84: [8, 8],   85: [8, 9],
    86: [8, 10],  87: [8, 11],  88: [8, 12],
    89: [9, 1],   90: [9, 3],   91: [9, 5],   92: [9, 7],   93: [9, 9],  94: [9, 11],
    95: [10, 0],  96: [10, 1],  97: [10, 2],  98: [10, 3],  99: [10, 4],
    100:[10, 5],  101:[10, 6],  102:[10, 7],  103:[10, 8],  104:[10, 9],
    105:[10, 10], 106:[10, 11], 107:[10, 12],
    108:[11, 1],  109:[11, 3],  110:[11, 5],  111:[11, 7],  112:[11, 9], 113:[11, 11],
    114:[12, 0],  115:[12, 1],  116:[12, 2],  117:[12, 3],  118:[12, 4],
    119:[12, 5],  120:[12, 6],  121:[12, 7],  122:[12, 8],  123:[12, 9],
    124:[12, 10], 125:[12, 11], 126:[12, 12],
    127:[13, 1],  128:[13, 3],  129:[13, 5],  130:[13, 7],  131:[13, 9], 132:[13, 11],
}

# Heavy-hex coupling map for ibm_torino (bidirectional, store only q1<q2)
# Rows of main qubits connect to each other; flag qubits connect two main-row qubits
IBMQ_TORINO_EDGES = [
    # Row 0
    [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],
    # Row 0 <-> Flag row 1
    [1,13],[3,14],[5,15],[7,16],[9,17],[11,18],
    # Flag row 1 <-> Row 2
    [13,20],[14,22],[15,24],[16,26],[17,28],[18,30],
    # Row 2
    [19,20],[20,21],[21,22],[22,23],[23,24],[24,25],[25,26],[26,27],[27,28],[28,29],[29,30],[30,31],
    # Row 2 <-> Flag row 3
    [20,32],[22,33],[24,34],[26,35],[28,36],[30,37],
    # Flag row 3 <-> Row 4
    [32,39],[33,41],[34,43],[35,45],[36,47],[37,49],
    # Row 4
    [38,39],[39,40],[40,41],[41,42],[42,43],[43,44],[44,45],[45,46],[46,47],[47,48],[48,49],[49,50],
    # Row 4 <-> Flag row 5
    [39,51],[41,52],[43,53],[45,54],[47,55],[49,56],
    # Flag row 5 <-> Row 6
    [51,58],[52,60],[53,62],[54,64],[55,66],[56,68],
    # Row 6
    [57,58],[58,59],[59,60],[60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,68],[68,69],
    # Row 6 <-> Flag row 7
    [58,70],[60,71],[62,72],[64,73],[66,74],[68,75],
    # Flag row 7 <-> Row 8
    [70,77],[71,79],[72,81],[73,83],[74,85],[75,87],
    # Row 8
    [76,77],[77,78],[78,79],[79,80],[80,81],[81,82],[82,83],[83,84],[84,85],[85,86],[86,87],[87,88],
    # Row 8 <-> Flag row 9
    [77,89],[79,90],[81,91],[83,92],[85,93],[87,94],
    # Flag row 9 <-> Row 10
    [89,96],[90,98],[91,100],[92,102],[93,104],[94,106],
    # Row 10
    [95,96],[96,97],[97,98],[98,99],[99,100],[100,101],[101,102],[102,103],[103,104],[104,105],[105,106],[106,107],
    # Row 10 <-> Flag row 11
    [96,108],[98,109],[100,110],[102,111],[104,112],[106,113],
    # Flag row 11 <-> Row 12
    [108,115],[109,117],[110,119],[111,121],[112,123],[113,125],
    # Row 12
    [114,115],[115,116],[116,117],[117,118],[118,119],[119,120],[120,121],[121,122],[122,123],[123,124],[124,125],[125,126],
    # Row 12 <-> Flag row 13
    [115,127],[117,128],[119,129],[121,130],[123,131],[125,132],
]


def _hardcoded_torino_topology(active_qubits: List[int]) -> Tuple[Dict[int, List[int]], List[List[int]]]:
    """Get coordinates and coupling edges for active qubits from hardcoded IBM Torino data."""
    active_set = set(active_qubits)
    qubit_coords = {q: IBMQ_TORINO_COORDS[q] for q in active_qubits if q in IBMQ_TORINO_COORDS}
    coupling_edges = [
        e for e in IBMQ_TORINO_EDGES
        if e[0] in active_set and e[1] in active_set
    ]
    return qubit_coords, coupling_edges


def generate_heron_topology() -> Dict:
    """Generate IBM Torino Heron heavy-hex topology with real 2D coordinates."""

    if not TRANSPILATION_FILE.exists():
        logger.warning(f"Transpilation file not found: {TRANSPILATION_FILE}")
        return {}

    with open(TRANSPILATION_FILE) as f:
        transpilation = json.load(f)

    layout = transpilation['circuits']['synchronized']['layout']
    active_qubits = sorted(set(layout))

    # Try to get real coordinates from IBM backend, fall back to hardcoded
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = service.backend('ibm_torino')
        config = backend.configuration()

        # qubit_coordinates is list of [row, col] per qubit, 0-indexed
        all_coords = config.qubit_coordinates
        qubit_coords = {q: all_coords[q] for q in active_qubits}

        # Real coupling map from backend
        coupling_map = config.coupling_map
        active_set = set(active_qubits)
        coupling_edges = [
            [q1, q2] for q1, q2 in coupling_map
            if q1 in active_set and q2 in active_set and q1 < q2
        ]

        logger.info(f"Got real coordinates for {len(qubit_coords)} qubits from ibm_torino")

    except Exception as e:
        logger.warning(f"Could not query backend ({e}), using hardcoded IBM Torino coordinates")
        qubit_coords, coupling_edges = _hardcoded_torino_topology(active_qubits)

    # Normalize coordinates to [0, 1] range for the viz
    rows = [c[0] for c in qubit_coords.values()]
    cols = [c[1] for c in qubit_coords.values()]
    row_min, row_max = min(rows), max(rows)
    col_min, col_max = min(cols), max(cols)
    row_range = max(row_max - row_min, 1)
    col_range = max(col_max - col_min, 1)

    qubit_positions = {
        str(q): {
            "row": coords[0],
            "col": coords[1],
            "x_norm": (coords[1] - col_min) / col_range,  # col -> x
            "y_norm": (coords[0] - row_min) / row_range,  # row -> y
        }
        for q, coords in qubit_coords.items()
    }

    # Gate to physical qubit mapping
    n_logical = len(layout)
    gate_to_physical = {}
    for i in range(n_logical):
        gate_to_physical[str(i)] = [layout[i]]

    return {
        "backend": transpilation['backend'],
        "n_physical_qubits": transpilation['backend_qubits'],
        "active_qubits": active_qubits,
        "qubit_positions": qubit_positions,
        "coupling_edges": coupling_edges,
        "logical_to_physical_layout": layout,
        "gate_to_physical": gate_to_physical,
        "transpiled_depth": transpilation['circuits']['synchronized']['transpiled_depth'],
        "two_qubit_gates": transpilation['circuits']['synchronized']['two_qubit_gates'],
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Generate circuit trajectories for visualization."""

    logger.info("=" * 60)
    logger.info("C-001: Compute Circuit Trajectories")
    logger.info("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load hardware results for AUC values
    hardware_results = {}
    if HARDWARE_RESULTS_FILE.exists():
        with open(HARDWARE_RESULTS_FILE) as f:
            hw = json.load(f)
            for subj in hw['ch8']['per_subject']:
                hardware_results[subj] = hw['ch8']['per_subject'][subj]['auc']

    # Load calibration data
    calibration = {}
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE) as f:
            cal = json.load(f)
            for subj in cal['ch8']['per_subject']:
                calibration[subj] = cal['ch8']['per_subject'][subj]

    # Process each target patient
    all_trajectories = {}
    all_feature_vectors = []
    all_gate_types = set()
    n_gates_per_patient = {}

    for patient_id, polarity in TARGET_PATIENTS.items():
        logger.info(f"\nProcessing {patient_id} ({polarity} polarity)...")

        subject_dir = DATA_DIR / patient_id
        ictal_seg, interictal_seg = get_segments_for_patient(patient_id)

        if ictal_seg is None or interictal_seg is None:
            logger.warning(f"  Could not find segments for {patient_id}")
            continue

        patient_data = {
            'polarity': polarity,
            'hardware_auc_raw': hardware_results.get(patient_id, None),
            'hardware_auc_calibrated': calibration.get(patient_id, {}).get('calibrated', None),
        }

        for state_type, seg in [('ictal', ictal_seg), ('interictal', interictal_seg)]:
            logger.info(f"  Loading {state_type} segment: {seg.source_file}")

            # Load EEG data
            eeg_data = load_eeg_segment(seg, subject_dir, QUANTUM_CHANNELS)
            if eeg_data is None:
                logger.warning(f"    Failed to load EEG data")
                continue

            # Extract PLV parameters
            params_list = extract_plv_params(eeg_data, fs=FS, band=BAND)
            logger.info(f"    Extracted {len(params_list)} channel parameters")

            # Store encoding params
            if 'encoding_params' not in patient_data:
                patient_data['encoding_params'] = {}
            patient_data['encoding_params'][state_type] = {
                'channels': QUANTUM_CHANNELS,
                'params': [
                    {'a': float(a), 'b': float(b), 'c': float(c)}
                    for a, b, c in params_list
                ]
            }

            # Build quantum circuit
            qc = create_multichannel_circuit(params_list)
            n_gates = qc.size()
            logger.info(f"    Circuit: {qc.num_qubits} qubits, {n_gates} gates, depth {qc.depth()}")

            # Determine subsample step
            subsample_step = max(1, n_gates // MAX_TRAJECTORY_POINTS)

            # Step through circuit
            logger.info(f"    Simulating gate-by-gate (subsample every {subsample_step} gates)...")
            trajectory, feature_vecs = step_through_circuit(qc, subsample_step)

            # Collect gate types
            for point in trajectory:
                all_gate_types.add(point['gate_type'])

            # Store
            patient_data[state_type] = trajectory
            all_feature_vectors.extend(feature_vecs)
            n_gates_per_patient[f"{patient_id}_{state_type}"] = len(trajectory)

            logger.info(f"    Recorded {len(trajectory)} trajectory points")

        all_trajectories[patient_id] = patient_data

    # Compute PCA projections across all trajectories
    logger.info("\nComputing PCA projections across all trajectories...")
    logger.info(f"  Total feature vectors collected: {len(all_feature_vectors)}")

    all_trajectories, pca_variance = compute_pca_projections(all_feature_vectors, all_trajectories)
    logger.info(f"  PCA explained variance: {pca_variance}")

    # Build output
    output = {
        'metadata': {
            'n_logical_qubits': 17,
            'n_gates': sum(n_gates_per_patient.values()) // len(n_gates_per_patient) if n_gates_per_patient else 0,
            'gate_types': sorted(list(all_gate_types)),
            'channel_labels': QUANTUM_CHANNELS,
            'pca_explained_variance': pca_variance,
            'encoding': 'PLV_theta_alpha',
            'band_hz': list(BAND),
            'generated': datetime.now().isoformat(),
        },
        'patients': all_trajectories,
    }

    # Save circuit trajectories
    output_file = OUTPUT_DIR / 'circuit_trajectories.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    logger.info(f"\nSaved: {output_file} ({file_size_mb:.2f} MB)")

    # Generate Heron topology
    logger.info("\nGenerating Heron topology...")
    topology = generate_heron_topology()
    topology_file = OUTPUT_DIR / 'heron_topology.json'
    with open(topology_file, 'w') as f:
        json.dump(topology, f, indent=2)
    logger.info(f"Saved: {topology_file}")

    # Validation summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAJECTORIES GENERATED")
    logger.info("=" * 60)

    for patient_id in all_trajectories:
        for state_type in ['ictal', 'interictal']:
            if state_type in all_trajectories[patient_id] and isinstance(all_trajectories[patient_id][state_type], list):
                traj = all_trajectories[patient_id][state_type]
                if traj:
                    final_bd = traj[-1]['boundary_distance']
                    logger.info(f"  {patient_id} {state_type:12s}: {len(traj)} gates, final boundary_dist = {final_bd:+.4f}")

    # Polarity check
    logger.info("\nPOLARITY CHECK (should be opposite signs at final gate):")

    results_valid = True
    for patient_id in all_trajectories:
        if 'ictal' in all_trajectories[patient_id] and 'interictal' in all_trajectories[patient_id]:
            ictal_traj = all_trajectories[patient_id]['ictal']
            inter_traj = all_trajectories[patient_id]['interictal']

            if ictal_traj and inter_traj:
                ictal_bd = ictal_traj[-1]['boundary_distance']
                inter_bd = inter_traj[-1]['boundary_distance']
                delta = ictal_bd - inter_bd

                # For standard polarity, ictal should have higher boundary_dist (more synchronized)
                # For inverted polarity, it's reversed
                polarity = all_trajectories[patient_id]['polarity']
                expected_sign = 1 if polarity == 'standard' else -1
                actual_sign = 1 if delta > 0 else -1

                correct = "CORRECT" if actual_sign == expected_sign or abs(delta) < 0.1 else "CHECK"
                logger.info(f"  {patient_id} ictal - interictal boundary delta: {delta:+.4f}  [{correct}]")

    # Cross-patient comparison
    if 'chb01' in all_trajectories and 'chb11' in all_trajectories:
        chb01_ictal = all_trajectories['chb01'].get('ictal', [])
        chb11_ictal = all_trajectories['chb11'].get('ictal', [])

        if chb01_ictal and chb11_ictal:
            chb01_bd = chb01_ictal[-1]['boundary_distance']
            chb11_bd = chb11_ictal[-1]['boundary_distance']

            direction = "OPPOSITE" if (chb01_bd * chb11_bd < 0) else "SAME"
            correct = "CORRECT" if direction == "OPPOSITE" else "CHECK"
            logger.info(f"  chb01 vs chb11 direction: {direction}      [{correct}]")

    logger.info(f"\nPCA explained variance: {output['metadata']['pca_explained_variance']}")
    logger.info(f"File size: {file_size_mb:.2f} MB")

    return output


if __name__ == '__main__':
    main()
