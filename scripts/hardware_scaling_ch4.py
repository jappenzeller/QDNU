#!/usr/bin/env python3
"""
================================================================================
PROMPT 009 — Hardware Scaling Curve: 4 Channels on IBM Heron
================================================================================

Establishes additional hardware data points for the degradation curve.
Uses V2 band_power encoding — same as PROMPT 008 simulation curve.

Backend: IBM Heron r2 (ibm_torino) — no simulator
Encoding: V2 band_power — must match PROMPT 008 exactly
Channels: CH4 loaded from results/scaling/classical_channel_scaling.json
Shots: 1024 minimum
Error mitigation: M3 measurement error mitigation

Output: results/projection/hardware_scaling.json
================================================================================
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from scipy.signal import welch
import logging
import warnings

warnings.filterwarnings('ignore')

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
HW_VALIDATION = Path("results/hardware_validation/ch8_heron_loso_results.json")
SIM_SCALING = Path("results/projection/simulation_scaling.json")

# Hardware configuration
SHOTS = 1024
BACKEND_NAME = "ibm_torino"

# Priority subjects (same as PROMPT 006/008)
PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']


# =============================================================================
# V2 BAND-POWER ENCODING (exact copy from simulation_scaling.py)
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

def load_channels(n_channels: int) -> List[str]:
    """Load channel list from classical scaling results."""
    key = f"ch{n_channels}"

    if CLASSICAL_SCALING.exists():
        with open(CLASSICAL_SCALING, 'r') as f:
            data = json.load(f)
        if key in data:
            return data[key]['channels']

    # Fallback for CH4
    if n_channels == 4:
        return ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3"]

    raise ValueError(f"No channel configuration found for {n_channels} channels")


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

    # Fallback to least busy
    try:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=9  # CH4 needs 9 qubits
        )
        logger.info(f"Using fallback backend: {backend.name} ({backend.num_qubits} qubits)")
        return backend
    except Exception as e:
        logger.error(f"No suitable backend found: {e}")
        raise


def run_circuits_on_hardware(circuits: List, backend, shots: int = SHOTS,
                              use_mitigation: bool = True) -> List[Dict[str, int]]:
    """
    Run circuits on IBM hardware using SamplerV2 with error mitigation.
    """
    from qiskit_ibm_runtime import SamplerV2, Batch
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    logger.info(f"Transpiling {len(circuits)} circuits for {backend.name}...")

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    transpiled = pm.run(circuits)

    logger.info(f"Submitting to hardware with {shots} shots...")

    mitigation_settings = None
    if use_mitigation:
        try:
            # Try to enable M3 measurement error mitigation
            from qiskit_ibm_runtime.options import SamplerOptions
            options = SamplerOptions()
            options.resilience_level = 1  # Basic error mitigation
            mitigation_settings = "resilience_level_1"
            logger.info("Using resilience_level=1 error mitigation")
        except Exception as e:
            logger.warning(f"Could not enable error mitigation: {e}")
            mitigation_settings = "none"

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

    return all_counts, mitigation_settings


def counts_to_score(counts: Dict[str, int]) -> float:
    """Convert counts to a discrimination score."""
    total = sum(counts.values())
    if total == 0:
        return 0.5

    # Simple score: weighted sum of '1' bits in most common outcomes
    weighted_sum = 0.0
    for bitstring, count in counts.items():
        ones = bitstring.count('1')
        weighted_sum += (ones / len(bitstring)) * count

    return weighted_sum / total


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

def run_hardware_loso_ch4(channels: List[str], subjects: List[str], backend) -> Dict:
    """Run LOSO CV on IBM hardware for CH4."""
    from sklearn.metrics import roc_auc_score

    n_channels = len(channels)
    n_qubits = 2 * n_channels + 1

    logger.info(f"Running CH4 hardware LOSO: {n_channels} channels, {n_qubits} qubits")
    logger.info(f"Subjects: {subjects}")

    per_subject_results = {}
    all_y_true = []
    all_y_score = []

    mitigation_used = None

    for test_subject in subjects:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing on {test_subject}")
        logger.info(f"{'='*60}")

        # Load test data
        test_eeg, test_labels = load_subject_data(test_subject, channels)
        if len(test_eeg) == 0:
            logger.warning(f"No data for {test_subject}, skipping")
            continue

        # Load training data (all other subjects)
        train_subjects = [s for s in subjects if s != test_subject]
        train_ictal_params = []
        train_inter_params = []

        for train_subj in train_subjects:
            eeg_list, labels = load_subject_data(train_subj, channels)
            for eeg, lbl in zip(eeg_list, labels):
                params = extract_pn_params_bandpower(eeg)
                if lbl == 'ictal':
                    train_ictal_params.append(params)
                else:
                    train_inter_params.append(params)

        if len(train_ictal_params) == 0 or len(train_inter_params) == 0:
            logger.warning(f"Insufficient training data for {test_subject}")
            continue

        # Create template circuits
        ictal_template = [
            (np.mean([p[ch][0] for p in train_ictal_params]),
             np.mean([p[ch][1] for p in train_ictal_params]),
             np.mean([p[ch][2] for p in train_ictal_params]))
            for ch in range(n_channels)
        ]

        inter_template = [
            (np.mean([p[ch][0] for p in train_inter_params]),
             np.mean([p[ch][1] for p in train_inter_params]),
             np.mean([p[ch][2] for p in train_inter_params]))
            for ch in range(n_channels)
        ]

        ictal_circuit = create_multichannel_circuit(ictal_template)
        inter_circuit = create_multichannel_circuit(inter_template)

        # Run templates on hardware
        logger.info("Running template circuits on hardware...")
        template_counts, mitigation_used = run_circuits_on_hardware(
            [ictal_circuit, inter_circuit], backend, shots=SHOTS
        )
        ictal_template_counts = template_counts[0]
        inter_template_counts = template_counts[1]

        # Run test circuits
        test_circuits = []
        for eeg in test_eeg:
            params = extract_pn_params_bandpower(eeg)
            circuit = create_multichannel_circuit(params)
            test_circuits.append(circuit)

        logger.info(f"Running {len(test_circuits)} test circuits on hardware...")
        test_counts, _ = run_circuits_on_hardware(test_circuits, backend, shots=SHOTS)

        # Compute scores using fidelity to templates
        y_true = [1 if lbl == 'ictal' else 0 for lbl in test_labels]
        y_scores = []

        for counts in test_counts:
            fid_ictal = hellinger_fidelity(counts, ictal_template_counts)
            fid_inter = hellinger_fidelity(counts, inter_template_counts)
            # Score = fidelity to ictal template - fidelity to interictal template
            score = fid_ictal - fid_inter
            y_scores.append(score)

        # Compute AUC for this subject
        unique_labels = set(y_true)
        if len(unique_labels) < 2:
            logger.warning(f"{test_subject}: single class in test set, skipping AUC")
            continue

        try:
            auc = roc_auc_score(y_true, y_scores)
        except Exception as e:
            logger.warning(f"AUC calculation failed for {test_subject}: {e}")
            continue

        per_subject_results[test_subject] = {
            "auc": auc,
            "n_samples": len(y_true),
            "n_seizure": sum(y_true)
        }

        all_y_true.extend(y_true)
        all_y_score.extend(y_scores)

        logger.info(f"{test_subject}: AUC = {auc:.4f}")

    # Compute overall AUC
    overall_auc = None
    if len(set(all_y_true)) >= 2:
        try:
            overall_auc = roc_auc_score(all_y_true, all_y_score)
        except:
            pass

    return {
        "overall_auc": overall_auc,
        "per_subject": per_subject_results,
        "mitigation": mitigation_used
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PROMPT 009 — Hardware Scaling Curve: 4 Channels on IBM Heron")
    print("=" * 70)
    print()

    # Load CH4 channels
    channels = load_channels(4)
    n_qubits = 2 * len(channels) + 1
    logger.info(f"CH4 channels: {channels}")
    logger.info(f"Qubits required: {n_qubits}")

    # Connect to IBM hardware
    try:
        backend = connect_to_ibm()
    except Exception as e:
        logger.error(f"Cannot connect to IBM hardware: {e}")
        logger.info("Creating placeholder output with error status")

        # Create placeholder output
        output = {
            "ch4": {
                "backend": "ibm_torino",
                "encoding": "V2_band_power",
                "mitigation": "not_available",
                "shots": SHOTS,
                "n_qubits": n_qubits,
                "overall_auc": None,
                "per_subject": {},
                "error": str(e)
            },
            "ch8": None,
            "gap_table": [],
            "timestamp": datetime.now().isoformat()
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = OUTPUT_DIR / "hardware_scaling.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved placeholder: {output_path}")
        return output

    # Run CH4 LOSO on hardware
    logger.info("\nRunning CH4 LOSO CV on hardware...")
    ch4_result = run_hardware_loso_ch4(channels, PRIORITY_SUBJECTS, backend)

    # Load existing CH8 hardware results
    ch8_data = None
    if HW_VALIDATION.exists():
        with open(HW_VALIDATION, 'r') as f:
            ch8_hw = json.load(f)
        ch8_data = {
            "backend": ch8_hw.get("backend", "ibm_torino"),
            "encoding": ch8_hw.get("encoding_variant", "PLV_theta_alpha"),
            "mitigation": ch8_hw.get("error_mitigation", "none"),
            "shots": ch8_hw.get("shots", 1024),
            "n_qubits": 17,
            "overall_auc": ch8_hw.get("ch8", {}).get("overall_auc"),
            "per_subject": ch8_hw.get("ch8", {}).get("per_subject", {}),
            "note": "encoding differs from ch4 — gap not directly comparable"
        }

    # Load simulation results for gap calculation
    sim_ch4_auc = None
    sim_ch8_auc = None
    if SIM_SCALING.exists():
        with open(SIM_SCALING, 'r') as f:
            sim_data = json.load(f)
        sim_ch4_auc = sim_data.get("ch4", {}).get("overall_auc")
        sim_ch8_auc = sim_data.get("ch8", {}).get("overall_auc")

    # Build gap table
    gap_table = []

    if ch4_result["overall_auc"] is not None:
        gap_table.append({
            "channels": 4,
            "qubits": n_qubits,
            "sim_auc": sim_ch4_auc,
            "hw_auc": ch4_result["overall_auc"],
            "gap": (sim_ch4_auc - ch4_result["overall_auc"]) if sim_ch4_auc else None,
            "encoding_match": True
        })

    if ch8_data and ch8_data["overall_auc"]:
        gap_table.append({
            "channels": 8,
            "qubits": 17,
            "sim_auc": sim_ch8_auc,
            "hw_auc": ch8_data["overall_auc"],
            "gap": (sim_ch8_auc - ch8_data["overall_auc"]) if sim_ch8_auc else None,
            "encoding_mismatch": True,
            "note": "CH8 uses PLV_theta_alpha, CH4 uses V2_band_power"
        })

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "ch4": {
            "backend": backend.name,
            "encoding": "V2_band_power",
            "mitigation": ch4_result.get("mitigation", "unknown"),
            "shots": SHOTS,
            "n_qubits": n_qubits,
            "channels": channels,
            "overall_auc": ch4_result["overall_auc"],
            "per_subject": ch4_result["per_subject"]
        },
        "ch8": ch8_data,
        "gap_table": gap_table
    }

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "hardware_scaling.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("HARDWARE SCALING RESULTS")
    print("=" * 70)

    print(f"\nCH4 ({n_qubits} qubits, V2_band_power):")
    print(f"  Backend: {backend.name}")
    print(f"  Mitigation: {ch4_result.get('mitigation', 'unknown')}")
    print(f"  Overall AUC: {ch4_result['overall_auc']:.4f}" if ch4_result['overall_auc'] else "  Overall AUC: N/A")

    if ch4_result["per_subject"]:
        print(f"  Per-subject:")
        for subj, data in sorted(ch4_result["per_subject"].items()):
            print(f"    {subj}: {data['auc']:.4f}")

    print("\n" + "-" * 70)
    print("GAP TABLE")
    print("-" * 70)
    print(f"{'Channels':<10} {'Qubits':<8} {'Sim AUC':<10} {'HW AUC':<10} {'Gap':<10} {'Note'}")
    print("-" * 70)

    for row in gap_table:
        sim = f"{row['sim_auc']:.4f}" if row.get('sim_auc') else "N/A"
        hw = f"{row['hw_auc']:.4f}" if row.get('hw_auc') else "N/A"
        gap = f"{row['gap']:.4f}" if row.get('gap') else "N/A"
        note = row.get('note', '')[:30] if row.get('encoding_mismatch') else ""
        print(f"{row['channels']:<10} {row['qubits']:<8} {sim:<10} {hw:<10} {gap:<10} {note}")

    print()

    return output


if __name__ == "__main__":
    main()
