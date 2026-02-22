#!/usr/bin/env python3
"""
================================================================================
PROMPT 009B — Hardware CH8 with V2 Band Power Encoding
================================================================================

Runs CH8 on IBM Heron using V2_band_power encoding (matching CH4 from PROMPT 009)
to provide a clean scaling curve comparison.

Backend: IBM Heron r2 (ibm_torino) — no simulator
Encoding: V2 band_power — MUST match PROMPT 009 exactly
Channels: CH8 loaded from results/scaling/classical_channel_scaling.json
Shots: 1024 (matching PROMPT 009)
Error mitigation: match PROMPT 009 (none — resilience_level not available)

Outputs:
- results/projection/hardware_scaling_v2bp_ch8.json
- results/projection/hardware_v2bp_ch8_raw_scores.json (all sub-window scores)
- Updated results/projection/hardware_scaling.json with all three CH entries
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
EXISTING_HW_SCALING = Path("results/projection/hardware_scaling.json")

# Hardware configuration
SHOTS = 1024
BACKEND_NAME = "ibm_torino"

# Priority subjects (same as PROMPT 006/009)
PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']

# Sub-window configuration
SUB_WINDOW_SEC = 1  # 1-second sub-windows
N_SUB_WINDOWS = 30  # 30 sub-windows per 30-second segment


# =============================================================================
# V2 BAND-POWER ENCODING (exact copy from hardware_scaling_ch4.py)
# =============================================================================

def extract_pn_params_bandpower(eeg_segment: np.ndarray, fs: float = 256.0) -> List[Tuple[float, float, float]]:
    """
    Extract (a, b, c) per channel using band power ratios.

    This is the V2 band_power encoding that produced AUC 0.534 in simulation.
    MUST match PROMPT 009 exactly.
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

    # Fallback for CH8
    if n_channels == 8:
        return ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3", "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]

    raise ValueError(f"No channel configuration found for {n_channels} channels")


def load_eeg_segment_with_subwindows(seg: EEGSegment, subject_dir: Path, channels: List[str],
                                      fs: float = 256.0) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]]]:
    """
    Load EEG segment for specified channels.
    Returns full segment and list of 30 x 1-second sub-windows.
    """
    try:
        import pyedflib

        edf_path = subject_dir / seg.source_file
        if not edf_path.exists():
            return None, None

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
                    return None, None

            file_fs = f.getSampleFrequency(0)
            start_sample = int(seg.start_sec * file_fs)
            n_samples = int(WINDOW_SEC * file_fs)

            data = np.zeros((len(channels), n_samples))
            for i, ch_idx in enumerate(channel_indices):
                signal = f.readSignal(ch_idx)
                end_sample = min(start_sample + n_samples, len(signal))
                actual_samples = end_sample - start_sample
                if actual_samples < n_samples:
                    return None, None
                data[i] = signal[start_sample:end_sample]

            # Split into sub-windows
            samples_per_subwindow = int(SUB_WINDOW_SEC * file_fs)
            sub_windows = []
            for sw in range(N_SUB_WINDOWS):
                sw_start = sw * samples_per_subwindow
                sw_end = sw_start + samples_per_subwindow
                if sw_end <= data.shape[1]:
                    sub_windows.append(data[:, sw_start:sw_end])

            return data, sub_windows

    except Exception as e:
        logger.warning(f"Failed to load {seg.source_file}: {e}")
        return None, None


def load_subject_data_with_subwindows(subject_id: str, channels: List[str]) -> Tuple[List[np.ndarray], List[List[np.ndarray]], List[str]]:
    """Load all segments for a subject with sub-windows."""
    subject_dir = DATA_DIR / subject_id
    if not subject_dir.exists():
        return [], [], []

    segs = _extract_segments_for_subject(subject_dir)

    eeg_list = []
    subwindow_list = []
    labels = []

    for seg in segs:
        eeg, subwindows = load_eeg_segment_with_subwindows(seg, subject_dir, channels)
        if eeg is not None and subwindows is not None and len(subwindows) == N_SUB_WINDOWS:
            eeg_list.append(eeg)
            subwindow_list.append(subwindows)
            labels.append(seg.label)

    return eeg_list, subwindow_list, labels


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
            min_num_qubits=17  # CH8 needs 17 qubits
        )
        logger.info(f"Using fallback backend: {backend.name} ({backend.num_qubits} qubits)")
        return backend
    except Exception as e:
        logger.error(f"No suitable backend found: {e}")
        raise


def run_circuits_on_hardware(circuits: List, backend, shots: int = SHOTS) -> Tuple[List[Dict[str, int]], str]:
    """
    Run circuits on IBM hardware using SamplerV2.
    Matches PROMPT 009 configuration (no error mitigation).
    """
    from qiskit_ibm_runtime import SamplerV2, Batch
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    logger.info(f"Transpiling {len(circuits)} circuits for {backend.name}...")

    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    transpiled = pm.run(circuits)

    logger.info(f"Submitting to hardware with {shots} shots...")

    # Match PROMPT 009: no error mitigation (resilience_level not available)
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
# POLARITY CALIBRATION
# =============================================================================

def compute_calibrated_auc(raw_auc: float) -> Tuple[float, str]:
    """Compute calibrated AUC and polarity."""
    if raw_auc is None:
        return None, None
    calibrated = max(raw_auc, 1.0 - raw_auc)
    polarity = "standard" if raw_auc >= 0.5 else "inverted"
    return calibrated, polarity


# =============================================================================
# LOSO CV WITH SUB-WINDOW STORAGE
# =============================================================================

def run_hardware_loso_ch8(channels: List[str], subjects: List[str], backend) -> Tuple[Dict, Dict]:
    """
    Run LOSO CV on IBM hardware for CH8 with sub-window score storage.
    Returns results dict and raw_scores dict.
    """
    from sklearn.metrics import roc_auc_score

    n_channels = len(channels)
    n_qubits = 2 * n_channels + 1

    logger.info(f"Running CH8 hardware LOSO: {n_channels} channels, {n_qubits} qubits")
    logger.info(f"Subjects: {subjects}")

    per_subject_results = {}
    raw_scores_data = {}  # Store all sub-window scores

    all_y_true = []
    all_y_score_max = []
    all_y_score_mean = []
    all_y_score_std = []

    mitigation_used = None

    for test_subject in subjects:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing on {test_subject}")
        logger.info(f"{'='*60}")

        # Load test data with sub-windows
        test_eeg, test_subwindows, test_labels = load_subject_data_with_subwindows(test_subject, channels)
        if len(test_eeg) == 0:
            logger.warning(f"No data for {test_subject}, skipping")
            continue

        # Load training data (all other subjects)
        train_subjects = [s for s in subjects if s != test_subject]
        train_ictal_params = []
        train_inter_params = []

        for train_subj in train_subjects:
            eeg_list, _, labels = load_subject_data_with_subwindows(train_subj, channels)
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

        # Process each segment with all sub-windows
        raw_scores_data[test_subject] = {}
        y_true_subj = []
        y_scores_max_subj = []
        y_scores_mean_subj = []
        y_scores_std_subj = []

        for seg_idx, (subwindows, label) in enumerate(zip(test_subwindows, test_labels)):
            # Build circuits for all 30 sub-windows
            sw_circuits = []
            for sw_eeg in subwindows:
                params = extract_pn_params_bandpower(sw_eeg)
                circuit = create_multichannel_circuit(params)
                sw_circuits.append(circuit)

            # Run all sub-window circuits on hardware
            logger.info(f"Running {len(sw_circuits)} sub-window circuits for segment {seg_idx}...")
            sw_counts, _ = run_circuits_on_hardware(sw_circuits, backend, shots=SHOTS)

            # Compute scores for each sub-window
            sw_scores = []
            for counts in sw_counts:
                fid_ictal = hellinger_fidelity(counts, ictal_template_counts)
                fid_inter = hellinger_fidelity(counts, inter_template_counts)
                score = fid_ictal - fid_inter
                sw_scores.append(score)

            # Compute aggregation variants
            max_score = float(np.max(sw_scores))
            mean_score = float(np.mean(sw_scores))
            std_score = float(np.std(sw_scores))

            # Store raw scores
            raw_scores_data[test_subject][str(seg_idx)] = {
                "label": 1 if label == 'ictal' else 0,
                "sub_window_scores": [float(s) for s in sw_scores],
                "max_score": max_score,
                "mean_score": mean_score,
                "std_score": std_score
            }

            # Collect for AUC computation
            y_true_subj.append(1 if label == 'ictal' else 0)
            y_scores_max_subj.append(max_score)
            y_scores_mean_subj.append(mean_score)
            y_scores_std_subj.append(std_score)

        # Compute AUC variants for this subject
        unique_labels = set(y_true_subj)
        if len(unique_labels) < 2:
            logger.warning(f"{test_subject}: single class in test set, skipping AUC")
            continue

        try:
            auc_max = roc_auc_score(y_true_subj, y_scores_max_subj)
            auc_mean = roc_auc_score(y_true_subj, y_scores_mean_subj)
            # For STD, higher STD may indicate seizure (more variability)
            auc_std = roc_auc_score(y_true_subj, y_scores_std_subj)
        except Exception as e:
            logger.warning(f"AUC calculation failed for {test_subject}: {e}")
            continue

        # Compute calibrated AUC
        calibrated_auc, polarity = compute_calibrated_auc(auc_max)

        per_subject_results[test_subject] = {
            "raw_auc": auc_max,
            "calibrated_auc": calibrated_auc,
            "polarity": polarity,
            "n_samples": len(y_true_subj),
            "n_seizure": sum(y_true_subj),
            "auc_mean": auc_mean,
            "auc_std": auc_std
        }

        all_y_true.extend(y_true_subj)
        all_y_score_max.extend(y_scores_max_subj)
        all_y_score_mean.extend(y_scores_mean_subj)
        all_y_score_std.extend(y_scores_std_subj)

        logger.info(f"{test_subject}: Raw AUC = {auc_max:.4f}, Calibrated = {calibrated_auc:.4f}, Polarity = {polarity}")

    # Compute overall AUC variants
    overall_auc_max = None
    overall_auc_mean = None
    overall_auc_std = None
    calibrated_overall = None
    n_inverted = 0

    if len(set(all_y_true)) >= 2:
        try:
            overall_auc_max = roc_auc_score(all_y_true, all_y_score_max)
            overall_auc_mean = roc_auc_score(all_y_true, all_y_score_mean)
            overall_auc_std = roc_auc_score(all_y_true, all_y_score_std)
            calibrated_overall, _ = compute_calibrated_auc(overall_auc_max)
        except:
            pass

    # Count inverted subjects
    for subj_data in per_subject_results.values():
        if subj_data.get('polarity') == 'inverted':
            n_inverted += 1

    results = {
        "overall_auc": overall_auc_max,
        "calibrated_overall_auc": calibrated_overall,
        "n_inverted": n_inverted,
        "per_subject": per_subject_results,
        "mitigation": mitigation_used,
        "aggregation_variants": {
            "max_auc": overall_auc_max,
            "mean_auc": overall_auc_mean,
            "std_auc": overall_auc_std
        }
    }

    return results, raw_scores_data


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PROMPT 009B — Hardware CH8 with V2 Band Power Encoding")
    print("=" * 70)
    print()

    # Load CH8 channels
    channels = load_channels(8)
    n_qubits = 2 * len(channels) + 1
    logger.info(f"CH8 channels: {channels}")
    logger.info(f"Qubits required: {n_qubits}")

    # Connect to IBM hardware
    try:
        backend = connect_to_ibm()
    except Exception as e:
        logger.error(f"Cannot connect to IBM hardware: {e}")
        return None

    # Run CH8 LOSO on hardware
    logger.info("\nRunning CH8 LOSO CV with V2_band_power on hardware...")
    ch8_result, raw_scores = run_hardware_loso_ch8(channels, PRIORITY_SUBJECTS, backend)

    # Save raw scores
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_scores_path = OUTPUT_DIR / "hardware_v2bp_ch8_raw_scores.json"
    with open(raw_scores_path, 'w') as f:
        json.dump(raw_scores, f, indent=2)
    logger.info(f"Saved raw scores: {raw_scores_path}")

    # Build CH8 V2 output
    ch8_v2_output = {
        "backend": backend.name,
        "encoding": "V2_band_power",
        "mitigation": ch8_result.get("mitigation", "none"),
        "shots": SHOTS,
        "n_qubits": n_qubits,
        "channels": channels,
        "overall_auc": ch8_result["overall_auc"],
        "calibrated_overall_auc": ch8_result["calibrated_overall_auc"],
        "n_inverted": ch8_result["n_inverted"],
        "per_subject": ch8_result["per_subject"],
        "aggregation_variants": ch8_result["aggregation_variants"],
        "timestamp": datetime.now().isoformat()
    }

    # Save CH8 V2 specific output
    ch8_v2_path = OUTPUT_DIR / "hardware_scaling_v2bp_ch8.json"
    with open(ch8_v2_path, 'w') as f:
        json.dump(ch8_v2_output, f, indent=2)
    logger.info(f"Saved: {ch8_v2_path}")

    # Load existing hardware_scaling.json and update
    existing_data = {}
    if EXISTING_HW_SCALING.exists():
        with open(EXISTING_HW_SCALING, 'r') as f:
            existing_data = json.load(f)

    # Load simulation results for gap calculation
    sim_ch4_auc = None
    sim_ch8_auc = None
    if SIM_SCALING.exists():
        with open(SIM_SCALING, 'r') as f:
            sim_data = json.load(f)
        sim_ch4_auc = sim_data.get("ch4", {}).get("overall_auc")
        sim_ch8_auc = sim_data.get("ch8", {}).get("overall_auc")

    # Load CH8 PLV results for cross-encoding comparison
    ch8_plv_data = None
    if HW_VALIDATION.exists():
        with open(HW_VALIDATION, 'r') as f:
            ch8_hw = json.load(f)
        ch8_plv_data = {
            "backend": ch8_hw.get("backend", "ibm_torino"),
            "encoding": ch8_hw.get("encoding_variant", "PLV_theta_alpha"),
            "mitigation": ch8_hw.get("error_mitigation", "none"),
            "shots": ch8_hw.get("shots", 1024),
            "n_qubits": 17,
            "overall_auc": ch8_hw.get("ch8", {}).get("overall_auc"),
            "per_subject": ch8_hw.get("ch8", {}).get("per_subject", {}),
            "note": "Cross-encoding comparison (PLV vs V2_band_power)"
        }

    # Build comprehensive gap table
    gap_table = []

    # Entry 1: CH4 V2_band_power
    ch4_data = existing_data.get("ch4", {})
    if ch4_data.get("overall_auc") is not None:
        gap_table.append({
            "channels": 4,
            "qubits": 9,
            "encoding": "V2_band_power",
            "sim_auc": sim_ch4_auc,
            "hw_raw_auc": ch4_data["overall_auc"],
            "hw_cal_auc": ch4_data["overall_auc"],  # CH4 all at 0.5, no calibration change
            "gap_calibrated": (sim_ch4_auc - ch4_data["overall_auc"]) if sim_ch4_auc else None
        })

    # Entry 2: CH8 V2_band_power (this run)
    if ch8_result["overall_auc"] is not None:
        gap_table.append({
            "channels": 8,
            "qubits": 17,
            "encoding": "V2_band_power",
            "sim_auc": sim_ch8_auc,
            "hw_raw_auc": ch8_result["overall_auc"],
            "hw_cal_auc": ch8_result["calibrated_overall_auc"],
            "gap_calibrated": (sim_ch8_auc - ch8_result["calibrated_overall_auc"]) if sim_ch8_auc and ch8_result["calibrated_overall_auc"] else None,
            "n_inverted": ch8_result["n_inverted"]
        })

    # Entry 3: CH8 PLV_theta_alpha (prior run, cross-encoding)
    if ch8_plv_data and ch8_plv_data.get("overall_auc"):
        # Load calibrated results from patient profiles if available
        patient_profiles_path = Path("results/patient_analysis/quantum_patient_profiles.json")
        plv_cal_auc = ch8_plv_data["overall_auc"]
        if patient_profiles_path.exists():
            with open(patient_profiles_path, 'r') as f:
                pp_data = json.load(f)
            plv_cal_auc = pp_data.get("calibrated_overall_auc", ch8_plv_data["overall_auc"])

        gap_table.append({
            "channels": 8,
            "qubits": 17,
            "encoding": "PLV_theta_alpha",
            "sim_auc": 0.4597,  # V3_PLV_4-13 simulation AUC
            "hw_raw_auc": ch8_plv_data["overall_auc"],
            "hw_cal_auc": plv_cal_auc,
            "gap_calibrated": None,  # Cross-encoding, not directly comparable
            "note": "Cross-encoding comparison"
        })

    # Update hardware_scaling.json with all entries
    updated_output = {
        "timestamp": datetime.now().isoformat(),
        "ch4": existing_data.get("ch4", {}),
        "ch8_v2bp": ch8_v2_output,
        "ch8_plv": ch8_plv_data,
        "gap_table": gap_table
    }

    with open(EXISTING_HW_SCALING, 'w') as f:
        json.dump(updated_output, f, indent=2)
    logger.info(f"Updated: {EXISTING_HW_SCALING}")

    # Print summary
    print("\n" + "=" * 70)
    print("HARDWARE SCALING RESULTS — V2_band_power Encoding")
    print("=" * 70)

    print(f"\nCH8 ({n_qubits} qubits, V2_band_power):")
    print(f"  Backend: {backend.name}")
    print(f"  Mitigation: {ch8_result.get('mitigation', 'none')}")
    print(f"  Raw AUC: {ch8_result['overall_auc']:.4f}" if ch8_result['overall_auc'] else "  Raw AUC: N/A")
    print(f"  Calibrated AUC: {ch8_result['calibrated_overall_auc']:.4f}" if ch8_result['calibrated_overall_auc'] else "  Calibrated AUC: N/A")
    print(f"  Inverted subjects: {ch8_result['n_inverted']}")

    if ch8_result["per_subject"]:
        print(f"  Per-subject:")
        for subj, data in sorted(ch8_result["per_subject"].items()):
            pol = data.get('polarity', 'unknown')
            raw = data.get('raw_auc', 0)
            cal = data.get('calibrated_auc', 0)
            print(f"    {subj}: raw={raw:.4f}, cal={cal:.4f}, {pol}")

    print("\n" + "-" * 90)
    print("GAP TABLE")
    print("-" * 90)
    print(f"{'Ch':<4} {'Qubits':<8} {'Encoding':<16} {'Sim AUC':<10} {'HW Raw':<10} {'HW Cal':<10} {'Gap(cal)':<10}")
    print("-" * 90)

    for row in gap_table:
        sim = f"{row['sim_auc']:.4f}" if row.get('sim_auc') else "N/A"
        hw_raw = f"{row['hw_raw_auc']:.4f}" if row.get('hw_raw_auc') else "N/A"
        hw_cal = f"{row['hw_cal_auc']:.4f}" if row.get('hw_cal_auc') else "N/A"
        gap = f"{row['gap_calibrated']:.4f}" if row.get('gap_calibrated') else "n/a"
        enc = row.get('encoding', 'unknown')[:14]
        print(f"{row['channels']:<4} {row['qubits']:<8} {enc:<16} {sim:<10} {hw_raw:<10} {hw_cal:<10} {gap:<10}")

    print()

    return updated_output


if __name__ == "__main__":
    main()
