#!/usr/bin/env python3
"""
Quantum CC_freq Encoding (V6)

Adapts CC_freq eigenvalues as quantum rotation angle inputs for the A-Gate circuit.

This provides an alternative to the existing PLV-based encoding (V3) that uses
phase synchronization. CC_freq eigenvalues capture cross-channel correlation
structure in the frequency domain, which may provide complementary information.

Encoding strategy:
1. Extract top-N CC_freq eigenvalues (N = number of encoding qubits)
2. Normalize to [0, 2π] using arctan scaling: theta = 2 * arctan(λ / scale_factor)
3. Map to A-Gate (a, b, c) parameters:
   - a = arctan(λ_1) / π  (first eigenvalue -> excitatory amplitude)
   - b = arctan(λ_2) / π * 2π  (second eigenvalue -> phase)
   - c = arctan(λ_3) / π  (third eigenvalue -> inhibitory amplitude)

Reference: A-005 from research prompts
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'QA1'))

from scripts.features.cc_freq_encoding import extract_cc_freq
from sagemaker.train_chbmit import (
    extract_segments_for_subject,
    read_edf_segment,
    preprocess_eeg,
    FS,
)

# Try to import quantum modules
try:
    from QA1.multichannel_circuit import create_multichannel_circuit, get_statevector
    from qiskit import QuantumCircuit
    QUANTUM_AVAILABLE = True
except ImportError:
    QUANTUM_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Encoding Functions
# ============================================================================

def eigenvalue_to_angle(eigenvalue: float, scale_factor: float = 1.0) -> float:
    """
    Map eigenvalue to rotation angle using arctan scaling.

    arctan maps (-∞, +∞) -> (-π/2, +π/2), so we scale by 2 to get full [0, 2π] range.

    theta = 2 * arctan(eigenvalue / scale_factor)

    Args:
        eigenvalue: The eigenvalue to convert
        scale_factor: Scaling factor (higher = more compression)

    Returns:
        Angle in [0, 2π]
    """
    # arctan output is in (-π/2, π/2)
    # Shift and scale to [0, 2π]
    angle = 2 * np.arctan(eigenvalue / scale_factor)
    # Shift from [-π, π] to [0, 2π]
    if angle < 0:
        angle += 2 * np.pi
    return angle


def eigenvalues_to_params(
    eigenvalues: np.ndarray,
    n_channels: int,
    scale_factor: float = 1.0
) -> List[Tuple[float, float, float]]:
    """
    Convert eigenvalues to (a, b, c) parameters for A-Gate circuit.

    For each channel i (0 to n_channels-1):
    - Use eigenvalue i*3, i*3+1, i*3+2 for (a, b, c)
    - If not enough eigenvalues, cycle through available ones

    Args:
        eigenvalues: Sorted eigenvalues (descending magnitude)
        n_channels: Number of channels/encoding units
        scale_factor: Scaling factor for arctan normalization

    Returns:
        List of (a, b, c) tuples for each channel
    """
    n_eigen = len(eigenvalues)

    params_list = []
    for ch in range(n_channels):
        # Get indices for this channel's (a, b, c) parameters
        idx_a = ch % n_eigen
        idx_b = (ch + 1) % n_eigen
        idx_c = (ch + 2) % n_eigen

        # Convert eigenvalues to angles
        angle_a = eigenvalue_to_angle(eigenvalues[idx_a], scale_factor)
        angle_b = eigenvalue_to_angle(eigenvalues[idx_b], scale_factor)
        angle_c = eigenvalue_to_angle(eigenvalues[idx_c], scale_factor)

        # Map to valid parameter ranges
        # a: amplitude for Rx gate, should be in [0, 1] so Rx(2a) spans [0, 2π]
        a = angle_a / (2 * np.pi)  # [0, 1]
        a = np.clip(a, 0.05, 0.95)  # Avoid edge cases

        # b: phase for P gate, should be in [0, 2π]
        b = angle_b  # Already in [0, 2π]

        # c: amplitude for Ry gate, should be in [0, 1] so Ry(2c) spans [0, 2π]
        c = angle_c / (2 * np.pi)  # [0, 1]
        c = np.clip(c, 0.05, 0.95)

        params_list.append((a, b, c))

    return params_list


def extract_cc_freq_params(
    eeg_segment: np.ndarray,
    fs: int = 256,
    freq_low: float = 1.0,
    freq_high: float = 47.0,
    scale_factor: float = 1.0
) -> List[Tuple[float, float, float]]:
    """
    Extract quantum encoding parameters from EEG using CC_freq eigenvalues.

    Args:
        eeg_segment: EEG data (n_channels, n_samples)
        fs: Sampling frequency
        freq_low: Lower frequency bound for CC_freq
        freq_high: Upper frequency bound for CC_freq
        scale_factor: Scaling factor for arctan normalization

    Returns:
        List of (a, b, c) tuples for A-Gate circuit
    """
    n_channels = eeg_segment.shape[0]

    # Extract CC_freq features
    cc_result = extract_cc_freq(eeg_segment, fs=fs, freq_low=freq_low, freq_high=freq_high)
    eigenvalues = cc_result['cc_eigenvalues']

    # Convert to quantum parameters
    params = eigenvalues_to_params(eigenvalues, n_channels, scale_factor)

    return params


def extract_cc_freq_params_subwindows(
    eeg_segment: np.ndarray,
    fs: int = 256,
    window_sec: float = 2.0,
    freq_low: float = 1.0,
    freq_high: float = 47.0,
    scale_factor: float = 1.0
) -> List[List[Tuple[float, float, float]]]:
    """
    Extract quantum encoding parameters from sub-windows.

    Args:
        eeg_segment: EEG data (n_channels, n_samples)
        fs: Sampling frequency
        window_sec: Sub-window length in seconds
        freq_low: Lower frequency bound
        freq_high: Upper frequency bound
        scale_factor: Scaling factor

    Returns:
        List of param_lists, one per sub-window
    """
    n_channels, n_samples = eeg_segment.shape
    window_samples = int(window_sec * fs)
    step = window_samples  # Non-overlapping

    all_subwindow_params = []

    for start in range(0, n_samples - window_samples + 1, step):
        sub = eeg_segment[:, start:start + window_samples]
        try:
            params = extract_cc_freq_params(sub, fs, freq_low, freq_high, scale_factor)
            all_subwindow_params.append(params)
        except Exception as e:
            logger.debug(f"Subwindow extraction failed: {e}")
            continue

    # Ensure at least one result
    if not all_subwindow_params:
        all_subwindow_params.append(extract_cc_freq_params(eeg_segment, fs, freq_low, freq_high, scale_factor))

    return all_subwindow_params


# ============================================================================
# Quantum Circuit Simulation
# ============================================================================

def run_statevector_simulation(
    params_list: List[Tuple[float, float, float]]
) -> Optional[np.ndarray]:
    """
    Run statevector simulation with given parameters.

    Args:
        params_list: List of (a, b, c) tuples for each channel

    Returns:
        Statevector as numpy array, or None if quantum not available
    """
    if not QUANTUM_AVAILABLE:
        logger.warning("Quantum modules not available")
        return None

    try:
        circuit = create_multichannel_circuit(params_list)
        statevector = get_statevector(circuit)
        return statevector
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return None


def compute_fidelity(sv1: np.ndarray, sv2: np.ndarray) -> float:
    """Compute fidelity between two statevectors."""
    fidelity = np.abs(np.vdot(sv1, sv2)) ** 2
    return float(fidelity)


# ============================================================================
# Comparison with PLV Encoding
# ============================================================================

def compare_encodings(
    eeg_segment: np.ndarray,
    fs: int = 256,
    band: Tuple[float, float] = (4, 13)
) -> Dict:
    """
    Compare CC_freq encoding vs PLV encoding for the same segment.

    Returns parameter statistics and differences.
    """
    n_channels = eeg_segment.shape[0]

    # CC_freq encoding
    cc_params = extract_cc_freq_params(eeg_segment, fs)

    # PLV encoding (import from existing V3)
    try:
        from scripts.run_quantum_loso_v3 import extract_plv_params
        plv_params = extract_plv_params(eeg_segment, fs=fs, band=band)
    except ImportError:
        plv_params = None

    # Statistics
    cc_a = [p[0] for p in cc_params]
    cc_b = [p[1] for p in cc_params]
    cc_c = [p[2] for p in cc_params]

    result = {
        'cc_freq': {
            'a': {'mean': np.mean(cc_a), 'std': np.std(cc_a), 'range': [min(cc_a), max(cc_a)]},
            'b': {'mean': np.mean(cc_b), 'std': np.std(cc_b), 'range': [min(cc_b), max(cc_b)]},
            'c': {'mean': np.mean(cc_c), 'std': np.std(cc_c), 'range': [min(cc_c), max(cc_c)]},
        },
    }

    if plv_params:
        plv_a = [p[0] for p in plv_params]
        plv_b = [p[1] for p in plv_params]
        plv_c = [p[2] for p in plv_params]

        result['plv'] = {
            'a': {'mean': np.mean(plv_a), 'std': np.std(plv_a), 'range': [min(plv_a), max(plv_a)]},
            'b': {'mean': np.mean(plv_b), 'std': np.std(plv_b), 'range': [min(plv_b), max(plv_b)]},
            'c': {'mean': np.mean(plv_c), 'std': np.std(plv_c), 'range': [min(plv_c), max(plv_c)]},
        }

        # Correlation between encodings
        result['correlation'] = {
            'a': np.corrcoef(cc_a, plv_a)[0, 1],
            'b': np.corrcoef(cc_b, plv_b)[0, 1],
            'c': np.corrcoef(cc_c, plv_c)[0, 1],
        }

    return result


# ============================================================================
# Validation / Test
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Quantum CC_freq Encoding (V6) Validation")
    print("=" * 60)

    DATA_ROOT = Path("H:/Data/PythonDNU/EEG/chbmit")
    CHANNELS = ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3", "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]

    # Load chb01 data
    if DATA_ROOT.exists():
        subject_dir = DATA_ROOT / "chb01"
        print(f"\nLoading segments from {subject_dir}...")

        segments = extract_segments_for_subject(subject_dir)

        ictal_seg = None
        interictal_seg = None

        for seg in segments:
            if seg.label == 'ictal' and ictal_seg is None:
                ictal_seg = seg
            elif seg.label == 'interictal' and interictal_seg is None:
                interictal_seg = seg
            if ictal_seg and interictal_seg:
                break

        if ictal_seg and interictal_seg:
            # Load ictal
            ictal_data, fs = read_edf_segment(
                str(subject_dir / ictal_seg.source_file),
                ictal_seg.start_sec,
                ictal_seg.end_sec,
                target_channels=CHANNELS
            )
            ictal_data = preprocess_eeg(ictal_data, fs=int(fs))

            # Load interictal
            inter_data, fs = read_edf_segment(
                str(subject_dir / interictal_seg.source_file),
                interictal_seg.start_sec,
                interictal_seg.end_sec,
                target_channels=CHANNELS
            )
            inter_data = preprocess_eeg(inter_data, fs=int(fs))

            print(f"\n--- ICTAL Segment ---")
            print(f"Shape: {ictal_data.shape}")

            ictal_params = extract_cc_freq_params(ictal_data, fs=int(fs))
            print(f"\nCC_freq Quantum Parameters (a, b, c):")
            for i, (a, b, c) in enumerate(ictal_params):
                print(f"  Channel {i+1}: a={a:.4f}, b={b:.4f}, c={c:.4f}")

            print(f"\n--- INTERICTAL Segment ---")
            print(f"Shape: {inter_data.shape}")

            inter_params = extract_cc_freq_params(inter_data, fs=int(fs))
            print(f"\nCC_freq Quantum Parameters (a, b, c):")
            for i, (a, b, c) in enumerate(inter_params):
                print(f"  Channel {i+1}: a={a:.4f}, b={b:.4f}, c={c:.4f}")

            # Compare with PLV encoding
            print(f"\n--- Encoding Comparison (Ictal) ---")
            comparison = compare_encodings(ictal_data, fs=int(fs))
            print(f"CC_freq params:")
            for param in ['a', 'b', 'c']:
                stats = comparison['cc_freq'][param]
                print(f"  {param}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

            if 'plv' in comparison:
                print(f"\nPLV params:")
                for param in ['a', 'b', 'c']:
                    stats = comparison['plv'][param]
                    print(f"  {param}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")

                print(f"\nCorrelation (CC_freq vs PLV):")
                for param in ['a', 'b', 'c']:
                    corr = comparison['correlation'][param]
                    print(f"  {param}: {corr:.4f}")

            # Run quantum simulation if available
            if QUANTUM_AVAILABLE:
                print(f"\n--- Quantum Simulation ---")
                print("Running statevector simulation...")

                sv_ictal = run_statevector_simulation(ictal_params)
                sv_inter = run_statevector_simulation(inter_params)

                if sv_ictal is not None and sv_inter is not None:
                    fidelity = compute_fidelity(sv_ictal, sv_inter)
                    print(f"Fidelity (ictal vs interictal): {fidelity:.6f}")

                    # Self-fidelities (should be 1.0)
                    self_fid_ictal = compute_fidelity(sv_ictal, sv_ictal)
                    self_fid_inter = compute_fidelity(sv_inter, sv_inter)
                    print(f"Self-fidelity ictal: {self_fid_ictal:.6f}")
                    print(f"Self-fidelity inter: {self_fid_inter:.6f}")
            else:
                print(f"\nQuantum simulation not available (Qiskit not installed)")

    else:
        print(f"Data directory not found: {DATA_ROOT}")
        print("Running with synthetic data...")

        # Synthetic test
        np.random.seed(42)
        n_channels = 8
        n_samples = 256 * 10
        fs = 256

        t = np.arange(n_samples) / fs

        # Ictal: synchronized
        ictal_data = np.array([
            np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 0.5))
            + 0.3 * np.random.randn(n_samples)
            for _ in range(n_channels)
        ])

        # Interictal: independent
        inter_data = np.random.randn(n_channels, n_samples)

        print(f"\n--- Synthetic ICTAL ---")
        ictal_params = extract_cc_freq_params(ictal_data, fs=fs)
        print(f"Parameters: {ictal_params}")

        print(f"\n--- Synthetic INTERICTAL ---")
        inter_params = extract_cc_freq_params(inter_data, fs=fs)
        print(f"Parameters: {inter_params}")

    print("\n" + "=" * 60)
    print("Validation complete")
    print("=" * 60)
