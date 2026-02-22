#!/usr/bin/env python3
"""
CC_freq (Frequency-Domain Correlation) Feature Extraction

Implements the frequency-domain correlation feature block from the Kaggle
winning solution (MichaelHills/Barachant approach).

The key insight: correlations in the frequency domain capture how channels
co-vary in their spectral content, which is more robust to phase shifts
than time-domain correlation alone.

Procedure:
1. Compute FFT magnitude for each channel
2. Keep only 1-47 Hz bins (clinically relevant for scalp EEG)
3. Normalize column-by-column (per frequency bin) across channels
4. Compute Pearson correlation matrix on normalized FFT magnitudes
5. Extract upper triangle and eigenvalues as separate feature vectors

Reference: A-001 from research prompts
"""

import numpy as np
from typing import Dict, Tuple, Optional


def extract_cc_freq(
    data: np.ndarray,
    fs: int = 256,
    freq_low: float = 1.0,
    freq_high: float = 47.0,
    epsilon: float = 1e-10
) -> Dict[str, np.ndarray]:
    """
    Extract frequency-domain correlation (CC_freq) features.

    Args:
        data: EEG data array of shape (n_channels, n_samples)
        fs: Sampling frequency in Hz (default 256 for CHB-MIT)
        freq_low: Lower frequency bound in Hz (default 1)
        freq_high: Upper frequency bound in Hz (default 47)
        epsilon: Small constant for numerical stability

    Returns:
        Dictionary with keys:
            - 'cc_upper': Upper triangle of CC_freq matrix, shape (n_channels*(n_channels-1)/2,)
            - 'cc_eigenvalues': Sorted eigenvalues (descending magnitude), shape (n_channels,)
            - 'cc_matrix': Full correlation matrix, shape (n_channels, n_channels)
            - 'fft_mag_normalized': Normalized FFT magnitudes, shape (n_channels, n_freq_bins)
    """
    n_channels, n_samples = data.shape

    # Step 1: Compute FFT magnitude
    # rfft gives positive frequencies only: 0, fs/N, 2*fs/N, ..., fs/2
    fft_result = np.fft.rfft(data, axis=1)
    fft_mag = np.abs(fft_result)

    # Compute frequency bins
    n_fft = fft_mag.shape[1]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    # Step 2: Find bin indices for freq_low to freq_high Hz
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    bin_indices = np.where(freq_mask)[0]

    if len(bin_indices) == 0:
        raise ValueError(
            f"No frequency bins found in range [{freq_low}, {freq_high}] Hz. "
            f"Available range: [{freqs[1]:.2f}, {freqs[-1]:.2f}] Hz"
        )

    # Extract only the relevant frequency bins
    fft_mag_band = fft_mag[:, bin_indices]  # shape: (n_channels, n_freq_bins)

    # Step 3: Normalize column-by-column (per frequency bin) across channels
    # For each frequency bin k: fft_norm[:, k] = (fft_mag[:, k] - mean) / std
    fft_norm = np.zeros_like(fft_mag_band)
    for k in range(fft_mag_band.shape[1]):
        col = fft_mag_band[:, k]
        col_mean = np.mean(col)
        col_std = np.std(col)
        if col_std > epsilon:
            fft_norm[:, k] = (col - col_mean) / col_std
        else:
            fft_norm[:, k] = col - col_mean  # Zero std case

    # Step 4: Compute correlation matrix on normalized FFT magnitudes
    # corrcoef treats each row as a variable, columns as observations
    # So we have n_channels variables, each with n_freq_bins observations
    try:
        CC_freq = np.corrcoef(fft_norm)  # shape: (n_channels, n_channels)
    except Exception:
        # Fallback: return identity matrix if correlation fails
        CC_freq = np.eye(n_channels)

    # Handle NaN values (can occur with constant rows)
    if np.isnan(CC_freq).any():
        CC_freq = np.nan_to_num(CC_freq, nan=0.0)
        np.fill_diagonal(CC_freq, 1.0)

    # Step 5: Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n_channels, k=1)
    cc_upper = CC_freq[triu_indices]

    # Step 6: Compute eigenvalues
    try:
        # Use eigvalsh for symmetric matrices (numerically stable)
        eigenvalues = np.linalg.eigvalsh(CC_freq)
        # Sort by magnitude descending
        eigenvalues_sorted = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
    except np.linalg.LinAlgError:
        # Fallback: return ones if eigenvalue decomposition fails
        eigenvalues_sorted = np.ones(n_channels)

    return {
        'cc_upper': cc_upper,
        'cc_eigenvalues': eigenvalues_sorted,
        'cc_matrix': CC_freq,
        'fft_mag_normalized': fft_norm,
        'freq_bins': freqs[bin_indices],
        'n_freq_bins': len(bin_indices),
    }


def validate_eigenvalue_sum(eigenvalues: np.ndarray, n_channels: int, tol: float = 1e-6) -> bool:
    """
    Validate that eigenvalues of a correlation matrix sum to n_channels.

    For a correlation matrix (diagonal = 1), trace = n_channels,
    and trace equals sum of eigenvalues.

    Args:
        eigenvalues: Array of eigenvalues
        n_channels: Number of channels
        tol: Tolerance for comparison

    Returns:
        True if eigenvalues sum to n_channels within tolerance
    """
    return np.abs(np.sum(eigenvalues) - n_channels) < tol


# =============================================================================
# Validation / Test
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directories to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("=" * 60)
    print("CC_freq Feature Extraction Validation")
    print("=" * 60)

    # Try to load real CHB-MIT data
    try:
        from sagemaker.train_chbmit import (
            extract_segments_for_subject,
            read_edf_segment,
            preprocess_eeg,
            FS,
        )

        DATA_ROOT = Path("H:/Data/PythonDNU/EEG/chbmit")
        CHANNELS = ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3", "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]

        if DATA_ROOT.exists():
            subject_dir = DATA_ROOT / "chb01"

            print(f"\nLoading segments from {subject_dir}...")
            segments = extract_segments_for_subject(subject_dir)

            # Find one ictal and one interictal segment
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
                print(f"\nFound ictal segment: {ictal_seg.source_file} [{ictal_seg.start_sec:.1f}-{ictal_seg.end_sec:.1f}s]")
                print(f"Found interictal segment: {interictal_seg.source_file} [{interictal_seg.start_sec:.1f}-{interictal_seg.end_sec:.1f}s]")

                # Load and preprocess ictal data
                ictal_data, fs = read_edf_segment(
                    str(subject_dir / ictal_seg.source_file),
                    ictal_seg.start_sec,
                    ictal_seg.end_sec,
                    target_channels=CHANNELS
                )
                ictal_data = preprocess_eeg(ictal_data, fs=int(fs))

                # Load and preprocess interictal data
                inter_data, fs = read_edf_segment(
                    str(subject_dir / interictal_seg.source_file),
                    interictal_seg.start_sec,
                    interictal_seg.end_sec,
                    target_channels=CHANNELS
                )
                inter_data = preprocess_eeg(inter_data, fs=int(fs))

                print(f"\n--- ICTAL Segment (chb01) ---")
                print(f"Data shape: {ictal_data.shape}")
                print(f"Duration: {ictal_data.shape[1] / fs:.2f} seconds")

                result_ictal = extract_cc_freq(ictal_data, fs=int(fs))

                print(f"\nCC_freq Results:")
                print(f"  cc_upper shape: {result_ictal['cc_upper'].shape}")
                print(f"  cc_upper range: [{result_ictal['cc_upper'].min():.4f}, {result_ictal['cc_upper'].max():.4f}]")
                print(f"  cc_eigenvalues shape: {result_ictal['cc_eigenvalues'].shape}")
                print(f"  cc_eigenvalues: {result_ictal['cc_eigenvalues']}")
                print(f"  Eigenvalue sum: {np.sum(result_ictal['cc_eigenvalues']):.6f} (should be {ictal_data.shape[0]})")

                valid = validate_eigenvalue_sum(result_ictal['cc_eigenvalues'], ictal_data.shape[0])
                print(f"  Eigenvalue sum validation: {'PASSED' if valid else 'FAILED'}")

                print(f"\n--- INTERICTAL Segment (chb01) ---")
                print(f"Data shape: {inter_data.shape}")
                print(f"Duration: {inter_data.shape[1] / fs:.2f} seconds")

                result_inter = extract_cc_freq(inter_data, fs=int(fs))

                print(f"\nCC_freq Results:")
                print(f"  cc_upper shape: {result_inter['cc_upper'].shape}")
                print(f"  cc_upper range: [{result_inter['cc_upper'].min():.4f}, {result_inter['cc_upper'].max():.4f}]")
                print(f"  cc_eigenvalues shape: {result_inter['cc_eigenvalues'].shape}")
                print(f"  cc_eigenvalues: {result_inter['cc_eigenvalues']}")
                print(f"  Eigenvalue sum: {np.sum(result_inter['cc_eigenvalues']):.6f} (should be {inter_data.shape[0]})")

                valid = validate_eigenvalue_sum(result_inter['cc_eigenvalues'], inter_data.shape[0])
                print(f"  Eigenvalue sum validation: {'PASSED' if valid else 'FAILED'}")

                # Compare ictal vs interictal
                print(f"\n--- Comparison ---")
                eigen_diff = result_ictal['cc_eigenvalues'] - result_inter['cc_eigenvalues']
                print(f"Eigenvalue differences (ictal - interictal):")
                for i, diff in enumerate(eigen_diff):
                    print(f"  λ_{i+1}: {diff:+.4f}")

                cc_upper_diff = np.mean(np.abs(result_ictal['cc_upper'] - result_inter['cc_upper']))
                print(f"\nMean |Δ cc_upper|: {cc_upper_diff:.4f}")

            else:
                print("Could not find both ictal and interictal segments")

        else:
            print(f"Data directory not found: {DATA_ROOT}")
            print("Running with synthetic data...")
            raise FileNotFoundError("Use synthetic data")

    except Exception as e:
        print(f"\nUsing synthetic data for validation: {e}")

        # Synthetic data test
        np.random.seed(42)
        n_channels = 8
        n_samples = 256 * 10  # 10 seconds at 256 Hz
        fs = 256

        # Create synthetic EEG with known spectral properties
        t = np.arange(n_samples) / fs

        # Ictal: synchronized 10 Hz activity across channels
        ictal_data = np.array([
            np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 0.5))
            + 0.3 * np.random.randn(n_samples)
            for _ in range(n_channels)
        ])

        # Interictal: independent random signals
        inter_data = np.random.randn(n_channels, n_samples)

        print(f"\n--- Synthetic ICTAL ---")
        print(f"Data shape: {ictal_data.shape}")
        result_ictal = extract_cc_freq(ictal_data, fs=fs)
        print(f"cc_upper shape: {result_ictal['cc_upper'].shape}")
        print(f"cc_upper range: [{result_ictal['cc_upper'].min():.4f}, {result_ictal['cc_upper'].max():.4f}]")
        print(f"cc_eigenvalues: {result_ictal['cc_eigenvalues']}")
        print(f"Eigenvalue sum: {np.sum(result_ictal['cc_eigenvalues']):.6f} (should be {n_channels})")
        print(f"Validation: {'PASSED' if validate_eigenvalue_sum(result_ictal['cc_eigenvalues'], n_channels) else 'FAILED'}")

        print(f"\n--- Synthetic INTERICTAL ---")
        print(f"Data shape: {inter_data.shape}")
        result_inter = extract_cc_freq(inter_data, fs=fs)
        print(f"cc_upper shape: {result_inter['cc_upper'].shape}")
        print(f"cc_upper range: [{result_inter['cc_upper'].min():.4f}, {result_inter['cc_upper'].max():.4f}]")
        print(f"cc_eigenvalues: {result_inter['cc_eigenvalues']}")
        print(f"Eigenvalue sum: {np.sum(result_inter['cc_eigenvalues']):.6f} (should be {n_channels})")
        print(f"Validation: {'PASSED' if validate_eigenvalue_sum(result_inter['cc_eigenvalues'], n_channels) else 'FAILED'}")

        # Ictal should have higher correlation (more synchronized)
        print(f"\n--- Comparison ---")
        print(f"Mean |cc_upper| ictal: {np.mean(np.abs(result_ictal['cc_upper'])):.4f}")
        print(f"Mean |cc_upper| interictal: {np.mean(np.abs(result_inter['cc_upper'])):.4f}")
        print(f"Max eigenvalue ictal: {result_ictal['cc_eigenvalues'][0]:.4f}")
        print(f"Max eigenvalue interictal: {result_inter['cc_eigenvalues'][0]:.4f}")

    print("\n" + "=" * 60)
    print("Validation complete")
    print("=" * 60)
