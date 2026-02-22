#!/usr/bin/env python3
"""
Log10 FFT Magnitude Feature Extraction

Implements the log10 FFT magnitude feature block from the Kaggle winning
solution (MichaelHills/Barachant Block 1).

Simple but effective: log-transform of FFT magnitudes preserves the
multiplicative nature of spectral power differences between states.

Procedure:
1. Compute FFT magnitude across 1-47 Hz
2. Apply log10 transform (add small epsilon for stability)
3. Result is a matrix (n_channels x n_freq_bins) - flatten to 1D feature vector

Reference: A-003 from research prompts
"""

import numpy as np
from typing import Dict, Tuple


def extract_log_fft(
    data: np.ndarray,
    fs: int = 256,
    freq_low: float = 1.0,
    freq_high: float = 47.0,
    epsilon: float = 1e-10,
    flatten: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract log10 FFT magnitude features.

    Args:
        data: EEG data array of shape (n_channels, n_samples)
        fs: Sampling frequency in Hz (default 256 for CHB-MIT)
        freq_low: Lower frequency bound in Hz (default 1)
        freq_high: Upper frequency bound in Hz (default 47)
        epsilon: Small constant for numerical stability in log
        flatten: If True, return flattened feature vector; else return matrix

    Returns:
        Dictionary with keys:
            - 'features': Flattened feature vector of shape (n_channels * n_freq_bins,)
                          or matrix of shape (n_channels, n_freq_bins) if flatten=False
            - 'fft_mag': Raw FFT magnitudes (before log transform)
            - 'log_fft_mag': Log10 FFT magnitudes (matrix form)
            - 'freq_bins': Frequency values for each bin
            - 'n_channels': Number of channels
            - 'n_freq_bins': Number of frequency bins
    """
    n_channels, n_samples = data.shape

    # Compute FFT magnitude
    fft_result = np.fft.rfft(data, axis=1)
    fft_mag = np.abs(fft_result)

    # Compute frequency bins
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    # Find bin indices for freq_low to freq_high Hz
    freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
    bin_indices = np.where(freq_mask)[0]

    if len(bin_indices) == 0:
        raise ValueError(
            f"No frequency bins found in range [{freq_low}, {freq_high}] Hz. "
            f"Available range: [{freqs[1]:.2f}, {freqs[-1]:.2f}] Hz"
        )

    # Extract only the relevant frequency bins
    fft_mag_band = fft_mag[:, bin_indices]  # shape: (n_channels, n_freq_bins)

    # Apply log10 transform with epsilon for stability
    log_fft_mag = np.log10(fft_mag_band + epsilon)

    # Flatten if requested
    if flatten:
        features = log_fft_mag.flatten()
    else:
        features = log_fft_mag

    return {
        'features': features,
        'fft_mag': fft_mag_band,
        'log_fft_mag': log_fft_mag,
        'freq_bins': freqs[bin_indices],
        'n_channels': n_channels,
        'n_freq_bins': len(bin_indices),
    }


def extract_log_fft_simple(
    data: np.ndarray,
    fs: int = 256,
    freq_low: float = 1.0,
    freq_high: float = 47.0,
    epsilon: float = 1e-10
) -> np.ndarray:
    """
    Simplified version that returns only the flattened feature vector.

    This is the minimal interface for use in pipelines.

    Args:
        data: EEG data array of shape (n_channels, n_samples)
        fs: Sampling frequency in Hz
        freq_low: Lower frequency bound in Hz
        freq_high: Upper frequency bound in Hz
        epsilon: Small constant for numerical stability

    Returns:
        Flattened feature vector of shape (n_channels * n_freq_bins,)
    """
    result = extract_log_fft(data, fs, freq_low, freq_high, epsilon, flatten=True)
    return result['features']


def extract_log_fft_per_band(
    data: np.ndarray,
    fs: int = 256,
    bands: Dict[str, Tuple[float, float]] = None,
    epsilon: float = 1e-10
) -> Dict[str, np.ndarray]:
    """
    Extract log FFT features aggregated per frequency band.

    This provides a more compact representation by averaging log power
    within standard EEG frequency bands.

    Args:
        data: EEG data array of shape (n_channels, n_samples)
        fs: Sampling frequency in Hz
        bands: Dict of {band_name: (low_freq, high_freq)}
               Default: delta, theta, alpha, beta, low_gamma
        epsilon: Small constant for numerical stability

    Returns:
        Dictionary with:
            - 'features': Flattened feature vector (n_channels * n_bands,)
            - 'band_power': Matrix of shape (n_channels, n_bands)
            - 'band_names': List of band names in order
    """
    if bands is None:
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'low_gamma': (30, 47),
        }

    n_channels, n_samples = data.shape

    # Get full log FFT
    full_result = extract_log_fft(data, fs, freq_low=0.5, freq_high=47.0,
                                   epsilon=epsilon, flatten=False)
    log_fft_mag = full_result['log_fft_mag']
    freq_bins = full_result['freq_bins']

    # Aggregate per band
    band_power = np.zeros((n_channels, len(bands)))
    band_names = list(bands.keys())

    for i, (band_name, (f_low, f_high)) in enumerate(bands.items()):
        band_mask = (freq_bins >= f_low) & (freq_bins <= f_high)
        if np.any(band_mask):
            # Mean log power in band
            band_power[:, i] = np.mean(log_fft_mag[:, band_mask], axis=1)
        else:
            band_power[:, i] = -10  # Very low power fallback

    return {
        'features': band_power.flatten(),
        'band_power': band_power,
        'band_names': band_names,
        'n_channels': n_channels,
        'n_bands': len(bands),
    }


# =============================================================================
# Validation / Test
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directories to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("=" * 60)
    print("Log FFT Feature Extraction Validation")
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
                print(f"Found ictal: {ictal_seg.source_file}")
                print(f"Found interictal: {interictal_seg.source_file}")

                # Load ictal data
                ictal_data, fs = read_edf_segment(
                    str(subject_dir / ictal_seg.source_file),
                    ictal_seg.start_sec,
                    ictal_seg.end_sec,
                    target_channels=CHANNELS
                )
                ictal_data = preprocess_eeg(ictal_data, fs=int(fs))

                # Load interictal data
                inter_data, fs = read_edf_segment(
                    str(subject_dir / interictal_seg.source_file),
                    interictal_seg.start_sec,
                    interictal_seg.end_sec,
                    target_channels=CHANNELS
                )
                inter_data = preprocess_eeg(inter_data, fs=int(fs))

                print(f"\n--- ICTAL Segment (chb01) ---")
                print(f"Data shape: {ictal_data.shape}")

                result_ictal = extract_log_fft(ictal_data, fs=int(fs))
                print(f"\nLog FFT Results:")
                print(f"  Feature vector shape: {result_ictal['features'].shape}")
                print(f"  Feature range: [{result_ictal['features'].min():.4f}, {result_ictal['features'].max():.4f}]")
                print(f"  n_channels: {result_ictal['n_channels']}")
                print(f"  n_freq_bins: {result_ictal['n_freq_bins']}")
                print(f"  Freq range: [{result_ictal['freq_bins'][0]:.2f}, {result_ictal['freq_bins'][-1]:.2f}] Hz")

                # Band aggregated
                band_result_ictal = extract_log_fft_per_band(ictal_data, fs=int(fs))
                print(f"\nBand-aggregated features:")
                print(f"  Feature shape: {band_result_ictal['features'].shape}")
                print(f"  Bands: {band_result_ictal['band_names']}")
                print(f"  Mean power per band (channel 0): {band_result_ictal['band_power'][0, :]}")

                print(f"\n--- INTERICTAL Segment (chb01) ---")
                print(f"Data shape: {inter_data.shape}")

                result_inter = extract_log_fft(inter_data, fs=int(fs))
                print(f"\nLog FFT Results:")
                print(f"  Feature vector shape: {result_inter['features'].shape}")
                print(f"  Feature range: [{result_inter['features'].min():.4f}, {result_inter['features'].max():.4f}]")

                band_result_inter = extract_log_fft_per_band(inter_data, fs=int(fs))
                print(f"\nBand-aggregated features:")
                print(f"  Mean power per band (channel 0): {band_result_inter['band_power'][0, :]}")

                # Compare ictal vs interictal
                print(f"\n--- Comparison ---")
                mean_diff = np.mean(result_ictal['features']) - np.mean(result_inter['features'])
                print(f"Mean log FFT difference (ictal - interictal): {mean_diff:.4f}")

                for i, band in enumerate(band_result_ictal['band_names']):
                    ictal_mean = np.mean(band_result_ictal['band_power'][:, i])
                    inter_mean = np.mean(band_result_inter['band_power'][:, i])
                    print(f"  {band}: ictal={ictal_mean:.3f}, inter={inter_mean:.3f}, diff={ictal_mean-inter_mean:+.3f}")

            else:
                print("Could not find both segment types")
        else:
            raise FileNotFoundError("Data directory not found")

    except Exception as e:
        print(f"\nUsing synthetic data: {e}")

        # Synthetic data test
        np.random.seed(42)
        n_channels = 8
        n_samples = 256 * 10
        fs = 256

        t = np.arange(n_samples) / fs

        # Ictal: strong 10 Hz activity
        ictal_data = np.array([
            3.0 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 0.5))
            + 0.5 * np.random.randn(n_samples)
            for _ in range(n_channels)
        ])

        # Interictal: broadband noise
        inter_data = np.random.randn(n_channels, n_samples)

        print(f"\n--- Synthetic ICTAL ---")
        print(f"Data shape: {ictal_data.shape}")
        result = extract_log_fft(ictal_data, fs=fs)
        print(f"Feature shape: {result['features'].shape}")
        print(f"Feature range: [{result['features'].min():.4f}, {result['features'].max():.4f}]")
        print(f"Expected: {n_channels} x {result['n_freq_bins']} = {n_channels * result['n_freq_bins']} features")

        print(f"\n--- Synthetic INTERICTAL ---")
        print(f"Data shape: {inter_data.shape}")
        result = extract_log_fft(inter_data, fs=fs)
        print(f"Feature shape: {result['features'].shape}")
        print(f"Feature range: [{result['features'].min():.4f}, {result['features'].max():.4f}]")

        # Simple function test
        print(f"\n--- Simple API Test ---")
        simple_features = extract_log_fft_simple(ictal_data, fs=fs)
        print(f"Simple API output shape: {simple_features.shape}")

    print("\n" + "=" * 60)
    print("Validation complete")
    print("=" * 60)
