#!/usr/bin/env python3
"""
Preprocessing Validation Script for CHB-MIT EEG

Validates the corrected preprocessing pipeline:
- 0.5-40 Hz bandpass (5th order Butterworth, zero-phase)
- 60 Hz notch filter (Q=30)
- Per-channel z-score normalization
- Non-overlapping 1-second windows

Produces diagnostic plots:
1. Raw vs filtered signal overlay
2. Power spectral density before/after filtering
3. Window counts per class summary

Usage:
    python preprocessing_validation.py --data_dir /path/to/chb-mit --patient chb01
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import welch
from pathlib import Path
import argparse
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# PREPROCESSING FUNCTION (Prompt 1 reference implementation)
# =============================================================================

def preprocess_eeg(raw_signal: np.ndarray, fs: int = 256,
                   bandpass: tuple = (0.5, 40), notch: float = 60,
                   window_sec: float = 1, normalize: str = 'channel'
                   ) -> np.ndarray:
    """
    Preprocess EEG for seizure detection with validated clinical pipeline.

    Steps:
    1. DC offset removal (per channel)
    2. Bandpass filter (0.5-40 Hz, 5th order Butterworth, zero-phase)
    3. Notch filter (60 Hz, Q=30)
    4. Per-channel z-score normalization
    5. Non-overlapping windowing

    Args:
        raw_signal: EEG array (n_channels, n_samples)
        fs: Sampling frequency (default 256 Hz for CHB-MIT)
        bandpass: (low, high) cutoff frequencies in Hz
        notch: Notch filter frequency (60 Hz for US, 50 Hz for EU)
        window_sec: Window size in seconds (default 1s)
        normalize: 'channel' for per-channel z-score, 'global', or None

    Returns:
        Windowed array of shape (n_windows, n_channels, window_samples)
    """
    n_channels, n_samples = raw_signal.shape
    window_samples = int(window_sec * fs)

    # Work on copy
    data = raw_signal.astype(np.float64).copy()

    # Step 1: DC offset removal per channel
    data = data - np.mean(data, axis=1, keepdims=True)

    # Step 2: Bandpass filter (0.5-40 Hz)
    nyq = fs / 2
    low = bandpass[0] / nyq
    high = min(bandpass[1] / nyq, 0.99)

    if low > 0 and low < high:
        b, a = signal.butter(5, [low, high], btype='band')
        # Zero-phase filtering with filtfilt (critical for temporal features)
        data = signal.filtfilt(b, a, data, axis=1)

    # Step 3: Notch filter at 60 Hz
    if notch and notch < nyq:
        b_notch, a_notch = signal.iirnotch(notch, Q=30, fs=fs)
        data = signal.filtfilt(b_notch, a_notch, data, axis=1)

    # Step 4: Per-channel z-score normalization
    if normalize == 'channel':
        for ch in range(n_channels):
            std = np.std(data[ch])
            if std > 1e-10:
                data[ch] = (data[ch] - np.mean(data[ch])) / std
    elif normalize == 'global':
        std = np.std(data)
        if std > 1e-10:
            data = (data - np.mean(data)) / std

    # Step 5: Non-overlapping windows (critical for CV integrity)
    n_windows = n_samples // window_samples
    if n_windows == 0:
        return np.array([])

    # Truncate to exact window boundaries
    data = data[:, :n_windows * window_samples]

    # Reshape to (n_windows, n_channels, window_samples)
    windows = data.reshape(n_channels, n_windows, window_samples)
    windows = np.transpose(windows, (1, 0, 2))  # (n_windows, n_channels, window_samples)

    return windows


# =============================================================================
# DIAGNOSTIC PLOTTING
# =============================================================================

def plot_raw_vs_filtered(raw: np.ndarray, filtered: np.ndarray, fs: int,
                         channel: int = 0, duration_sec: float = 5,
                         save_path: str = None):
    """Plot raw vs filtered signal overlay for visual inspection."""

    n_samples = min(int(duration_sec * fs), raw.shape[1])
    t = np.arange(n_samples) / fs

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    # Raw signal
    axes[0].plot(t, raw[channel, :n_samples], 'b-', alpha=0.7, linewidth=0.5)
    axes[0].set_ylabel('Amplitude (μV)')
    axes[0].set_title(f'Raw EEG Signal - Channel {channel}')
    axes[0].grid(True, alpha=0.3)

    # Filtered signal
    axes[1].plot(t, filtered[channel, :n_samples], 'g-', alpha=0.7, linewidth=0.5)
    axes[1].set_ylabel('Amplitude (z-score)')
    axes[1].set_title(f'Filtered EEG (0.5-40 Hz + 60 Hz notch + z-score) - Channel {channel}')
    axes[1].grid(True, alpha=0.3)

    # Overlay
    ax3 = axes[2]
    # Normalize raw for overlay comparison
    raw_norm = (raw[channel, :n_samples] - np.mean(raw[channel, :n_samples])) / np.std(raw[channel, :n_samples])
    ax3.plot(t, raw_norm, 'b-', alpha=0.5, linewidth=0.5, label='Raw (normalized)')
    ax3.plot(t, filtered[channel, :n_samples], 'g-', alpha=0.7, linewidth=0.5, label='Filtered')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude (normalized)')
    ax3.set_title('Overlay Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_psd_comparison(raw: np.ndarray, filtered: np.ndarray, fs: int,
                        channel: int = 0, save_path: str = None):
    """Plot power spectral density before/after filtering."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Compute PSD using Welch's method
    nperseg = min(1024, raw.shape[1] // 4)

    freqs_raw, psd_raw = welch(raw[channel], fs=fs, nperseg=nperseg)
    freqs_filt, psd_filt = welch(filtered[channel], fs=fs, nperseg=nperseg)

    # Plot raw PSD
    axes[0].semilogy(freqs_raw, psd_raw, 'b-', linewidth=1)
    axes[0].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='0.5 Hz cutoff')
    axes[0].axvline(x=40, color='r', linestyle='--', alpha=0.5, label='40 Hz cutoff')
    axes[0].axvline(x=60, color='orange', linestyle='--', alpha=0.5, label='60 Hz notch')
    axes[0].set_xlim([0, 80])
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('PSD (V²/Hz)')
    axes[0].set_title(f'Power Spectral Density - Raw Signal - Channel {channel}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot filtered PSD
    axes[1].semilogy(freqs_filt, psd_filt, 'g-', linewidth=1)
    axes[1].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='0.5 Hz cutoff')
    axes[1].axvline(x=40, color='r', linestyle='--', alpha=0.5, label='40 Hz cutoff')
    axes[1].axvline(x=60, color='orange', linestyle='--', alpha=0.5, label='60 Hz notch')
    axes[1].set_xlim([0, 80])
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('PSD (normalized)')
    axes[1].set_title(f'Power Spectral Density - Filtered Signal - Channel {channel}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


def plot_filter_response(fs: int = 256, bandpass: tuple = (0.5, 40),
                         notch: float = 60, save_path: str = None):
    """Plot the frequency response of the filter chain."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    nyq = fs / 2

    # Bandpass filter response
    low = bandpass[0] / nyq
    high = bandpass[1] / nyq
    b_bp, a_bp = signal.butter(5, [low, high], btype='band')
    w_bp, h_bp = signal.freqz(b_bp, a_bp, worN=2000, fs=fs)

    axes[0].plot(w_bp, 20 * np.log10(np.abs(h_bp) + 1e-10), 'b-', linewidth=1.5)
    axes[0].axvline(x=bandpass[0], color='r', linestyle='--', alpha=0.7)
    axes[0].axvline(x=bandpass[1], color='r', linestyle='--', alpha=0.7)
    axes[0].axhline(y=-3, color='gray', linestyle=':', alpha=0.7, label='-3 dB cutoff')
    axes[0].set_xlim([0, 80])
    axes[0].set_ylim([-60, 5])
    axes[0].set_xlabel('Frequency (Hz)')
    axes[0].set_ylabel('Magnitude (dB)')
    axes[0].set_title(f'Bandpass Filter Response ({bandpass[0]}-{bandpass[1]} Hz, 5th order Butterworth)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Notch filter response
    b_notch, a_notch = signal.iirnotch(notch, Q=30, fs=fs)
    w_notch, h_notch = signal.freqz(b_notch, a_notch, worN=2000, fs=fs)

    axes[1].plot(w_notch, 20 * np.log10(np.abs(h_notch) + 1e-10), 'g-', linewidth=1.5)
    axes[1].axvline(x=notch, color='orange', linestyle='--', alpha=0.7, label=f'{notch} Hz notch')
    axes[1].set_xlim([40, 80])
    axes[1].set_ylim([-40, 5])
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude (dB)')
    axes[1].set_title(f'Notch Filter Response ({notch} Hz, Q=30)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# CHB-MIT DATA LOADING
# =============================================================================

def load_chbmit_edf(edf_path: Path) -> tuple:
    """Load CHB-MIT EDF file and return (data, fs, channel_names)."""
    try:
        import mne
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
        data = raw.get_data() * 1e6  # Convert to microvolts
        fs = int(raw.info['sfreq'])
        ch_names = raw.info['ch_names']
        return data, fs, ch_names
    except ImportError:
        print("MNE not installed. Install with: pip install mne")
        return None, None, None


def load_chbmit_summary(patient_dir: Path) -> dict:
    """Parse CHB-MIT summary file for seizure times."""
    summary_file = patient_dir / f"{patient_dir.name}-summary.txt"

    if not summary_file.exists():
        return {}

    seizures = {}
    current_file = None

    with open(summary_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('File Name:'):
                current_file = line.split(':')[1].strip()
                seizures[current_file] = []
            elif 'Seizure Start Time' in line and current_file:
                try:
                    start = int(line.split(':')[1].strip().replace(' seconds', ''))
                    seizures[current_file].append({'start': start})
                except:
                    pass
            elif 'Seizure End Time' in line and current_file and seizures[current_file]:
                try:
                    end = int(line.split(':')[1].strip().replace(' seconds', ''))
                    seizures[current_file][-1]['end'] = end
                except:
                    pass

    return seizures


# =============================================================================
# VALIDATION RUNNER
# =============================================================================

def run_validation(data_dir: Path, patient: str, output_dir: Path = None):
    """Run full preprocessing validation on a CHB-MIT patient."""

    patient_dir = data_dir / patient

    if not patient_dir.exists():
        print(f"Patient directory not found: {patient_dir}")
        return

    if output_dir is None:
        output_dir = Path(__file__).parent / 'validation_output'
    output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"PREPROCESSING VALIDATION - {patient}")
    print(f"{'='*60}")

    # Load seizure info
    seizures = load_chbmit_summary(patient_dir)

    # Find first file with seizure
    edf_files = sorted(patient_dir.glob('*.edf'))

    if not edf_files:
        print("No EDF files found")
        return

    test_file = None
    for edf in edf_files:
        if edf.name in seizures and seizures[edf.name]:
            test_file = edf
            break

    if test_file is None:
        test_file = edf_files[0]
        print(f"No seizure files found, using: {test_file.name}")
    else:
        print(f"Using seizure file: {test_file.name}")
        print(f"Seizures: {seizures[test_file.name]}")

    # Load data
    print("\nLoading EDF file...")
    data, fs, ch_names = load_chbmit_edf(test_file)

    if data is None:
        print("Failed to load data")
        return

    print(f"Data shape: {data.shape}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Channels: {len(ch_names)}")
    print(f"Duration: {data.shape[1] / fs:.1f} seconds")

    # Filter to EEG channels only (exclude ECG, etc.)
    # Note: CHB-MIT uses bipolar montage like "FP1-F7" - don't filter on '-'
    eeg_channels = [i for i, ch in enumerate(ch_names)
                    if not any(x in ch.upper() for x in ['ECG', 'VNS', 'PHOTIC', 'EKG', 'LOC', 'ROC'])]

    if len(eeg_channels) < len(ch_names):
        print(f"Filtering to {len(eeg_channels)} EEG channels")
        data = data[eeg_channels]

    # Take 60 seconds for validation
    duration = min(60, data.shape[1] / fs)
    n_samples = int(duration * fs)
    raw_segment = data[:, :n_samples]

    print(f"\nProcessing {duration:.0f} second segment...")

    # Apply preprocessing (get continuous filtered signal first for plots)
    filtered = preprocess_eeg(raw_segment, fs=fs, bandpass=(0.5, 40),
                              notch=60, window_sec=duration, normalize='channel')

    # Flatten back to continuous for plotting
    if len(filtered) > 0:
        filtered_continuous = filtered[0]  # First (only) window
    else:
        print("No windows produced - segment too short")
        return

    # Apply windowing
    windows = preprocess_eeg(raw_segment, fs=fs, bandpass=(0.5, 40),
                             notch=60, window_sec=1, normalize='channel')

    print(f"Windows produced: {windows.shape[0]} (1-second, non-overlapping)")

    # Plot diagnostics
    print("\nGenerating diagnostic plots...")

    # 1. Filter frequency response
    plot_filter_response(
        fs=fs, bandpass=(0.5, 40), notch=60,
        save_path=str(output_dir / f'{patient}_filter_response.png')
    )

    # 2. Raw vs filtered
    plot_raw_vs_filtered(
        raw_segment, filtered_continuous, fs=fs, channel=0, duration_sec=5,
        save_path=str(output_dir / f'{patient}_raw_vs_filtered.png')
    )

    # 3. PSD comparison
    plot_psd_comparison(
        raw_segment, filtered_continuous, fs=fs, channel=0,
        save_path=str(output_dir / f'{patient}_psd_comparison.png')
    )

    # Print summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"Patient: {patient}")
    print(f"Sampling rate: {fs} Hz")
    print(f"Channels: {data.shape[0]}")
    print(f"Duration: {data.shape[1] / fs:.1f} seconds")
    print(f"")
    print("Preprocessing pipeline:")
    print("  1. DC offset removal (per channel)")
    print("  2. Bandpass filter: 0.5-40 Hz (5th order Butterworth, zero-phase)")
    print("  3. Notch filter: 60 Hz (Q=30, zero-phase)")
    print("  4. Z-score normalization: per channel")
    print("  5. Windowing: 1 second, non-overlapping")
    print(f"")
    print(f"Windows produced: {windows.shape[0]}")
    print(f"Window shape: {windows.shape[1:]} (channels, samples)")
    print(f"")
    print(f"Output directory: {output_dir}")
    print(f"  - {patient}_filter_response.png")
    print(f"  - {patient}_raw_vs_filtered.png")
    print(f"  - {patient}_psd_comparison.png")

    # Validate window integrity
    print(f"\n{'='*60}")
    print("CV INTEGRITY CHECK")
    print(f"{'='*60}")
    print("Window overlap: NONE (non-overlapping for CV integrity)")
    print("Window size: 256 samples (1 second at 256 Hz)")
    print("")
    print("NOTE: For cross-validation, splits should occur at the SEIZURE level,")
    print("      not the window level. This prevents temporal leakage where")
    print("      windows from the same seizure appear in both train and test sets.")

    return {
        'patient': patient,
        'fs': fs,
        'n_channels': data.shape[0],
        'duration_sec': data.shape[1] / fs,
        'n_windows': windows.shape[0],
        'window_shape': windows.shape[1:]
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate EEG preprocessing pipeline')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to CHB-MIT dataset directory')
    parser.add_argument('--patient', type=str, default='chb01',
                        help='Patient ID (e.g., chb01)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots')

    args = parser.parse_args()

    # Try common data locations
    data_paths = [
        Path(args.data_dir) if args.data_dir else None,
        Path.home() / 'data' / 'chb-mit',
        Path('/data/chb-mit'),
        Path('../data/chb-mit'),
        Path('../../data/chb-mit'),
    ]

    data_dir = None
    for p in data_paths:
        if p and p.exists():
            data_dir = p
            break

    if data_dir is None:
        print("CHB-MIT data directory not found.")
        print("Please specify with --data_dir /path/to/chb-mit")
        print("\nRunning with synthetic data for demonstration...")

        # Generate synthetic data for demo
        fs = 256
        duration = 30  # seconds
        n_channels = 23
        n_samples = fs * duration

        # Create synthetic EEG with known characteristics
        t = np.arange(n_samples) / fs
        np.random.seed(42)

        raw_signal = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            # Mix of EEG bands
            raw_signal[ch] = (
                2.0 * np.sin(2 * np.pi * 2 * t) +  # Delta (2 Hz)
                1.5 * np.sin(2 * np.pi * 6 * t) +  # Theta (6 Hz)
                1.0 * np.sin(2 * np.pi * 10 * t) + # Alpha (10 Hz)
                0.5 * np.sin(2 * np.pi * 20 * t) + # Beta (20 Hz)
                0.2 * np.sin(2 * np.pi * 60 * t) + # Line noise (60 Hz)
                0.1 * np.sin(2 * np.pi * 100 * t) + # High freq artifact
                0.5 * np.random.randn(n_samples) +  # Noise
                10.0  # DC offset
            )

        print(f"\nSynthetic data: {n_channels} channels, {duration} seconds, {fs} Hz")

        # Process
        windows = preprocess_eeg(raw_signal, fs=fs, bandpass=(0.5, 40),
                                 notch=60, window_sec=1, normalize='channel')

        print(f"Windows produced: {windows.shape[0]} (1-second, non-overlapping)")

        # Create output directory
        output_dir = Path(__file__).parent / 'validation_output'
        output_dir.mkdir(exist_ok=True)

        # Get continuous filtered for plotting
        filtered = preprocess_eeg(raw_signal, fs=fs, bandpass=(0.5, 40),
                                  notch=60, window_sec=duration, normalize='channel')
        filtered_continuous = filtered[0]

        # Generate plots
        plot_filter_response(fs=fs, bandpass=(0.5, 40), notch=60,
                            save_path=str(output_dir / 'synthetic_filter_response.png'))

        plot_raw_vs_filtered(raw_signal, filtered_continuous, fs=fs, channel=0, duration_sec=5,
                            save_path=str(output_dir / 'synthetic_raw_vs_filtered.png'))

        plot_psd_comparison(raw_signal, filtered_continuous, fs=fs, channel=0,
                           save_path=str(output_dir / 'synthetic_psd_comparison.png'))

        print(f"\nDiagnostic plots saved to: {output_dir}")
        print("  - synthetic_filter_response.png")
        print("  - synthetic_raw_vs_filtered.png")
        print("  - synthetic_psd_comparison.png")

    else:
        output_dir = Path(args.output_dir) if args.output_dir else None
        run_validation(data_dir, args.patient, output_dir)
