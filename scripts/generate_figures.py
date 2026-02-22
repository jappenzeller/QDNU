#!/usr/bin/env python3
"""
E-001: Generate Publication Figures

Generates three publication-quality figures for the QDNU paper:
1. Spectrogram comparison (2x2 grid)
2. Inter-channel coherence
3. Manifold polarity schematic

Uses existing EDF loading infrastructure from sagemaker/train_chbmit.py.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import numpy as np
from scipy.signal import spectrogram, coherence

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

DATA_ROOT = Path("H:/Data/PythonDNU/EEG/chbmit")
METADATA_PATH = Path("data/chb_mit_full_metadata.json")
TRAJECTORY_PATH = Path("visualization_data/circuit_trajectories.json")
OUTPUT_DIR = Path("figures")

FS = 256  # Sample rate
TARGET_CHANNELS = ["FP1-F7", "F7-T7"]  # For coherence

# Hardware AUC values from calibrated results
HW_AUC = {
    'chb01': 0.686,  # Standard polarity
    'chb11': 0.717,  # Calibrated (inverted polarity)
}

# =============================================================================
# EDF Loading (from existing pipeline)
# =============================================================================

def normalize_channel_label(label: str) -> str:
    """Normalize channel label for matching."""
    return label.strip().upper().replace(' ', '').replace('EEG', '').strip()


def read_edf_segment(filepath: str, start_sec: float, end_sec: float,
                     target_channels: List[str] = None) -> Tuple[Optional[np.ndarray], float]:
    """
    Read a segment from an EDF file with channel selection.
    Uses pyedflib as in the existing pipeline.
    """
    import pyedflib

    try:
        with pyedflib.EdfReader(str(filepath)) as f:
            n_signals = f.signals_in_file
            fs = f.getSampleFrequency(0)

            start_sample = int(start_sec * fs)
            end_sample = int(end_sec * fs)
            n_samples = end_sample - start_sample

            if n_samples <= 0:
                return None, fs

            if target_channels is None:
                channel_indices = list(range(n_signals))
            else:
                label_to_idx = {}
                for i in range(n_signals):
                    label = normalize_channel_label(f.getLabel(i))
                    label_to_idx[label] = i

                channel_indices = []
                for target in target_channels:
                    target_norm = normalize_channel_label(target)
                    if target_norm in label_to_idx:
                        channel_indices.append(label_to_idx[target_norm])
                    else:
                        # Try partial match
                        for label, idx in label_to_idx.items():
                            if target_norm in label or label in target_norm:
                                channel_indices.append(idx)
                                break

                if len(channel_indices) != len(target_channels):
                    print(f"Warning: Only found {len(channel_indices)}/{len(target_channels)} channels")
                    if len(channel_indices) == 0:
                        return None, fs

            # Read data
            data = np.zeros((len(channel_indices), n_samples))
            for i, ch_idx in enumerate(channel_indices):
                signal = f.readSignal(ch_idx)
                if len(signal) >= end_sample:
                    data[i, :] = signal[start_sample:end_sample]
                else:
                    available = len(signal) - start_sample
                    if available > 0:
                        data[i, :available] = signal[start_sample:]

            return data, fs

    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, FS


def get_channel_labels(filepath: str) -> List[str]:
    """Get channel labels from EDF file."""
    import pyedflib

    try:
        with pyedflib.EdfReader(str(filepath)) as f:
            return [normalize_channel_label(f.getLabel(i)) for i in range(f.signals_in_file)]
    except Exception as e:
        print(f"Error getting labels from {filepath}: {e}")
        return []


# =============================================================================
# Figure 1: Spectrogram Comparison
# =============================================================================

def generate_spectrogram_figure(metadata: Dict, save_path: Path):
    """Generate 2x2 spectrogram comparison figure."""

    print("Generating Figure 1: Spectrograms...")

    # Configuration
    segment_duration = 10  # seconds
    channel = "FP1-F7"

    # Get segment info for each patient/state
    segments = {}

    for patient in ['chb01', 'chb11']:
        subj_data = metadata['subjects'][patient]

        # Ictal: middle of first seizure
        seizure = subj_data['seizures'][0]
        edf_file = DATA_ROOT / patient / seizure['file']
        seizure_start = seizure['start']
        seizure_end = seizure['end']
        seizure_mid = (seizure_start + seizure_end) // 2
        ictal_start = max(seizure_start, seizure_mid - segment_duration // 2)

        segments[f'{patient}_ictal'] = {
            'file': edf_file,
            'start': ictal_start,
            'end': ictal_start + segment_duration,
        }

        # Interictal: from a file without seizures, 30+ min from any seizure
        # For chb01: use chb01_01.edf (no seizures)
        # For chb11: use chb11_01.edf (no seizures)
        interictal_file = DATA_ROOT / patient / f"{patient}_01.edf"
        if interictal_file.exists():
            segments[f'{patient}_interictal'] = {
                'file': interictal_file,
                'start': 1800,  # 30 minutes into recording
                'end': 1800 + segment_duration,
            }
        else:
            # Fallback to first file
            first_file = subj_data['edf_files'][0]['filename']
            segments[f'{patient}_interictal'] = {
                'file': DATA_ROOT / patient / first_file,
                'start': 1800,
                'end': 1800 + segment_duration,
            }

    # Read data and compute spectrograms
    spectrograms = {}
    vmin, vmax = float('inf'), float('-inf')

    for key, seg_info in segments.items():
        data, fs = read_edf_segment(
            str(seg_info['file']),
            seg_info['start'],
            seg_info['end'],
            target_channels=[channel]
        )

        if data is None or data.shape[0] == 0:
            print(f"Warning: Could not read data for {key}")
            continue

        signal = data[0, :]  # First (only) channel

        # Compute spectrogram
        f, t, Sxx = spectrogram(signal, fs=int(fs), window='hann',
                                nperseg=256, noverlap=128, scaling='density')

        # Convert to dB
        Sxx_db = 10 * np.log10(Sxx + 1e-12)

        # Limit to 0-80 Hz
        freq_mask = f <= 80

        spectrograms[key] = {
            'f': f[freq_mask],
            't': t,
            'Sxx_db': Sxx_db[freq_mask, :],
        }

        # Track global min/max for shared colorbar
        vmin = min(vmin, np.percentile(Sxx_db[freq_mask, :], 2))
        vmax = max(vmax, np.percentile(Sxx_db[freq_mask, :], 98))

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=300)

    # Panel layout:
    # [chb01 ictal]    [chb01 interictal]
    # [chb11 ictal]    [chb11 interictal]

    panel_config = [
        ('chb01_ictal', axes[0, 0], f'chb01 — Ictal (HW AUC: {HW_AUC["chb01"]:.3f})'),
        ('chb01_interictal', axes[0, 1], 'chb01 — Interictal'),
        ('chb11_ictal', axes[1, 0], f'chb11 — Ictal (HW AUC: {HW_AUC["chb11"]:.3f} cal.)'),
        ('chb11_interictal', axes[1, 1], 'chb11 — Interictal'),
    ]

    ims = []
    for key, ax, title in panel_config:
        if key in spectrograms:
            spec = spectrograms[key]
            im = ax.pcolormesh(spec['t'], spec['f'], spec['Sxx_db'],
                              shading='gouraud', cmap='inferno',
                              vmin=vmin, vmax=vmax)
            ims.append(im)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency (Hz)', fontsize=10)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylim(0, 80)

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    if ims:
        cbar = fig.colorbar(ims[0], cax=cbar_ax)
        cbar.set_label('Power (dB)', fontsize=10)

    # Main title
    fig.suptitle(f'EEG Spectrograms — Channel {channel}', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 0.88, 0.95])

    # Save
    fig.savefig(save_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    fig.savefig(save_path.with_suffix('.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {save_path.with_suffix('.pdf')}")
    print(f"  Saved: {save_path.with_suffix('.png')}")


# =============================================================================
# Figure 2: Inter-channel Coherence
# =============================================================================

def generate_coherence_figure(metadata: Dict, save_path: Path):
    """Generate coherence comparison figure."""

    print("Generating Figure 2: Coherence...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

    for idx, patient in enumerate(['chb01', 'chb11']):
        ax = axes[idx]
        subj_data = metadata['subjects'][patient]

        # Collect multiple ictal segments
        ictal_coherences = []
        for seizure in subj_data['seizures'][:3]:  # Up to 3 seizures
            edf_file = DATA_ROOT / patient / seizure['file']
            if not edf_file.exists():
                continue

            # Read segment from middle of seizure
            mid = (seizure['start'] + seizure['end']) // 2
            data, fs = read_edf_segment(
                str(edf_file),
                max(0, mid - 5),
                mid + 5,
                target_channels=TARGET_CHANNELS
            )

            if data is not None and data.shape[0] >= 2:
                f, Cxy = coherence(data[0, :], data[1, :], fs=int(fs), nperseg=256)
                freq_mask = f <= 80
                ictal_coherences.append(Cxy[freq_mask])

        # Collect interictal segments
        interictal_coherences = []
        interictal_file = DATA_ROOT / patient / f"{patient}_01.edf"
        if interictal_file.exists():
            # Sample multiple windows from interictal recording
            for start in [600, 1200, 1800, 2400, 3000]:
                data, fs = read_edf_segment(
                    str(interictal_file),
                    start,
                    start + 10,
                    target_channels=TARGET_CHANNELS
                )

                if data is not None and data.shape[0] >= 2:
                    f, Cxy = coherence(data[0, :], data[1, :], fs=int(fs), nperseg=256)
                    freq_mask = f <= 80
                    interictal_coherences.append(Cxy[freq_mask])

        # Get frequency axis
        f_plot = f[freq_mask]

        # Plot ictal (red/orange)
        if ictal_coherences:
            ictal_array = np.array(ictal_coherences)
            ictal_mean = ictal_array.mean(axis=0)
            ictal_std = ictal_array.std(axis=0)

            ax.fill_between(f_plot, ictal_mean - ictal_std, ictal_mean + ictal_std,
                           alpha=0.3, color='#ff6b35')
            ax.plot(f_plot, ictal_mean, color='#ff6b35', linewidth=2, label='Ictal (mean ± std)')

        # Plot interictal (blue)
        if interictal_coherences:
            inter_array = np.array(interictal_coherences)
            inter_mean = inter_array.mean(axis=0)
            inter_std = inter_array.std(axis=0)

            ax.fill_between(f_plot, inter_mean - inter_std, inter_mean + inter_std,
                           alpha=0.3, color='#4a90d9')
            ax.plot(f_plot, inter_mean, color='#4a90d9', linewidth=2, label='Interictal (mean ± std)')

        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Coherence', fontsize=10)
        ax.set_title(f'{patient}', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 80)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Inter-channel Coherence ({TARGET_CHANNELS[0]} vs {TARGET_CHANNELS[1]})',
                fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    # Save
    fig.savefig(save_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight')
    fig.savefig(save_path.with_suffix('.png'), format='png', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  Saved: {save_path.with_suffix('.pdf')}")
    print(f"  Saved: {save_path.with_suffix('.png')}")


# =============================================================================
# Figure 3: Manifold Polarity Schematic
# =============================================================================

def generate_manifold_figure(save_path: Path):
    """Generate manifold polarity schematic using actual PCA coordinates."""

    print("Generating Figure 3: Manifold Polarity...")

    # Load trajectory data
    with open(TRAJECTORY_PATH) as f:
        traj_data = json.load(f)

    # Set dark style
    plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor('#080810')
    ax.set_facecolor('#0d0d1a')

    # Colors matching visualization
    colors = {
        'chb01_ictal': '#ff4455',
        'chb01_interictal': '#4488ff',
        'chb11_ictal': '#ff8800',
        'chb11_interictal': '#44aaff',
    }

    # Extract trajectories
    trajectories = {}
    for patient in ['chb01', 'chb11']:
        patient_data = traj_data['patients'][patient]
        for state in ['ictal', 'interictal']:
            key = f'{patient}_{state}'
            traj = patient_data[state]

            # Extract PCA coordinates
            pcs = np.array([p['pca'] for p in traj])
            boundary_dists = np.array([p['boundary_distance'] for p in traj])

            trajectories[key] = {
                'pcs': pcs,
                'boundary_dists': boundary_dists,
                'final_boundary': boundary_dists[-1],
            }

    # Plot trajectories
    for key, data in trajectories.items():
        pcs = data['pcs']
        color = colors[key]

        # Plot trajectory as thin line
        ax.plot(pcs[:, 0], pcs[:, 1], color=color, linewidth=1.5, alpha=0.7)

        # Plot endpoint as filled circle
        ax.scatter(pcs[-1, 0], pcs[-1, 1], c=color, s=100, zorder=5, edgecolor='white', linewidth=1)

        # Plot start point as small circle
        ax.scatter(pcs[0, 0], pcs[0, 1], c=color, s=30, zorder=4, alpha=0.5)

    # Decision surface (vertical dashed line at PC1=0)
    ax.axvline(x=0, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label='Decision Surface')

    # Annotations
    annotations = [
        ('chb01_ictal', 'chb01 ictal\n(dist: -0.190)', (-0.15, 0.02)),
        ('chb11_ictal', 'chb11 ictal\n(dist: +0.211)', (0.15, -0.05)),
        ('chb11_interictal', 'chb11 interictal\n(dist: +0.001)', (0.05, 0.12)),
        ('chb01_interictal', 'chb01 interictal\n(dist: +0.068)', (0.08, -0.12)),
    ]

    for key, text, offset in annotations:
        pcs = trajectories[key]['pcs']
        endpoint = pcs[-1, :2]
        ax.annotate(text, xy=endpoint, xytext=(endpoint[0] + offset[0], endpoint[1] + offset[1]),
                   fontsize=9, color=colors[key],
                   arrowprops=dict(arrowstyle='->', color=colors[key], alpha=0.7),
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d0d1a', edgecolor=colors[key], alpha=0.8))

    # Legend
    legend_elements = [
        mpatches.Patch(color='#ff4455', label='chb01 ictal'),
        mpatches.Patch(color='#4488ff', label='chb01 interictal'),
        mpatches.Patch(color='#ff8800', label='chb11 ictal'),
        mpatches.Patch(color='#44aaff', label='chb11 interictal'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
             facecolor='#0d0d1a', edgecolor='white', framealpha=0.8)

    # Labels
    ax.set_xlabel('PC1 — Boundary Separation (81.4%)', fontsize=11, color='white')
    ax.set_ylabel('PC2 — State Entropy (18.5%)', fontsize=11, color='white')
    ax.set_title('Quantum Circuit Trajectories — Polarity Inversion',
                fontsize=14, fontweight='bold', color='#00d4ff')

    # Grid
    ax.grid(True, alpha=0.2, color='white')

    # Axis styling
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.tick_params(colors='white')

    plt.tight_layout()

    # Save
    fig.savefig(save_path.with_suffix('.pdf'), format='pdf', bbox_inches='tight',
               facecolor=fig.get_facecolor())
    fig.savefig(save_path.with_suffix('.png'), format='png', dpi=300, bbox_inches='tight',
               facecolor=fig.get_facecolor())
    plt.close(fig)

    # Reset style
    plt.style.use('default')

    print(f"  Saved: {save_path.with_suffix('.pdf')}")
    print(f"  Saved: {save_path.with_suffix('.png')}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Generate all publication figures."""

    print("=" * 60)
    print("E-001: Generating Publication Figures")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load metadata
    print(f"\nLoading metadata from {METADATA_PATH}...")
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    # Check data availability
    print(f"Data root: {DATA_ROOT}")
    if not DATA_ROOT.exists():
        print(f"ERROR: Data root does not exist: {DATA_ROOT}")
        print("Skipping Figures 1 and 2 (require EDF data)")
        generate_manifold_figure(OUTPUT_DIR / 'fig3_manifold_polarity')
        return

    # Generate figures
    print("\n" + "-" * 40)
    generate_spectrogram_figure(metadata, OUTPUT_DIR / 'fig1_spectrograms')

    print("\n" + "-" * 40)
    generate_coherence_figure(metadata, OUTPUT_DIR / 'fig2_coherence')

    print("\n" + "-" * 40)
    generate_manifold_figure(OUTPUT_DIR / 'fig3_manifold_polarity')

    print("\n" + "=" * 60)
    print("E-001 COMPLETE")
    print("=" * 60)
    print(f"\nGenerated files in {OUTPUT_DIR}/:")
    for f in sorted(OUTPUT_DIR.glob('*')):
        print(f"  {f.name}")


if __name__ == '__main__':
    main()
