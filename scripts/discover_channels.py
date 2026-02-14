#!/usr/bin/env python3
"""
================================================================================
CHB-MIT CHANNEL DISCOVERY
================================================================================

Discovers common EEG channels across all subjects in the CHB-MIT dataset.
This is needed because subjects have varying channel counts (23-24 channels)
and we need a consistent set of channels for feature extraction.

Usage:
    python scripts/discover_channels.py

================================================================================
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

# Try to import pyedflib
try:
    import pyedflib
except ImportError:
    print("Installing pyedflib...")
    os.system(f"{sys.executable} -m pip install pyedflib --no-deps -q")
    import pyedflib


# CHB-MIT data directory
DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")


def normalize_channel_label(label: str) -> str:
    """
    Normalize channel label for consistent matching.

    - Strip whitespace
    - Convert to uppercase
    - Remove common prefixes like 'EEG '
    - Standardize separators
    """
    label = label.strip().upper()

    # Remove common prefixes
    for prefix in ['EEG ', 'EEG-']:
        if label.startswith(prefix):
            label = label[len(prefix):]

    # Standardize separators (some use - some use -)
    label = label.replace('--', '-')

    return label


def get_edf_channels(edf_path: Path) -> set:
    """Read channel labels from an EDF file."""
    try:
        with pyedflib.EdfReader(str(edf_path)) as f:
            n_channels = f.signals_in_file
            channels = set()
            for i in range(n_channels):
                label = f.getLabel(i)
                normalized = normalize_channel_label(label)
                # Skip non-EEG channels
                if normalized in ['ECG', 'VNS', '-', 'LOC-ROC', '.']:
                    continue
                channels.add(normalized)
            return channels
    except Exception as e:
        print(f"  Error reading {edf_path.name}: {e}")
        return set()


def discover_common_channels():
    """Discover channels common to all subjects."""

    if not DATA_DIR.exists():
        print(f"Error: Data directory not found: {DATA_DIR}")
        return None

    # Get all subject directories
    subjects = sorted([d for d in DATA_DIR.iterdir()
                      if d.is_dir() and d.name.startswith('chb')])

    # Known problematic subjects:
    # - chb12: Major montage changes mid-recording (different reference systems)
    # - chb24: Re-recording of chb01, often has no EDF files or uses same data
    EXCLUDE_SUBJECTS = {'chb12', 'chb24'}

    print(f"Found {len(subjects)} subjects")
    print("=" * 70)

    # Track channels per subject
    subject_channels = {}
    all_channels = set()
    channel_counts = defaultdict(int)

    for subject_dir in subjects:
        subject_id = subject_dir.name

        if subject_id in EXCLUDE_SUBJECTS:
            print(f"\n{subject_id}: [EXCLUDED - known problematic montage]")
            continue

        print(f"\n{subject_id}:")

        # Get all EDF files for this subject
        edf_files = list(subject_dir.glob("*.edf"))

        if not edf_files:
            print(f"  No EDF files found")
            continue

        # Get channels from first file as reference
        subject_chans = None
        file_channel_sets = []

        for edf_file in edf_files:
            channels = get_edf_channels(edf_file)
            if channels:
                file_channel_sets.append((edf_file.name, channels))
                if subject_chans is None:
                    subject_chans = channels.copy()
                else:
                    # Track if channels differ within subject
                    if channels != subject_chans:
                        diff = subject_chans.symmetric_difference(channels)
                        print(f"  Channel variation in {edf_file.name}: {diff}")

        if subject_chans:
            # Use intersection of all files for this subject
            consistent_channels = subject_chans.copy()
            for _, chans in file_channel_sets:
                consistent_channels &= chans

            subject_channels[subject_id] = consistent_channels
            all_channels |= consistent_channels

            for ch in consistent_channels:
                channel_counts[ch] += 1

            print(f"  Files: {len(edf_files)}, Channels: {len(consistent_channels)}")
            print(f"  Channels: {sorted(consistent_channels)}")

    print("\n" + "=" * 70)
    print("CHANNEL ANALYSIS")
    print("=" * 70)

    # Find channels present in ALL subjects
    n_subjects = len(subject_channels)
    common_channels = set()

    for ch, count in sorted(channel_counts.items(), key=lambda x: -x[1]):
        print(f"  {ch}: {count}/{n_subjects} subjects")
        if count == n_subjects:
            common_channels.add(ch)

    print("\n" + "=" * 70)
    print("COMMON CHANNELS (present in all subjects)")
    print("=" * 70)

    # Sort by standard 10-20 ordering
    sorted_common = sorted(common_channels)
    print(f"\nFound {len(common_channels)} common channels:")
    for ch in sorted_common:
        print(f"  - {ch}")

    # Output as Python constant
    print("\n" + "=" * 70)
    print("PYTHON CONSTANT")
    print("=" * 70)
    print("\nCOMMON_CHANNELS = [")
    for ch in sorted_common:
        print(f'    "{ch}",')
    print("]")

    return sorted_common


if __name__ == '__main__':
    common = discover_common_channels()

    if common:
        print(f"\n\nSUMMARY: {len(common)} channels common to all subjects")
