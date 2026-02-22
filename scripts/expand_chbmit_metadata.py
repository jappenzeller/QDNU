#!/usr/bin/env python3
"""
Expand CHB-MIT Processing to All 23 Patients (B-001)

Parses all .edf files and seizure annotation files for patients chb01-chb24
(excluding chb04 which is missing data per standard practice) and generates
comprehensive metadata.

Output: data/chb_mit_full_metadata.json

Reference: B-001 from research prompts
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CHB-MIT data path
DATA_ROOT = Path("H:/Data/PythonDNU/EEG/chbmit")

# Standard exclusions
# - chb12: Major montage changes mid-recording
# - chb24: Re-recording of chb01
EXCLUDE_SUBJECTS = {'chb12', 'chb24'}

# Minimum seizures for LOSO (need at least some positive samples)
MIN_SEIZURES_FOR_LOSO = 3


def parse_summary_file(summary_path: Path) -> Dict:
    """
    Parse CHB-MIT summary file to extract comprehensive metadata.

    Returns dict with:
        - seizures: List of {file, start, end, duration}
        - file_info: Dict of {filename: {duration, channels}}
    """
    result = {
        'seizures': [],
        'file_info': {},
        'raw_content': None
    }

    if not summary_path.exists():
        return result

    try:
        with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            result['raw_content'] = content[:500]  # First 500 chars for debugging
    except Exception as e:
        logger.warning(f"Failed to read {summary_path}: {e}")
        return result

    current_file = None
    current_channels = None
    lines = content.split('\n')

    for i, line in enumerate(lines):
        line = line.strip()

        # Extract file name
        if line.startswith('File Name:'):
            current_file = line.split(':')[1].strip()

        # Extract channel count
        elif line.startswith('Number of Channels:'):
            try:
                current_channels = int(line.split(':')[1].strip())
            except ValueError:
                current_channels = None

        # Store file info
        if current_file and current_file not in result['file_info']:
            result['file_info'][current_file] = {
                'channels': current_channels
            }

        # Extract seizure info
        if 'Seizure' in line and 'Start' in line and current_file:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    time_part = parts[-1].strip()
                    start_sec = int(time_part.replace('seconds', '').strip())

                    # Look for end time in next few lines
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if 'End' in lines[j]:
                            end_parts = lines[j].split(':')
                            if len(end_parts) >= 2:
                                end_time = end_parts[-1].strip()
                                end_sec = int(end_time.replace('seconds', '').strip())

                                result['seizures'].append({
                                    'file': current_file,
                                    'start': start_sec,
                                    'end': end_sec,
                                    'duration': end_sec - start_sec
                                })
                            break
            except (ValueError, IndexError):
                pass

    return result


def get_edf_info(edf_path: Path) -> Optional[Dict]:
    """Get basic info from EDF file without loading full data."""
    try:
        import pyedflib
        with pyedflib.EdfReader(str(edf_path)) as f:
            n_channels = f.signals_in_file
            # Get sample rate from first channel
            sample_rate = f.getSampleFrequency(0) if n_channels > 0 else 256
            # Get duration
            duration = f.file_duration

            return {
                'n_channels': n_channels,
                'sample_rate': sample_rate,
                'duration_sec': duration
            }
    except Exception as e:
        logger.debug(f"Failed to read EDF {edf_path}: {e}")
        return None


def process_subject(subject_dir: Path) -> Dict:
    """Process a single subject and return metadata."""
    subject_id = subject_dir.name

    metadata = {
        'subject_id': subject_id,
        'n_seizures': 0,
        'total_seizure_duration_sec': 0,
        'seizure_durations': [],
        'total_recording_hours': 0,
        'n_edf_files': 0,
        'channel_count': None,
        'sample_rate': 256,
        'seizures': [],
        'edf_files': [],
        'has_summary': False,
        'eligible_for_loso': False,
        'notes': []
    }

    # Check for summary file
    summary_files = list(subject_dir.glob('*-summary.txt'))
    if not summary_files:
        metadata['notes'].append('No summary file found')
        return metadata

    metadata['has_summary'] = True
    summary_data = parse_summary_file(summary_files[0])

    # Process seizures
    metadata['seizures'] = summary_data['seizures']
    metadata['n_seizures'] = len(summary_data['seizures'])
    metadata['seizure_durations'] = [s['duration'] for s in summary_data['seizures']]
    metadata['total_seizure_duration_sec'] = sum(metadata['seizure_durations'])

    # Process EDF files
    edf_files = sorted(subject_dir.glob('*.edf'))
    metadata['n_edf_files'] = len(edf_files)

    total_duration = 0
    channel_counts = set()
    sample_rates = set()

    for edf_path in edf_files:
        edf_info = get_edf_info(edf_path)
        if edf_info:
            metadata['edf_files'].append({
                'filename': edf_path.name,
                'duration_sec': edf_info['duration_sec'],
                'n_channels': edf_info['n_channels'],
                'sample_rate': edf_info['sample_rate']
            })
            total_duration += edf_info['duration_sec']
            channel_counts.add(edf_info['n_channels'])
            sample_rates.add(edf_info['sample_rate'])
        else:
            metadata['edf_files'].append({
                'filename': edf_path.name,
                'error': 'Could not read'
            })

    metadata['total_recording_hours'] = total_duration / 3600

    if len(channel_counts) == 1:
        metadata['channel_count'] = list(channel_counts)[0]
    elif len(channel_counts) > 1:
        metadata['channel_count'] = list(channel_counts)
        metadata['notes'].append(f'Variable channel count: {channel_counts}')

    if len(sample_rates) == 1:
        metadata['sample_rate'] = list(sample_rates)[0]
    elif len(sample_rates) > 1:
        metadata['sample_rate'] = list(sample_rates)
        metadata['notes'].append(f'Variable sample rate: {sample_rates}')

    # Check LOSO eligibility
    metadata['eligible_for_loso'] = metadata['n_seizures'] >= MIN_SEIZURES_FOR_LOSO

    if metadata['n_seizures'] < MIN_SEIZURES_FOR_LOSO:
        metadata['notes'].append(f'Insufficient seizures for LOSO ({metadata["n_seizures"]} < {MIN_SEIZURES_FOR_LOSO})')

    return metadata


def main():
    """Process all subjects and generate metadata JSON."""

    if not DATA_ROOT.exists():
        logger.error(f"Data directory not found: {DATA_ROOT}")
        return

    # Find all subject directories
    all_subjects = sorted([
        d for d in DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith('chb')
    ])

    logger.info(f"Found {len(all_subjects)} subject directories")

    # Process each subject
    all_metadata = {
        'generated': datetime.now().isoformat(),
        'data_root': str(DATA_ROOT),
        'total_subjects': len(all_subjects),
        'excluded_subjects': list(EXCLUDE_SUBJECTS),
        'min_seizures_for_loso': MIN_SEIZURES_FOR_LOSO,
        'subjects': {}
    }

    eligible_count = 0
    total_seizures = 0
    total_hours = 0

    for subject_dir in all_subjects:
        subject_id = subject_dir.name
        logger.info(f"Processing {subject_id}...")

        metadata = process_subject(subject_dir)
        all_metadata['subjects'][subject_id] = metadata

        if subject_id not in EXCLUDE_SUBJECTS:
            if metadata['eligible_for_loso']:
                eligible_count += 1
            total_seizures += metadata['n_seizures']
            total_hours += metadata['total_recording_hours']

    # Summary statistics
    all_metadata['summary'] = {
        'eligible_for_loso': eligible_count,
        'total_seizures': total_seizures,
        'total_recording_hours': round(total_hours, 2),
        'excluded_count': len([s for s in all_metadata['subjects'] if s in EXCLUDE_SUBJECTS])
    }

    # Save metadata
    output_dir = Path(__file__).parent.parent / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'chb_mit_full_metadata.json'

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)

    logger.info(f"\nMetadata saved to: {output_path}")

    # Print summary table
    print("\n" + "=" * 80)
    print("CHB-MIT Full Dataset Metadata Summary")
    print("=" * 80)
    print(f"\n{'Subject':<10} {'Seizures':>10} {'Duration (h)':>14} {'EDF Files':>12} {'Channels':>10} {'LOSO':>8}")
    print("-" * 80)

    for subject_id in sorted(all_metadata['subjects'].keys()):
        meta = all_metadata['subjects'][subject_id]
        excluded = " [EXCL]" if subject_id in EXCLUDE_SUBJECTS else ""
        loso_ok = "Yes" if meta['eligible_for_loso'] else "No"

        ch_str = str(meta['channel_count']) if isinstance(meta['channel_count'], int) else "varies"

        print(f"{subject_id:<10} {meta['n_seizures']:>10} {meta['total_recording_hours']:>14.2f} "
              f"{meta['n_edf_files']:>12} {ch_str:>10} {loso_ok:>8}{excluded}")

    print("-" * 80)
    print(f"\nTotal subjects: {all_metadata['total_subjects']}")
    print(f"Eligible for LOSO: {eligible_count}")
    print(f"Excluded: {len(EXCLUDE_SUBJECTS)} ({', '.join(EXCLUDE_SUBJECTS)})")
    print(f"Total seizures: {total_seizures}")
    print(f"Total recording hours: {total_hours:.2f}")

    # List subjects with insufficient seizures
    insufficient = [
        s for s, m in all_metadata['subjects'].items()
        if s not in EXCLUDE_SUBJECTS and not m['eligible_for_loso']
    ]
    if insufficient:
        print(f"\nSubjects with <{MIN_SEIZURES_FOR_LOSO} seizures (excluded from LOSO):")
        for s in insufficient:
            n = all_metadata['subjects'][s]['n_seizures']
            print(f"  {s}: {n} seizures")

    return all_metadata


if __name__ == '__main__':
    main()
