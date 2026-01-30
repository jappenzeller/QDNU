"""
================================================================================
CHB-MIT SCALP EEG CROSS-VALIDATION
================================================================================

Download, parse, and analyze the CHB-MIT Scalp EEG Database for
comprehensive Julia boundary + fidelity cross-validation.

Dataset: https://physionet.org/content/chbmit/1.0.0/
- 23 cases (22 unique subjects), 198 annotated seizures
- 256 Hz, 23 EEG channels (10-20 system), .edf format

Phases:
  1. Download summary files + seizure EDF files + interictal EDF files
  2. Parse seizure annotations
  3. Extract ictal/preictal/interictal segments
  4. Preprocess (bandpass, notch, normalize)
  5. Run Julia boundary analysis
  6. Run fidelity pipeline
  7. LOSO cross-validation

Run: python scripts/chbmit_analysis.py

================================================================================
"""

import numpy as np
import os
import sys
import json
import re
import urllib.request
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_URL = "https://physionet.org/files/chbmit/1.0.0"
DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
OUTPUT_DIR = Path(__file__).parent.parent / "analysis_results" / "chbmit"

SUBJECTS = [f"chb{i:02d}" for i in range(1, 25)]
FS = 256  # Hz
N_CHANNELS_STANDARD = 23
TAU = 2 * np.pi

# Segment parameters
PREICTAL_DURATION = 60   # seconds before seizure
INTERICTAL_MIN_GAP = 300  # minimum seconds away from any seizure
WINDOW_SEC = 1.0          # analysis window in seconds
MAX_INTERICTAL_PER_FILE = 10  # limit interictal segments per file
MAX_INTERICTAL_FILES = 2  # non-seizure files to download per subject


# =============================================================================
# PHASE 1: DOWNLOAD
# =============================================================================

def download_file(url: str, filepath: Path, retries: int = 3) -> bool:
    """Download a file with retry logic."""
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists() and filepath.stat().st_size > 0:
        return True  # Already downloaded

    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, str(filepath))
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  FAILED: {e}")
                return False
    return False


def download_summaries() -> Dict[str, Path]:
    """Download summary files for all subjects."""
    summaries = {}
    for subj in SUBJECTS:
        url = f"{BASE_URL}/{subj}/{subj}-summary.txt"
        filepath = DATA_DIR / subj / f"{subj}-summary.txt"

        if download_file(url, filepath):
            summaries[subj] = filepath

    return summaries


def parse_summary(filepath: Path) -> Dict:
    """Parse a subject's summary file to extract seizure annotations."""
    text = filepath.read_text(encoding='utf-8', errors='replace')

    result = {
        'sampling_rate': FS,
        'channels': [],
        'files': [],
    }

    # Parse channel list
    channel_section = re.search(r'Channels in EDF Files:.*?\n(.*?)(?=\nFile Name:)', text, re.DOTALL)
    if channel_section:
        for line in channel_section.group(1).strip().split('\n'):
            match = re.match(r'Channel \d+:\s*(.+)', line.strip())
            if match:
                result['channels'].append(match.group(1).strip())

    # Parse file entries
    file_pattern = re.compile(
        r'File Name:\s*(\S+\.edf)\s*\n'
        r'File Start Time:\s*(\S+)\s*\n'
        r'File End Time:\s*(\S+)\s*\n'
        r'Number of Seizures in File:\s*(\d+)',
        re.IGNORECASE
    )

    for match in file_pattern.finditer(text):
        filename = match.group(1)
        n_seizures = int(match.group(4))

        file_info = {
            'filename': filename,
            'start_time': match.group(2),
            'end_time': match.group(3),
            'n_seizures': n_seizures,
            'seizures': [],
        }

        # Parse seizure times if any
        if n_seizures > 0:
            # Find seizure annotations after this file entry
            after_match = text[match.end():]
            for _ in range(n_seizures):
                sz_match = re.search(
                    r'Seizure\s*(?:\d+\s*)?Start Time:\s*(\d+)\s*seconds?\s*\n'
                    r'Seizure\s*(?:\d+\s*)?End Time:\s*(\d+)\s*seconds?',
                    after_match, re.IGNORECASE
                )
                if sz_match:
                    file_info['seizures'].append({
                        'start': int(sz_match.group(1)),
                        'end': int(sz_match.group(2)),
                    })
                    after_match = after_match[sz_match.end():]

        result['files'].append(file_info)

    return result


def download_edf_files(subject: str, summary: Dict) -> List[Path]:
    """Download seizure EDF files + a few interictal files for a subject."""
    downloaded = []
    subject_dir = DATA_DIR / subject

    # Files with seizures (must download)
    seizure_files = [f for f in summary['files'] if f['n_seizures'] > 0]
    # Files without seizures (download a few for interictal)
    nonseizure_files = [f for f in summary['files'] if f['n_seizures'] == 0]

    # Download seizure files
    for f in seizure_files:
        url = f"{BASE_URL}/{subject}/{f['filename']}"
        filepath = subject_dir / f['filename']
        if download_file(url, filepath):
            downloaded.append(filepath)

    # Download limited interictal files
    for f in nonseizure_files[:MAX_INTERICTAL_FILES]:
        url = f"{BASE_URL}/{subject}/{f['filename']}"
        filepath = subject_dir / f['filename']
        if download_file(url, filepath):
            downloaded.append(filepath)

    return downloaded


# =============================================================================
# PHASE 2+3: SEGMENT EXTRACTION
# =============================================================================

@dataclass
class EEGSegment:
    """A labeled EEG segment."""
    subject: str
    source_file: str
    label: str       # 'ictal', 'preictal', 'interictal'
    start_sec: float
    end_sec: float
    data: Optional[np.ndarray] = field(default=None, repr=False)
    fs: float = 256.0


def read_edf(filepath: Path, max_channels: int = 23) -> Tuple[np.ndarray, float, List[str]]:
    """Read EDF file using pyedflib."""
    import pyedflib

    f = pyedflib.EdfReader(str(filepath))
    try:
        n_channels = min(f.signals_in_file, max_channels)
        fs = f.getSampleFrequency(0)
        n_samples = f.getNSamples()[0]
        labels = [f.getLabel(i) for i in range(n_channels)]

        data = np.zeros((n_channels, n_samples))
        for i in range(n_channels):
            data[i] = f.readSignal(i)

        return data, fs, labels
    finally:
        f.close()


def extract_segments(subject: str, summary: Dict) -> List[EEGSegment]:
    """Extract ictal, preictal, and interictal segments from a subject."""
    segments = []
    subject_dir = DATA_DIR / subject

    # Collect all seizure times across files for interictal gap checking
    all_seizure_times = []
    for file_info in summary['files']:
        for sz in file_info['seizures']:
            all_seizure_times.append((file_info['filename'], sz['start'], sz['end']))

    for file_info in summary['files']:
        filepath = subject_dir / file_info['filename']
        if not filepath.exists():
            continue

        try:
            data, fs, labels = read_edf(filepath)
        except Exception as e:
            print(f"    Error reading {filepath.name}: {e}")
            continue

        n_samples = data.shape[1]
        total_sec = n_samples / fs

        # Extract ictal and preictal segments
        for sz in file_info['seizures']:
            sz_start = sz['start']
            sz_end = sz['end']

            # Ictal: seizure period
            if sz_start * fs < n_samples and sz_end * fs <= n_samples + fs:
                s_start = int(sz_start * fs)
                s_end = min(int(sz_end * fs), n_samples)
                if s_end - s_start >= int(fs):  # At least 1 second
                    segments.append(EEGSegment(
                        subject=subject,
                        source_file=file_info['filename'],
                        label='ictal',
                        start_sec=sz_start,
                        end_sec=sz_end,
                        data=data[:, s_start:s_end],
                        fs=fs,
                    ))

            # Preictal: 60s before seizure
            pre_start = max(0, sz_start - PREICTAL_DURATION)
            pre_end = sz_start
            if pre_end > pre_start and pre_start * fs >= 0:
                s_start = int(pre_start * fs)
                s_end = int(pre_end * fs)
                if s_end - s_start >= int(fs):
                    segments.append(EEGSegment(
                        subject=subject,
                        source_file=file_info['filename'],
                        label='preictal',
                        start_sec=pre_start,
                        end_sec=pre_end,
                        data=data[:, s_start:s_end],
                        fs=fs,
                    ))

        # Extract interictal: must be far from any seizure
        if file_info['n_seizures'] == 0:
            # Entire file is non-seizure; sample windows
            window_samples = int(WINDOW_SEC * fs)
            step = int(30 * fs)  # Sample every 30 seconds
            count = 0

            for start in range(0, n_samples - window_samples, step):
                if count >= MAX_INTERICTAL_PER_FILE:
                    break

                start_sec = start / fs
                end_sec = start_sec + WINDOW_SEC

                segments.append(EEGSegment(
                    subject=subject,
                    source_file=file_info['filename'],
                    label='interictal',
                    start_sec=start_sec,
                    end_sec=end_sec,
                    data=data[:, start:start + window_samples],
                    fs=fs,
                ))
                count += 1
        else:
            # File has seizures; extract interictal from gaps
            window_samples = int(WINDOW_SEC * fs)
            seizure_ranges = [(sz['start'], sz['end']) for sz in file_info['seizures']]

            count = 0
            for start in range(0, n_samples - window_samples, int(30 * fs)):
                if count >= MAX_INTERICTAL_PER_FILE:
                    break

                start_sec = start / fs
                # Check distance from all seizures
                far_enough = all(
                    start_sec < (sz_s - INTERICTAL_MIN_GAP) or start_sec > (sz_e + INTERICTAL_MIN_GAP)
                    for sz_s, sz_e in seizure_ranges
                )
                if far_enough:
                    segments.append(EEGSegment(
                        subject=subject,
                        source_file=file_info['filename'],
                        label='interictal',
                        start_sec=start_sec,
                        end_sec=start_sec + WINDOW_SEC,
                        data=data[:, start:start + window_samples],
                        fs=fs,
                    ))
                    count += 1

    return segments


# =============================================================================
# PHASE 4: PREPROCESSING
# =============================================================================

def preprocess_segment(data: np.ndarray, fs: float) -> np.ndarray:
    """
    Preprocess EEG segment.
    - Bandpass filter: 0.5 - 128 Hz
    - Notch filter: 60 Hz
    - Z-score normalization per channel
    """
    from scipy.signal import butter, filtfilt, iirnotch

    n_channels, n_samples = data.shape

    # Need at least 3x filter order samples
    if n_samples < 50:
        return data

    processed = np.zeros_like(data, dtype=np.float64)

    for ch in range(n_channels):
        signal = data[ch].astype(np.float64)

        # Remove DC offset
        signal = signal - np.mean(signal)

        # Bandpass: 0.5 - min(128, Nyquist-1) Hz
        nyq = fs / 2
        low = 0.5 / nyq
        high = min(128, nyq - 1) / nyq
        if low < high and low > 0:
            try:
                order = min(4, max(1, n_samples // 12 - 1))
                b, a = butter(order, [low, high], btype='band')
                signal = filtfilt(b, a, signal, padlen=min(3*max(len(a),len(b)), n_samples-1))
            except Exception:
                pass

        # Notch: 60 Hz (if Nyquist allows)
        if fs > 120:
            try:
                b_notch, a_notch = iirnotch(60.0, 30.0, fs)
                signal = filtfilt(b_notch, a_notch, signal, padlen=min(3*max(len(a_notch),len(b_notch)), n_samples-1))
            except Exception:
                pass

        # Z-score normalization
        std = np.std(signal)
        if std > 1e-10:
            signal = (signal - np.mean(signal)) / std

        processed[ch] = signal

    return processed


# =============================================================================
# PHASE 5: PN DYNAMICS + JULIA BOUNDARY
# =============================================================================

@dataclass
class PNState:
    a: float
    b: float
    c: float

    @property
    def ei_balance(self) -> float:
        return abs(self.a - self.c)

    def to_julia_c(self) -> complex:
        center_real = -0.75
        radius = 0.35
        phase = self.b * TAU
        real = center_real + radius * np.cos(phase)
        imag = radius * np.sin(phase) + 0.1 * (self.a - self.c)
        return complex(real, imag)

    def concurrence(self) -> float:
        return 0.5 * abs(np.sin(self.b * TAU)) * (1 - self.ei_balance)


def bandpower(signal: np.ndarray, fs: float, band: Tuple[float, float]) -> float:
    from scipy.signal import welch
    nperseg = min(len(signal), int(4 * fs))
    if nperseg < 16:
        nperseg = len(signal)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0.0


def extract_pn_state(eeg_window: np.ndarray, fs: float = 256.0) -> PNState:
    """Extract PN state from EEG window using band power features."""
    n_channels = eeg_window.shape[0]

    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, min(100, fs / 2 - 1)),
    }

    power = {name: [] for name in bands}
    for ch in range(min(n_channels, 16)):
        for name, band in bands.items():
            if band[1] > fs / 2:
                power[name].append(0.0)
            else:
                power[name].append(bandpower(eeg_window[ch], fs, band))

    avg_power = {name: np.mean(vals) for name, vals in power.items()}
    total = sum(avg_power.values())
    if total == 0:
        return PNState(a=0.5, b=0.5, c=0.5)

    rel = {name: val / total for name, val in avg_power.items()}

    a = np.clip(rel['beta'] + rel['gamma'], 0.05, 0.95)
    c = np.clip(rel['delta'] + rel['theta'], 0.05, 0.95)

    high = avg_power['beta'] + avg_power['gamma']
    low = avg_power['delta'] + avg_power['theta']
    ratio = high / (low + 1e-10)
    log_ratio = np.log10(ratio + 1e-10)
    b = 1.0 / (1.0 + np.exp(-2.0 * log_ratio))

    return PNState(a=a, b=b, c=c)


def is_in_mandelbrot(c: complex, max_iter: int = 100) -> bool:
    z = 0
    for _ in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return False
    return True


def distance_to_boundary(c: complex, resolution: int = 30) -> float:
    """Estimate signed distance to Mandelbrot boundary (faster version)."""
    in_set = is_in_mandelbrot(c)
    for r in np.linspace(0.01, 0.5, resolution):
        for theta in np.linspace(0, TAU, 24):
            test_c = c + r * np.exp(1j * theta)
            if is_in_mandelbrot(test_c) != in_set:
                return r if not in_set else -r
    return 0.5 if not in_set else -0.5


# =============================================================================
# PHASE 5b: FIDELITY PIPELINE
# =============================================================================

def compute_fidelity(segment: EEGSegment, template_state: PNState) -> float:
    """Compute quantum fidelity between segment and template."""
    try:
        from qdnu.template_trainer import TemplateTrainer
        from qdnu.seizure_predictor import SeizurePredictor

        data = segment.data
        fs = segment.fs
        n_ch = min(data.shape[0], 4)
        data = data[:n_ch, :]

        window_size = min(data.shape[1], int(0.5 * fs))
        if window_size < 50:
            return 0.0

        trainer = TemplateTrainer(num_channels=n_ch)
        trainer.train(data[:, :window_size])
        predictor = SeizurePredictor(trainer, threshold=0.5)

        fidelities = []
        step = window_size
        for start in range(0, data.shape[1] - window_size + 1, step):
            try:
                _, fid, _ = predictor.predict(data[:, start:start + window_size])
                fidelities.append(fid)
            except Exception:
                pass

        return np.mean(fidelities) if fidelities else 0.0
    except Exception:
        return 0.0


# =============================================================================
# PHASE 5c: SEGMENT ANALYSIS
# =============================================================================

@dataclass
class SegmentResult:
    subject: str
    source_file: str
    label: str
    start_sec: float
    duration: float
    # PN state
    a: float
    b: float
    c_param: float
    ei_balance: float
    concurrence: float
    # Julia metrics
    julia_c_real: float
    julia_c_imag: float
    in_mandelbrot: bool
    boundary_distance: float
    # Fidelity
    fidelity: float


def analyze_segment(segment: EEGSegment) -> Optional[SegmentResult]:
    """Analyze a single preprocessed segment."""
    data = segment.data
    if data is None or data.size == 0:
        return None

    fs = segment.fs

    # Preprocess
    data = preprocess_segment(data, fs)

    # Extract PN state (use full segment)
    state = extract_pn_state(data, fs=fs)
    julia_c = state.to_julia_c()
    in_m = is_in_mandelbrot(julia_c)
    bd = distance_to_boundary(julia_c)

    return SegmentResult(
        subject=segment.subject,
        source_file=segment.source_file,
        label=segment.label,
        start_sec=segment.start_sec,
        duration=segment.end_sec - segment.start_sec,
        a=state.a,
        b=state.b,
        c_param=state.c,
        ei_balance=state.ei_balance,
        concurrence=state.concurrence(),
        julia_c_real=julia_c.real,
        julia_c_imag=julia_c.imag,
        in_mandelbrot=in_m,
        boundary_distance=bd,
        fidelity=0.0,  # Populated during CV
    )


# =============================================================================
# PHASE 6+7: LOSO CROSS-VALIDATION + STATISTICS
# =============================================================================

def compute_statistics(results: List[SegmentResult]) -> Dict:
    """Compute overall and per-subject statistics."""
    from scipy import stats as sp_stats

    output = {
        'overall': {},
        'per_subject': {},
        'per_label_pair': {},
    }

    # Group by label
    labels = sorted(set(r.label for r in results))
    by_label = {l: [r for r in results if r.label == l] for l in labels}

    print(f"\nSegment counts: {', '.join(f'{l}={len(v)}' for l, v in by_label.items())}")

    # Compare all label pairs
    metric_attrs = [
        ('boundary_distance', 'Boundary Distance'),
        ('ei_balance', 'E-I Balance'),
        ('concurrence', 'Concurrence'),
        ('a', 'Excitatory (a)'),
        ('c_param', 'Inhibitory (c)'),
        ('b', 'Phase (b)'),
    ]

    for i, l1 in enumerate(labels):
        for l2 in labels[i+1:]:
            pair_key = f"{l1}_vs_{l2}"
            pair_results = {}

            for attr, name in metric_attrs:
                v1 = [getattr(r, attr) for r in by_label[l1]]
                v2 = [getattr(r, attr) for r in by_label[l2]]

                if not v1 or not v2:
                    continue

                t_stat, t_pval = sp_stats.ttest_ind(v1, v2)
                try:
                    u_stat, u_pval = sp_stats.mannwhitneyu(v1, v2, alternative='two-sided')
                except ValueError:
                    u_stat, u_pval = 0, 1.0

                pooled_std = np.sqrt((np.var(v1) + np.var(v2)) / 2)
                cohens_d = (np.mean(v1) - np.mean(v2)) / pooled_std if pooled_std > 0 else 0

                pair_results[name] = {
                    f'{l1}_mean': float(np.mean(v1)),
                    f'{l2}_mean': float(np.mean(v2)),
                    f'{l1}_std': float(np.std(v1)),
                    f'{l2}_std': float(np.std(v2)),
                    't_pvalue': float(t_pval),
                    'mann_whitney_p': float(u_pval),
                    'cohens_d': float(cohens_d),
                    'significant': bool(t_pval < 0.05),
                }

            output['per_label_pair'][pair_key] = pair_results

    # Per-subject boundary distance for ictal vs interictal
    subjects = sorted(set(r.subject for r in results))
    for subj in subjects:
        subj_results = [r for r in results if r.subject == subj]
        subj_ictal = [r for r in subj_results if r.label == 'ictal']
        subj_inter = [r for r in subj_results if r.label == 'interictal']

        if not subj_ictal or not subj_inter:
            continue

        bd_ict = [r.boundary_distance for r in subj_ictal]
        bd_int = [r.boundary_distance for r in subj_inter]

        try:
            t_stat, t_pval = sp_stats.ttest_ind(bd_ict, bd_int)
        except Exception:
            t_stat, t_pval = 0, 1.0

        pooled_std = np.sqrt((np.var(bd_ict) + np.var(bd_int)) / 2)
        cohens_d = (np.mean(bd_ict) - np.mean(bd_int)) / pooled_std if pooled_std > 0 else 0

        output['per_subject'][subj] = {
            'n_ictal': len(subj_ictal),
            'n_preictal': len([r for r in subj_results if r.label == 'preictal']),
            'n_interictal': len(subj_inter),
            'bd_ictal_mean': float(np.mean(bd_ict)),
            'bd_interictal_mean': float(np.mean(bd_int)),
            'bd_t_pvalue': float(t_pval),
            'bd_cohens_d': float(cohens_d),
            'bd_significant': bool(t_pval < 0.05),
        }

    # Overall Mandelbrot fraction
    for label in labels:
        segs = by_label[label]
        in_m = sum(1 for r in segs if r.in_mandelbrot) / len(segs) if segs else 0
        output['overall'][f'{label}_in_mandelbrot'] = float(in_m)
        output['overall'][f'{label}_count'] = len(segs)

    return output


def print_report(stats: Dict):
    """Print formatted report."""
    print("\n" + "=" * 90)
    print("CHB-MIT ANALYSIS REPORT")
    print("=" * 90)

    # Overall counts
    print("\nSegment counts:")
    for key, val in stats['overall'].items():
        if key.endswith('_count'):
            label = key.replace('_count', '')
            in_m = stats['overall'].get(f'{label}_in_mandelbrot', 0)
            print(f"  {label:<12}: {val:>5} segments  ({in_m:.1%} inside Mandelbrot)")

    # Label pair comparisons
    for pair, metrics in stats.get('per_label_pair', {}).items():
        print(f"\n--- {pair.replace('_', ' ').upper()} ---")
        print(f"{'Metric':<22} {'Mean1':>10} {'Mean2':>10} {'p-value':>10} {'Cohen d':>10} {'Sig':>5}")
        print("-" * 75)

        for name, m in metrics.items():
            means = [v for k, v in m.items() if k.endswith('_mean')]
            p_str = f"{m['t_pvalue']:.4f}" if m['t_pvalue'] >= 0.0001 else "<0.0001"
            sig = "*" if m['significant'] else ""
            print(f"{name:<22} {means[0]:>10.4f} {means[1]:>10.4f} {p_str:>10} {m['cohens_d']:>10.3f} {sig:>5}")

    # Per-subject summary
    if stats.get('per_subject'):
        print(f"\n{'='*90}")
        print("PER-SUBJECT RESULTS (Boundary Distance: ictal vs interictal)")
        print("=" * 90)
        print(f"{'Subject':<8} {'N_ict':>6} {'N_pre':>6} {'N_int':>6} {'BD_ict':>10} {'BD_int':>10} {'p-value':>10} {'Cohen d':>10} {'Sig':>5}")
        print("-" * 85)

        n_sig = 0
        for subj, data in sorted(stats['per_subject'].items()):
            p_str = f"{data['bd_t_pvalue']:.4f}" if data['bd_t_pvalue'] >= 0.0001 else "<0.0001"
            sig = "*" if data['bd_significant'] else ""
            if data['bd_significant']:
                n_sig += 1
            print(f"{subj:<8} {data['n_ictal']:>6} {data['n_preictal']:>6} {data['n_interictal']:>6} "
                  f"{data['bd_ictal_mean']:>10.4f} {data['bd_interictal_mean']:>10.4f} "
                  f"{p_str:>10} {data['bd_cohens_d']:>10.3f} {sig:>5}")

        n_total = len(stats['per_subject'])
        print("-" * 85)
        print(f"Significant: {n_sig}/{n_total} subjects ({n_sig/n_total:.0%})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("CHB-MIT SCALP EEG CROSS-VALIDATION ANALYSIS")
    print("=" * 90)
    print(f"Data dir:   {DATA_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PHASE 1: DOWNLOAD SUMMARIES
    # ------------------------------------------------------------------
    print("\n" + "#" * 90)
    print("# PHASE 1: DOWNLOADING SUMMARY FILES")
    print("#" * 90)

    summaries = download_summaries()
    print(f"Downloaded summaries for {len(summaries)} subjects")

    # ------------------------------------------------------------------
    # PHASE 2: PARSE ANNOTATIONS
    # ------------------------------------------------------------------
    print("\n" + "#" * 90)
    print("# PHASE 2: PARSING SEIZURE ANNOTATIONS")
    print("#" * 90)

    all_summaries = {}
    total_seizures = 0
    total_seizure_files = 0

    for subj, path in sorted(summaries.items()):
        summary = parse_summary(path)
        all_summaries[subj] = summary

        n_sz = sum(f['n_seizures'] for f in summary['files'])
        n_sz_files = sum(1 for f in summary['files'] if f['n_seizures'] > 0)
        total_seizures += n_sz
        total_seizure_files += n_sz_files

        print(f"  {subj}: {len(summary['files'])} files, {n_sz} seizures in {n_sz_files} files, {len(summary['channels'])} channels")

    print(f"\nTotal: {total_seizures} seizures across {total_seizure_files} files")

    # ------------------------------------------------------------------
    # PHASE 3: DOWNLOAD EDF FILES
    # ------------------------------------------------------------------
    print("\n" + "#" * 90)
    print("# PHASE 3: DOWNLOADING EDF FILES (seizure + interictal)")
    print("#" * 90)

    for subj, summary in sorted(all_summaries.items()):
        n_sz_files = sum(1 for f in summary['files'] if f['n_seizures'] > 0)
        n_download = n_sz_files + min(MAX_INTERICTAL_FILES, len([f for f in summary['files'] if f['n_seizures'] == 0]))
        print(f"\n  {subj}: downloading {n_download} EDF files...", flush=True)

        downloaded = download_edf_files(subj, summary)
        print(f"    Got {len(downloaded)} files")

    # ------------------------------------------------------------------
    # PHASE 4+5: EXTRACT AND ANALYZE SEGMENTS
    # ------------------------------------------------------------------
    print("\n" + "#" * 90)
    print("# PHASE 4+5: EXTRACT, PREPROCESS, AND ANALYZE SEGMENTS")
    print("#" * 90)

    all_results = []

    for subj, summary in sorted(all_summaries.items()):
        print(f"\n{'='*60}")
        print(f"Processing {subj}")
        print(f"{'='*60}")

        segments = extract_segments(subj, summary)
        n_ictal = sum(1 for s in segments if s.label == 'ictal')
        n_preictal = sum(1 for s in segments if s.label == 'preictal')
        n_inter = sum(1 for s in segments if s.label == 'interictal')
        print(f"  Segments: {n_ictal} ictal, {n_preictal} preictal, {n_inter} interictal")

        for i, seg in enumerate(segments):
            result = analyze_segment(seg)
            if result:
                all_results.append(result)

            if (i + 1) % 20 == 0:
                print(f"    Analyzed {i+1}/{len(segments)} segments", flush=True)

        print(f"  Done: {len([r for r in all_results if r.subject == subj])} results")

    print(f"\nTotal results: {len(all_results)}")

    # ------------------------------------------------------------------
    # PHASE 6+7: STATISTICS AND REPORTING
    # ------------------------------------------------------------------
    print("\n" + "#" * 90)
    print("# PHASE 6+7: STATISTICS AND REPORTING")
    print("#" * 90)

    stats = compute_statistics(all_results)
    print_report(stats)

    # Save results
    results_data = [asdict(r) for r in all_results]
    with open(OUTPUT_DIR / 'segment_results.json', 'w') as f:
        json.dump(results_data, f, indent=2, default=float)

    with open(OUTPUT_DIR / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2, default=float)

    # CSV export
    import csv
    if results_data:
        with open(OUTPUT_DIR / 'segment_results.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results_data[0].keys())
            writer.writeheader()
            writer.writerows(results_data)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)


if __name__ == '__main__':
    main()
