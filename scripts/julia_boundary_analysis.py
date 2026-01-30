"""
================================================================================
JULIA BOUNDARY CORRELATION ANALYSIS
================================================================================

Cross-validation analysis correlating Julia set boundary metrics with
ictal vs interictal EEG states across 8 patients (Kaggle dataset).

Hypothesis: Ictal (seizure) states show different Julia boundary characteristics
than interictal states, potentially visible as:
- Different distance to Mandelbrot boundary
- Different E-I balance (|a - c|)
- Different concurrence values
- Different Julia well depths

Run: python scripts/julia_boundary_analysis.py

================================================================================
"""

import numpy as np
import scipy.io as sio
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT = Path("H:/Data/PythonDNU/EEG/DataKaggle")
OUTPUT_DIR = Path(__file__).parent.parent / "analysis_results"
TAU = 2 * np.pi

# =============================================================================
# PN DYNAMICS
# =============================================================================

@dataclass
class PNState:
    """PN neuron state from EEG window."""
    a: float  # Excitatory (0-1)
    b: float  # Phase (0-1 as tau fraction)
    c: float  # Inhibitory (0-1)

    @property
    def ei_balance(self) -> float:
        """E-I balance: |a - c|"""
        return abs(self.a - self.c)

    def to_julia_c(self) -> complex:
        """Map to Julia c parameter using boundary-crossing mapping."""
        center_real = -0.75
        radius = 0.35
        phase = self.b * TAU
        real = center_real + radius * np.cos(phase)
        imag = radius * np.sin(phase) + 0.1 * (self.a - self.c)
        return complex(real, imag)

    def concurrence(self) -> float:
        """Approximate concurrence from PN state."""
        return 0.5 * abs(np.sin(self.b * TAU)) * (1 - self.ei_balance)


def bandpower(signal: np.ndarray, fs: float,
              band: Tuple[float, float]) -> float:
    """Compute power in a frequency band using Welch's method."""
    from scipy.signal import welch
    nperseg = min(len(signal), int(4 * fs))
    if nperseg < 16:
        nperseg = len(signal)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    idx = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx], freqs[idx]) if np.any(idx) else 0.0


def extract_pn_state(eeg_window: np.ndarray,
                     fs: float = 5000.0) -> PNState:
    """
    Extract PN state from EEG window using band power features.

    Mapping (physiologically grounded):
      a (excitatory) <- high-freq power: beta (13-30 Hz) + gamma (30-100 Hz)
      c (inhibitory) <- low-freq power:  theta (4-8 Hz) + delta (1-4 Hz)
      b (phase)      <- ratio that sweeps through boundary:
                         log(high/low) mapped to [0, 1]
    """
    n_channels = eeg_window.shape[0]
    n_samples = eeg_window.shape[1] if eeg_window.ndim > 1 else len(eeg_window)

    # Compute band powers per channel, then average
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta':  (13, 30),
        'gamma': (30, min(100, fs / 2 - 1)),  # Cap at Nyquist
    }

    power = {name: [] for name in bands}

    if eeg_window.ndim == 1:
        for name, band in bands.items():
            if band[1] > fs / 2:
                power[name].append(0.0)
            else:
                power[name].append(bandpower(eeg_window, fs, band))
    else:
        for ch in range(min(n_channels, 16)):  # Limit channels for speed
            for name, band in bands.items():
                if band[1] > fs / 2:
                    power[name].append(0.0)
                else:
                    power[name].append(bandpower(eeg_window[ch], fs, band))

    # Average across channels
    avg_power = {name: np.mean(vals) for name, vals in power.items()}

    # Total power for normalization
    total = sum(avg_power.values())
    if total == 0:
        return PNState(a=0.5, b=0.5, c=0.5)

    # Relative powers
    rel = {name: val / total for name, val in avg_power.items()}

    # a (excitatory) = high-frequency fraction (beta + gamma)
    a = np.clip(rel['beta'] + rel['gamma'], 0.05, 0.95)

    # c (inhibitory) = low-frequency fraction (delta + theta)
    c = np.clip(rel['delta'] + rel['theta'], 0.05, 0.95)

    # b (phase) = log ratio of high/low, mapped to [0, 1]
    high = avg_power['beta'] + avg_power['gamma']
    low = avg_power['delta'] + avg_power['theta']
    ratio = high / (low + 1e-10)

    # Map log ratio to [0, 1] using sigmoid
    log_ratio = np.log10(ratio + 1e-10)
    b = 1.0 / (1.0 + np.exp(-2.0 * log_ratio))

    return PNState(a=a, b=b, c=c)


# =============================================================================
# MANDELBROT BOUNDARY
# =============================================================================

def is_in_mandelbrot(c: complex, max_iter: int = 100) -> bool:
    """Check if c is in the Mandelbrot set."""
    z = 0
    for _ in range(max_iter):
        z = z * z + c
        if abs(z) > 2:
            return False
    return True


def distance_to_boundary(c: complex, resolution: int = 50) -> float:
    """
    Estimate signed distance to Mandelbrot boundary.
    Positive = outside, Negative = inside.
    """
    in_set = is_in_mandelbrot(c)

    # Binary search for boundary
    for r in np.linspace(0.01, 0.5, resolution):
        found_transition = False
        for theta in np.linspace(0, TAU, 36):
            test_c = c + r * np.exp(1j * theta)
            if is_in_mandelbrot(test_c) != in_set:
                found_transition = True
                break
        if found_transition:
            return r if not in_set else -r

    return 0.5 if not in_set else -0.5


def compute_julia_max_depth(c: complex, resolution: int = 100, max_iter: int = 256) -> float:
    """Compute maximum escape time (depth) for Julia set."""
    x = np.linspace(-1.5, 1.5, resolution)
    y = np.linspace(-1.5, 1.5, resolution)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    escape_time = np.zeros_like(X, dtype=np.float32)
    mask = np.ones_like(X, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] ** 2 + c
        escaped = np.abs(Z) > 2
        new_escaped = escaped & mask
        escape_time[new_escaped] = i + 1
        mask[escaped] = False
        if not np.any(mask):
            break

    # Interior points get max depth
    escape_time[mask] = max_iter

    return np.max(escape_time) / max_iter


# =============================================================================
# DATA LOADING (Kaggle DataKaggle format)
# =============================================================================

def load_segment(mat_path: Path) -> Tuple[np.ndarray, dict]:
    """Load EEG segment from Kaggle .mat file.

    Kaggle format has flat keys: 'data', 'freq', 'channels', optional 'latency'.
    """
    mat = sio.loadmat(str(mat_path))

    # Check for Kaggle format (flat keys)
    if 'data' in mat:
        data = mat['data']  # (channels, samples)
        freq = mat['freq'].flat[0]
        metadata = {
            'sampling_freq': float(freq),
            'duration': data.shape[1] / freq,
        }
        return data, metadata

    # Fallback: old nested format
    key = [k for k in mat.keys() if not k.startswith('__')][0]
    seg = mat[key][0, 0]
    data = seg['data']
    metadata = {
        'sampling_freq': seg['sampling_frequency'].item(),
        'duration': seg['data_length_sec'].item(),
    }
    return data, metadata


def get_windows(data: np.ndarray, window_samples: int, step: int) -> List[np.ndarray]:
    """Extract sliding windows from EEG data."""
    windows = []
    n_samples = data.shape[1]

    for start in range(0, n_samples - window_samples + 1, step):
        windows.append(data[:, start:start + window_samples])

    return windows


# =============================================================================
# ANALYSIS PIPELINE
# =============================================================================

@dataclass
class SegmentMetrics:
    """Metrics for a single EEG segment."""
    segment_id: str
    label: str  # 'ictal' or 'interictal'
    patient: str

    # PN state (averaged across windows)
    a_mean: float
    a_std: float
    c_mean: float
    c_std: float
    b_mean: float

    # Derived metrics
    ei_balance_mean: float
    ei_balance_std: float
    concurrence_mean: float
    concurrence_std: float

    # Julia boundary metrics
    julia_c_real: float
    julia_c_imag: float
    in_mandelbrot: bool
    boundary_distance: float
    max_depth: float


def analyze_segment(mat_path: Path, label: str, patient: str,
                    max_channels: int = 16) -> SegmentMetrics:
    """Analyze a single EEG segment."""
    # Load data
    data, meta = load_segment(mat_path)
    fs = meta['sampling_freq']

    # Limit channels
    if data.shape[0] > max_channels:
        data = data[:max_channels, :]

    # For short segments (1s), use full segment as single window
    # For longer segments, use 1s sliding windows with 50% overlap
    n_samples = data.shape[1]
    window_samples = min(n_samples, int(fs))  # 1-second windows
    step = max(1, window_samples // 2)

    if n_samples <= window_samples:
        windows = [data]
    else:
        windows = get_windows(data, window_samples, step)

    if not windows:
        raise ValueError(f"No windows extracted from {mat_path}")

    # Extract PN states from each window
    states = [extract_pn_state(w, fs=fs) for w in windows]

    # Aggregate metrics
    a_vals = [s.a for s in states]
    c_vals = [s.c for s in states]
    b_vals = [s.b for s in states]
    ei_vals = [s.ei_balance for s in states]
    conc_vals = [s.concurrence() for s in states]

    # Use mean state for Julia analysis
    mean_state = PNState(
        a=np.mean(a_vals),
        b=np.mean(b_vals),
        c=np.mean(c_vals)
    )

    julia_c = mean_state.to_julia_c()
    in_m = is_in_mandelbrot(julia_c)

    return SegmentMetrics(
        segment_id=mat_path.stem,
        label=label,
        patient=patient,
        a_mean=np.mean(a_vals),
        a_std=np.std(a_vals),
        c_mean=np.mean(c_vals),
        c_std=np.std(c_vals),
        b_mean=np.mean(b_vals),
        ei_balance_mean=np.mean(ei_vals),
        ei_balance_std=np.std(ei_vals),
        concurrence_mean=np.mean(conc_vals),
        concurrence_std=np.std(conc_vals),
        julia_c_real=julia_c.real,
        julia_c_imag=julia_c.imag,
        in_mandelbrot=in_m,
        boundary_distance=distance_to_boundary(julia_c),
        max_depth=compute_julia_max_depth(julia_c),
    )


def run_analysis(patients: List[str] = None,
                 max_segments: int = 30) -> List[SegmentMetrics]:
    """
    Run Julia boundary analysis on ictal vs interictal segments.

    Args:
        patients: List of patient IDs (default: all Patient_1 through Patient_8)
        max_segments: Max segments per class per patient

    Returns:
        List of SegmentMetrics for all segments
    """
    if patients is None:
        patients = [f'Patient_{i}' for i in range(1, 9)]

    all_metrics = []

    for patient in patients:
        patient_dir = DATA_ROOT / patient

        if not patient_dir.exists():
            print(f"Skipping {patient}: directory not found")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing {patient}")
        print(f"{'='*60}")

        # Get segment files (ictal and interictal)
        ictal_files = sorted(patient_dir.glob(f"{patient}_ictal_segment_*.mat"))
        interictal_files = sorted(patient_dir.glob(f"{patient}_interictal_segment_*.mat"))

        # Sample segments if too many
        if max_segments and len(ictal_files) > max_segments:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(ictal_files), max_segments, replace=False)
            ictal_files = [ictal_files[i] for i in sorted(indices)]

        if max_segments and len(interictal_files) > max_segments:
            rng = np.random.RandomState(42)
            indices = rng.choice(len(interictal_files), max_segments, replace=False)
            interictal_files = [interictal_files[i] for i in sorted(indices)]

        print(f"Ictal segments:      {len(ictal_files)}")
        print(f"Interictal segments: {len(interictal_files)}")

        # Analyze ictal
        for i, f in enumerate(ictal_files):
            try:
                print(f"  Ictal {i+1}/{len(ictal_files)}: {f.name[:50]}...", end=" ", flush=True)
                metrics = analyze_segment(f, 'ictal', patient)
                all_metrics.append(metrics)
                print(f"OK (in_M={metrics.in_mandelbrot}, dist={metrics.boundary_distance:.3f})")
            except Exception as e:
                print(f"ERROR: {e}")

        # Analyze interictal
        for i, f in enumerate(interictal_files):
            try:
                print(f"  Interictal {i+1}/{len(interictal_files)}: {f.name[:50]}...", end=" ", flush=True)
                metrics = analyze_segment(f, 'interictal', patient)
                all_metrics.append(metrics)
                print(f"OK (in_M={metrics.in_mandelbrot}, dist={metrics.boundary_distance:.3f})")
            except Exception as e:
                print(f"ERROR: {e}")

    return all_metrics


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_statistics(metrics: List[SegmentMetrics]) -> Dict:
    """Compute statistical comparison between ictal and interictal."""
    from scipy import stats

    ictal = [m for m in metrics if m.label == 'ictal']
    interictal = [m for m in metrics if m.label == 'interictal']

    results = {
        'n_ictal': len(ictal),
        'n_interictal': len(interictal),
        'metrics': {},
        'per_patient': {},
    }

    # Metrics to compare
    metric_names = [
        ('ei_balance_mean', 'E-I Balance'),
        ('concurrence_mean', 'Concurrence'),
        ('boundary_distance', 'Boundary Distance'),
        ('max_depth', 'Max Depth'),
        ('a_mean', 'Excitatory (a)'),
        ('c_mean', 'Inhibitory (c)'),
        ('b_mean', 'Phase (b)'),
    ]

    # Overall statistics
    for attr, name in metric_names:
        ict_vals = [getattr(m, attr) for m in ictal]
        int_vals = [getattr(m, attr) for m in interictal]

        if not ict_vals or not int_vals:
            continue

        # t-test
        t_stat, t_pval = stats.ttest_ind(ict_vals, int_vals)

        # Mann-Whitney U test (non-parametric)
        try:
            u_stat, u_pval = stats.mannwhitneyu(ict_vals, int_vals, alternative='two-sided')
        except ValueError:
            u_stat, u_pval = 0, 1.0

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(ict_vals) + np.var(int_vals)) / 2)
        cohens_d = (np.mean(ict_vals) - np.mean(int_vals)) / pooled_std if pooled_std > 0 else 0

        results['metrics'][name] = {
            'ictal_mean': np.mean(ict_vals),
            'ictal_std': np.std(ict_vals),
            'interictal_mean': np.mean(int_vals),
            'interictal_std': np.std(int_vals),
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'mann_whitney_u': u_stat,
            'mann_whitney_p': u_pval,
            'cohens_d': cohens_d,
            'significant': t_pval < 0.05,
        }

    # In-Mandelbrot comparison
    ict_in_m = sum(1 for m in ictal if m.in_mandelbrot) / len(ictal) if ictal else 0
    int_in_m = sum(1 for m in interictal if m.in_mandelbrot) / len(interictal) if interictal else 0

    results['in_mandelbrot'] = {
        'ictal_fraction': ict_in_m,
        'interictal_fraction': int_in_m,
    }

    # Per-patient statistics
    patients = sorted(set(m.patient for m in metrics))
    for patient in patients:
        p_ictal = [m for m in ictal if m.patient == patient]
        p_inter = [m for m in interictal if m.patient == patient]

        if not p_ictal or not p_inter:
            continue

        p_results = {}
        for attr, name in metric_names:
            ict_vals = [getattr(m, attr) for m in p_ictal]
            int_vals = [getattr(m, attr) for m in p_inter]

            try:
                t_stat, t_pval = stats.ttest_ind(ict_vals, int_vals)
            except Exception:
                t_stat, t_pval = 0, 1.0

            pooled_std = np.sqrt((np.var(ict_vals) + np.var(int_vals)) / 2)
            cohens_d = (np.mean(ict_vals) - np.mean(int_vals)) / pooled_std if pooled_std > 0 else 0

            p_results[name] = {
                'ictal_mean': np.mean(ict_vals),
                'interictal_mean': np.mean(int_vals),
                't_pvalue': t_pval,
                'cohens_d': cohens_d,
                'significant': t_pval < 0.05,
            }

        results['per_patient'][patient] = {
            'n_ictal': len(p_ictal),
            'n_interictal': len(p_inter),
            'metrics': p_results,
        }

    return results


def print_results(stats: Dict):
    """Print statistical results."""
    print("\n" + "=" * 80)
    print("OVERALL STATISTICAL RESULTS")
    print("=" * 80)

    print(f"\nSample sizes: {stats['n_ictal']} ictal, {stats['n_interictal']} interictal")

    print(f"\nIn-Mandelbrot fraction:")
    print(f"  Ictal:      {stats['in_mandelbrot']['ictal_fraction']:.1%}")
    print(f"  Interictal: {stats['in_mandelbrot']['interictal_fraction']:.1%}")

    print("\n" + "-" * 80)
    print(f"{'Metric':<20} {'Ictal':>12} {'Interictal':>12} {'p-value':>10} {'Cohen d':>10} {'Sig':>5}")
    print("-" * 80)

    for name, m in stats['metrics'].items():
        ict_str = f"{m['ictal_mean']:.4f}"
        int_str = f"{m['interictal_mean']:.4f}"
        p_str = f"{m['t_pvalue']:.4f}" if m['t_pvalue'] >= 0.0001 else "<0.0001"
        d_str = f"{m['cohens_d']:.3f}"
        sig = "*" if m['significant'] else ""

        print(f"{name:<20} {ict_str:>12} {int_str:>12} {p_str:>10} {d_str:>10} {sig:>5}")

    print("-" * 80)
    print("* = statistically significant (p < 0.05)")

    # Best discriminator
    if stats['metrics']:
        best_metric = max(stats['metrics'].items(),
                          key=lambda x: abs(x[1]['cohens_d']))
        print(f"\nBest discriminator: {best_metric[0]} (Cohen's d = {best_metric[1]['cohens_d']:.3f})")

    # Per-patient summary
    if stats.get('per_patient'):
        print("\n" + "=" * 80)
        print("PER-PATIENT RESULTS (Boundary Distance)")
        print("=" * 80)
        print(f"{'Patient':<12} {'N_ict':>6} {'N_int':>6} {'Ict_dist':>10} {'Int_dist':>10} {'p-value':>10} {'Cohen d':>10} {'Sig':>5}")
        print("-" * 80)

        for patient, pdata in sorted(stats['per_patient'].items()):
            bd = pdata['metrics'].get('Boundary Distance', {})
            if bd:
                p_str = f"{bd['t_pvalue']:.4f}" if bd['t_pvalue'] >= 0.0001 else "<0.0001"
                print(f"{patient:<12} {pdata['n_ictal']:>6} {pdata['n_interictal']:>6} "
                      f"{bd['ictal_mean']:>10.4f} {bd['interictal_mean']:>10.4f} "
                      f"{p_str:>10} {bd['cohens_d']:>10.3f} {'*' if bd['significant'] else '':>5}")

        print("-" * 80)


def save_results(metrics: List[SegmentMetrics], stats: Dict, output_dir: Path):
    """Save results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw metrics as JSON
    metrics_data = [
        {
            'segment_id': m.segment_id,
            'label': m.label,
            'patient': m.patient,
            'a_mean': m.a_mean,
            'c_mean': m.c_mean,
            'b_mean': m.b_mean,
            'ei_balance': m.ei_balance_mean,
            'concurrence': m.concurrence_mean,
            'julia_c_real': m.julia_c_real,
            'julia_c_imag': m.julia_c_imag,
            'in_mandelbrot': m.in_mandelbrot,
            'boundary_distance': m.boundary_distance,
            'max_depth': m.max_depth,
        }
        for m in metrics
    ]

    with open(output_dir / 'segment_metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2, default=float)

    # Save statistics
    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=2, default=float)

    # Save CSV for easy import
    import csv
    with open(output_dir / 'segment_metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
        writer.writeheader()
        writer.writerows(metrics_data)

    print(f"\nResults saved to {output_dir}/")


# =============================================================================
# FIDELITY PIPELINE CROSS-VALIDATION
# =============================================================================

def run_fidelity_cv(patients: List[str] = None,
                    num_channels: int = 4,
                    n_windows: int = 10,
                    threshold: float = 0.7) -> Dict:
    """
    Run the validated QDNU fidelity pipeline with cross-validation.

    Uses TemplateTrainer + SeizurePredictor (the paper's actual pipeline).
    Adapted for Kaggle DataKaggle format (ictal vs interictal).
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from qdnu.template_trainer import TemplateTrainer
    from qdnu.seizure_predictor import SeizurePredictor

    if patients is None:
        patients = [f'Patient_{i}' for i in range(1, 9)]

    all_results = []

    for patient in patients:
        patient_dir = DATA_ROOT / patient
        if not patient_dir.exists():
            print(f"Skipping {patient}: not found")
            continue

        print(f"\n{'='*60}")
        print(f"Fidelity CV: {patient}")
        print(f"{'='*60}")

        ictal_files = sorted(patient_dir.glob(f"{patient}_ictal_segment_*.mat"))
        interictal_files = sorted(patient_dir.glob(f"{patient}_interictal_segment_*.mat"))

        if not ictal_files or not interictal_files:
            print(f"Skipping {patient}: no ictal or interictal files")
            continue

        # Determine appropriate window size from data
        sample_data, sample_meta = load_segment(ictal_files[0])
        fs = sample_meta['sampling_freq']
        n_samples = sample_data.shape[1]
        n_ch = min(sample_data.shape[0], num_channels)

        # Window size: use full segment or 0.5s, whichever is smaller
        window_size = min(n_samples, int(0.5 * fs))
        if window_size < 100:
            window_size = n_samples

        print(f"  Channels: {sample_data.shape[0]} (using {n_ch}), Fs: {fs:.0f} Hz")
        print(f"  Segment: {n_samples} samples ({n_samples/fs:.2f}s), Window: {window_size} samples")
        print(f"  Ictal: {len(ictal_files)}, Interictal: {len(interictal_files)}")

        # 5-fold cross-validation
        n_folds = min(len(ictal_files), 5)
        fold_results = []

        for fold in range(n_folds):
            print(f"\n  Fold {fold+1}/{n_folds}:", flush=True)

            # Train on one ictal segment
            train_idx = (fold + 1) % len(ictal_files)
            train_data, _ = load_segment(ictal_files[train_idx])
            if train_data.shape[0] > n_ch:
                train_data = train_data[:n_ch, :]
            train_window = train_data[:, :window_size]

            # Train template
            trainer = TemplateTrainer(num_channels=n_ch)
            trainer.train(train_window)
            predictor = SeizurePredictor(trainer, threshold=threshold)

            # Test on held-out ictal
            test_ict_data, _ = load_segment(ictal_files[fold])
            if test_ict_data.shape[0] > n_ch:
                test_ict_data = test_ict_data[:n_ch, :]
            ict_windows = get_windows(test_ict_data, window_size, window_size)[:n_windows]
            if not ict_windows:
                ict_windows = [test_ict_data[:, :window_size]]

            # Test on interictal
            inter_idx = fold % len(interictal_files)
            test_int_data, _ = load_segment(interictal_files[inter_idx])
            if test_int_data.shape[0] > n_ch:
                test_int_data = test_int_data[:n_ch, :]
            int_windows = get_windows(test_int_data, window_size, window_size)[:n_windows]
            if not int_windows:
                int_windows = [test_int_data[:, :window_size]]

            # Predict
            ict_fidelities = []
            for w in ict_windows:
                try:
                    pred, fid, _ = predictor.predict(w)
                    ict_fidelities.append(fid)
                except Exception:
                    pass

            int_fidelities = []
            for w in int_windows:
                try:
                    pred, fid, _ = predictor.predict(w)
                    int_fidelities.append(fid)
                except Exception:
                    pass

            if ict_fidelities and int_fidelities:
                ict_mean = np.mean(ict_fidelities)
                int_mean = np.mean(int_fidelities)
                separation = ict_mean - int_mean

                tp = sum(1 for f in ict_fidelities if f >= threshold)
                fn = len(ict_fidelities) - tp
                tn = sum(1 for f in int_fidelities if f < threshold)
                fp = len(int_fidelities) - tn

                total = tp + tn + fp + fn
                accuracy = (tp + tn) / total if total > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                fold_result = {
                    'fold': fold,
                    'ictal_fidelity': ict_mean,
                    'interictal_fidelity': int_mean,
                    'separation': separation,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                }
                fold_results.append(fold_result)

                print(f"    Ictal fidelity:      {ict_mean:.4f} (+/-{np.std(ict_fidelities):.4f})")
                print(f"    Interictal fidelity:  {int_mean:.4f} (+/-{np.std(int_fidelities):.4f})")
                print(f"    Separation:           {separation:.4f}")
                print(f"    Accuracy:             {accuracy:.1%}")

        if fold_results:
            avg_acc = np.mean([r['accuracy'] for r in fold_results])
            avg_f1 = np.mean([r['f1'] for r in fold_results])
            avg_sep = np.mean([r['separation'] for r in fold_results])

            print(f"\n  {patient} Summary:")
            print(f"    Avg Accuracy:   {avg_acc:.1%}")
            print(f"    Avg F1 Score:   {avg_f1:.3f}")
            print(f"    Avg Separation: {avg_sep:.4f}")

            all_results.append({
                'patient': patient,
                'folds': fold_results,
                'avg_accuracy': avg_acc,
                'avg_f1': avg_f1,
                'avg_separation': avg_sep,
            })

    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("QDNU ANALYSIS: FIDELITY PIPELINE + JULIA BOUNDARY (8 PATIENTS)")
    print("=" * 80)
    print(f"Data root: {DATA_ROOT}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check data exists
    if not DATA_ROOT.exists():
        print(f"\nERROR: Data directory not found: {DATA_ROOT}")
        return

    patients = [f'Patient_{i}' for i in range(1, 9)]

    # === PART 1: Fidelity Pipeline Cross-Validation ===
    print("\n" + "#" * 80)
    print("# PART 1: FIDELITY PIPELINE CROSS-VALIDATION (8 PATIENTS)")
    print("#" * 80)

    try:
        fidelity_results = run_fidelity_cv(
            patients=patients,
            num_channels=4,
            n_windows=10,
            threshold=0.7,
        )

        # Print fidelity summary
        if fidelity_results:
            print("\n" + "=" * 60)
            print("FIDELITY PIPELINE SUMMARY")
            print("=" * 60)
            print(f"{'Patient':<12} {'Accuracy':>10} {'F1':>8} {'Separation':>12}")
            print("-" * 50)
            for r in fidelity_results:
                print(f"{r['patient']:<12} {r['avg_accuracy']:>10.1%} {r['avg_f1']:>8.3f} {r['avg_separation']:>12.4f}")

    except Exception as e:
        print(f"Fidelity pipeline error: {e}")
        import traceback
        traceback.print_exc()
        fidelity_results = None

    # === PART 2: Julia Boundary Analysis ===
    print("\n" + "#" * 80)
    print("# PART 2: JULIA BOUNDARY ANALYSIS (8 PATIENTS)")
    print("#" * 80)

    metrics = run_analysis(
        patients=patients,
        max_segments=30,  # 30 per class per patient
    )

    if not metrics:
        print("No metrics collected. Check data paths.")
        return

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(metrics)

    # Print results
    print_results(stats)

    # Save results
    save_results(metrics, stats, OUTPUT_DIR)

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
