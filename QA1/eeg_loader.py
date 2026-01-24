"""
Real EEG Data Loader for QDNU

Loads EEG data from the Kaggle seizure detection dataset (.mat files).
Adapted from H:/Data/PythonDNU/EEG/loader.py

Data format:
- Ictal: during seizure (used as proxy for pre-ictal detection)
- Interictal: between seizures (normal baseline)

Reference: Kaggle American Epilepsy Society Seizure Prediction Challenge
"""

import numpy as np
import re
from pathlib import Path

# Default data path (can be overridden)
# Uses local Data/ junction which points to the Kaggle dataset
DEFAULT_DATA_ROOT = Path(__file__).parent / "Data"


def load_segment(mat_path):
    """
    Load a single EEG segment from a .mat file.

    Args:
        mat_path: Path to .mat file

    Returns:
        np.ndarray: shape (n_channels, n_samples)
    """
    import scipy.io as sio

    mat = sio.loadmat(str(mat_path))
    # Find the primary variable key (skip MATLAB metadata)
    var_keys = [k for k in mat.keys() if not k.startswith('__')]
    arr = mat[var_keys[0]]

    # Unwrap singleton arrays
    while isinstance(arr, np.ndarray) and arr.size == 1:
        arr = arr.squeeze()

    # Handle structured array with 'data' field
    if hasattr(arr, 'dtype') and arr.dtype.names:
        return arr['data']

    if isinstance(arr, np.ndarray):
        return arr

    raise ValueError(f"Cannot parse data in {mat_path}")


def sorted_paths(subject_dir, seg_type):
    """
    Get sorted list of segment files for a given type.

    Args:
        subject_dir: Path to subject directory
        seg_type: 'ictal' or 'interictal'

    Returns:
        list: Sorted list of Path objects
    """
    pattern = f"*_{seg_type}_segment_*.mat"

    def idx(p):
        m = re.search(rf"_{seg_type}_segment_(\d+)\.mat$", p.name)
        return int(m.group(1)) if m else -1

    return sorted(Path(subject_dir).glob(pattern), key=idx)


def load_subject_data(subject, root_dir=None, max_channels=None):
    """
    Load all EEG data for a subject.

    Args:
        subject: Subject ID (e.g., 'Patient_1', 'Dog_1')
        root_dir: Path to data root (default: DEFAULT_DATA_ROOT)
        max_channels: Limit number of channels (None = all)

    Returns:
        tuple: (ictal_data, interictal_data)
            Each is np.ndarray of shape (n_channels, n_samples)
    """
    if root_dir is None:
        root_dir = DEFAULT_DATA_ROOT

    subj_path = Path(root_dir) / subject

    if not subj_path.exists():
        raise FileNotFoundError(f"Subject directory not found: {subj_path}")

    def concat_segments(seg_type):
        files = sorted_paths(subj_path, seg_type)
        if not files:
            return np.empty((0, 0))
        arrays = [load_segment(p) for p in files]
        data = np.concatenate(arrays, axis=1)

        # Limit channels if requested
        if max_channels is not None and data.shape[0] > max_channels:
            data = data[:max_channels, :]

        return data

    ictal = concat_segments('ictal')
    interictal = concat_segments('interictal')

    return ictal, interictal


def get_eeg_windows(data, window_size, step=None, normalize=True):
    """
    Extract sliding windows from EEG data.

    Args:
        data: np.ndarray shape (n_channels, n_samples)
        window_size: Number of samples per window
        step: Step size (default: window_size // 2)
        normalize: If True, z-score normalize each window

    Yields:
        np.ndarray: Windows of shape (n_channels, window_size)
    """
    if step is None:
        step = window_size // 2

    n_samples = data.shape[1]

    for start in range(0, n_samples - window_size + 1, step):
        window = data[:, start:start + window_size].copy()

        if normalize:
            # Z-score per channel
            for ch in range(window.shape[0]):
                mean = np.mean(window[ch])
                std = np.std(window[ch])
                if std > 0:
                    window[ch] = (window[ch] - mean) / std

        yield window


def load_for_qdnu(subject, num_channels=4, window_size=500,
                  n_ictal=10, n_interictal=10, root_dir=None):
    """
    Load EEG data formatted for QDNU pipeline.

    Args:
        subject: Subject ID
        num_channels: Number of channels to use
        window_size: Samples per segment
        n_ictal: Number of ictal windows to extract
        n_interictal: Number of interictal windows
        root_dir: Data root path

    Returns:
        tuple: (ictal_windows, interictal_windows)
            Each is list of np.ndarray (num_channels, window_size)
    """
    ictal, interictal = load_subject_data(
        subject,
        root_dir=root_dir,
        max_channels=num_channels
    )

    print(f"Loaded {subject}:")
    print(f"  Ictal: {ictal.shape}")
    print(f"  Interictal: {interictal.shape}")

    # Extract windows
    ictal_windows = list(get_eeg_windows(
        ictal, window_size, step=window_size
    ))[:n_ictal]

    interictal_windows = list(get_eeg_windows(
        interictal, window_size, step=window_size
    ))[:n_interictal]

    print(f"  Extracted: {len(ictal_windows)} ictal, {len(interictal_windows)} interictal windows")

    return ictal_windows, interictal_windows


def list_available_subjects(root_dir=None):
    """
    List all available subjects in the data directory.

    Args:
        root_dir: Data root path

    Returns:
        list: Subject directory names
    """
    if root_dir is None:
        root_dir = DEFAULT_DATA_ROOT

    root = Path(root_dir)
    if not root.exists():
        return []

    subjects = []
    for p in root.iterdir():
        if p.is_dir() and (p.name.startswith('Patient_') or p.name.startswith('Dog_')):
            subjects.append(p.name)

    return sorted(subjects)


# === Test cases ===

if __name__ == "__main__":
    print("=" * 50)
    print("EEG Loader Test Suite")
    print("=" * 50)

    # Test 1: List subjects
    print("\n=== Test 1: Available Subjects ===")
    subjects = list_available_subjects()
    print(f"Found {len(subjects)} subjects: {subjects}")

    if not subjects:
        print("No data found. Skipping remaining tests.")
        exit(0)

    # Test 2: Load subject data
    print("\n=== Test 2: Load Subject Data ===")
    subject = subjects[0]
    ictal, interictal = load_subject_data(subject, max_channels=4)
    print(f"{subject} (4 channels):")
    print(f"  Ictal: {ictal.shape}")
    print(f"  Interictal: {interictal.shape}")

    # Test 3: Extract windows
    print("\n=== Test 3: Extract Windows ===")
    windows = list(get_eeg_windows(ictal, window_size=500, step=250))[:5]
    print(f"Extracted {len(windows)} windows from ictal data")
    for i, w in enumerate(windows):
        print(f"  Window {i}: shape={w.shape}, mean={w.mean():.4f}, std={w.std():.4f}")

    # Test 4: Load for QDNU
    print("\n=== Test 4: Load for QDNU ===")
    ictal_wins, inter_wins = load_for_qdnu(
        subject,
        num_channels=4,
        window_size=500,
        n_ictal=5,
        n_interictal=5
    )
    print(f"Ready for QDNU: {len(ictal_wins)} ictal, {len(inter_wins)} interictal")

    # Test 5: Integration with QDNU
    print("\n=== Test 5: QDNU Integration ===")
    try:
        from template_trainer import TemplateTrainer
        from seizure_predictor import SeizurePredictor

        # Train on ictal (seizure-like) data
        trainer = TemplateTrainer(num_channels=4, lambda_a=0.1, lambda_c=0.05)
        trainer.train(ictal_wins[0])

        # Create predictor
        predictor = SeizurePredictor(trainer, threshold=0.5)

        # Test on both types
        _, fid_ictal, _ = predictor.predict(ictal_wins[1])
        _, fid_inter, _ = predictor.predict(inter_wins[0])

        print(f"Ictal fidelity: {fid_ictal:.4f}")
        print(f"Interictal fidelity: {fid_inter:.4f}")
        print(f"Separation: {fid_ictal - fid_inter:.4f}")

        print("[OK] QDNU integration successful")

    except ImportError as e:
        print(f"[SKIP] QDNU modules not available: {e}")

    print("\n" + "=" * 50)
    print("All tests complete!")
    print("=" * 50)
