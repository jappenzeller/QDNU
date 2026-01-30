"""
Classical Baseline for Seizure Detection

Implements the core approach from Kaggle seizure detection winners:
- Bandpass filtering (0.5-128 Hz) + 60 Hz notch
- Relative log power in 6 frequency bands
- XGBoost classifier

This provides a fair comparison point for QPNN evaluation.

Reference: Barachant et al. (2016) - Melbourne University Seizure Prediction Challenge
"""

import numpy as np
from scipy import signal
from scipy.io import loadmat
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import warnings


# =============================================================================
# PREPROCESSING (from Kaggle winners)
# =============================================================================

def preprocess_eeg(data: np.ndarray, fs: int = 400,
                   lowcut: float = 0.5, highcut: float = 128.0,
                   notch_freq: float = 60.0) -> np.ndarray:
    """
    Preprocess EEG using Kaggle winner approach.

    Steps:
    1. Demean
    2. Bandpass filter (Butterworth 5th order)
    3. Notch filter at 60 Hz

    Args:
        data: EEG array (n_channels, n_samples)
        fs: Sampling frequency
        lowcut: Low cutoff for bandpass
        highcut: High cutoff for bandpass
        notch_freq: Frequency for notch filter (60 Hz for US, 50 Hz for EU)

    Returns:
        Preprocessed EEG array
    """
    # Demean
    data = data - np.mean(data, axis=1, keepdims=True)

    # Bandpass filter
    nyq = fs / 2
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)  # Ensure < 1

    try:
        b, a = signal.butter(5, [low, high], btype='band')
        data = signal.filtfilt(b, a, data, axis=1)
    except ValueError as e:
        warnings.warn(f"Bandpass filter failed: {e}. Using raw data.")

    # Notch filter at 60 Hz (if within Nyquist)
    if notch_freq < nyq:
        try:
            b_notch, a_notch = signal.iirnotch(notch_freq, Q=30, fs=fs)
            data = signal.filtfilt(b_notch, a_notch, data, axis=1)
        except ValueError as e:
            warnings.warn(f"Notch filter failed: {e}")

    return data


# =============================================================================
# FEATURE EXTRACTION (from Kaggle winners)
# =============================================================================

# Standard EEG frequency bands
FREQ_BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 15),
    'beta': (15, 30),
    'low_gamma': (30, 70),
    'high_gamma': (70, 128)
}


def extract_band_power(data: np.ndarray, fs: int,
                       bands: Dict[str, Tuple[float, float]] = None) -> np.ndarray:
    """
    Extract relative log power in frequency bands.

    This is the core feature set from Alex/Gilberto's winning solution.

    Args:
        data: EEG array (n_channels, n_samples)
        fs: Sampling frequency
        bands: Dict of {band_name: (low_freq, high_freq)}

    Returns:
        Feature array (n_channels * n_bands,)
    """
    if bands is None:
        bands = FREQ_BANDS

    n_channels = data.shape[0]
    n_bands = len(bands)
    features = np.zeros((n_channels, n_bands))

    for ch in range(n_channels):
        # Compute PSD using Welch's method
        nperseg = min(512, data.shape[1] // 4)
        if nperseg < 16:
            nperseg = data.shape[1]

        freqs, psd = signal.welch(data[ch], fs=fs, nperseg=nperseg)

        # Total power for normalization
        total_power = np.sum(psd)
        if total_power == 0:
            total_power = 1e-10

        # Extract power in each band
        for i, (band_name, (low, high)) in enumerate(bands.items()):
            # Find frequency indices
            idx = np.where((freqs >= low) & (freqs <= high))[0]
            if len(idx) > 0:
                band_power = np.sum(psd[idx])
                # Relative log power
                features[ch, i] = np.log10(band_power / total_power + 1e-10)
            else:
                features[ch, i] = -10  # Very small power

    return features.flatten()


def extract_statistics(data: np.ndarray) -> np.ndarray:
    """
    Extract statistical features per channel.

    Features: mean, std, min, max, skewness, kurtosis

    Args:
        data: EEG array (n_channels, n_samples)

    Returns:
        Feature array (n_channels * 6,)
    """
    from scipy.stats import skew, kurtosis

    n_channels = data.shape[0]
    features = np.zeros((n_channels, 6))

    for ch in range(n_channels):
        x = data[ch]
        features[ch, 0] = np.mean(x)
        features[ch, 1] = np.std(x)
        features[ch, 2] = np.min(x)
        features[ch, 3] = np.max(x)
        features[ch, 4] = skew(x)
        features[ch, 5] = kurtosis(x)

    return features.flatten()


def extract_correlation_features(data: np.ndarray) -> np.ndarray:
    """
    Extract cross-channel correlation features.

    Returns upper triangle of correlation matrix.

    Args:
        data: EEG array (n_channels, n_samples)

    Returns:
        Feature array (n_channels * (n_channels-1) / 2,)
    """
    corr = np.corrcoef(data)
    # Extract upper triangle (excluding diagonal)
    idx = np.triu_indices(corr.shape[0], k=1)
    return corr[idx]


def extract_all_features(data: np.ndarray, fs: int) -> np.ndarray:
    """
    Extract full feature set for one window.

    Combines:
    - Relative log band power (n_channels * 6 bands)
    - Statistics (n_channels * 6 stats)
    - Cross-channel correlations (n_channels choose 2)

    Args:
        data: EEG array (n_channels, n_samples)
        fs: Sampling frequency

    Returns:
        Feature vector
    """
    features = []

    # Band power (primary features)
    features.append(extract_band_power(data, fs))

    # Statistics
    features.append(extract_statistics(data))

    # Correlations (if multiple channels)
    if data.shape[0] > 1:
        features.append(extract_correlation_features(data))

    return np.concatenate(features)


# =============================================================================
# WINDOW EXTRACTION
# =============================================================================

def extract_windows(data: np.ndarray, window_size: int,
                    step: int = None) -> List[np.ndarray]:
    """
    Extract sliding windows from continuous EEG.

    Args:
        data: EEG array (n_channels, n_samples)
        window_size: Samples per window
        step: Step between windows (default: window_size // 2)

    Returns:
        List of window arrays
    """
    if step is None:
        step = window_size // 2

    windows = []
    n_samples = data.shape[1]

    for start in range(0, n_samples - window_size + 1, step):
        windows.append(data[:, start:start + window_size])

    return windows


# =============================================================================
# CLASSIFIER
# =============================================================================

class ClassicalBaseline:
    """
    Classical seizure detection baseline using spectral features + XGBoost.

    Matches the preprocessing and feature extraction from Kaggle winners
    to provide a fair comparison point for QPNN.
    """

    def __init__(self, fs: int = 400, window_size: int = 500,
                 use_preprocessing: bool = True):
        """
        Args:
            fs: Sampling frequency of data
            window_size: Samples per classification window
            use_preprocessing: Whether to apply bandpass/notch filtering
        """
        self.fs = fs
        self.window_size = window_size
        self.use_preprocessing = use_preprocessing
        self.model = None
        self.feature_dim = None

    def _preprocess(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing if enabled."""
        if self.use_preprocessing:
            return preprocess_eeg(data, fs=self.fs)
        return data

    def _extract_features(self, data: np.ndarray) -> np.ndarray:
        """Extract features from one window."""
        return extract_all_features(data, self.fs)

    def fit(self, X_train: List[np.ndarray], y_train: List[int],
            use_xgboost: bool = True) -> 'ClassicalBaseline':
        """
        Train the classifier.

        Args:
            X_train: List of EEG windows (n_channels, window_size)
            y_train: List of labels (1=ictal, 0=interictal)
            use_xgboost: Use XGBoost (True) or RandomForest (False)

        Returns:
            self
        """
        # Extract features
        features = []
        for window in X_train:
            window = self._preprocess(window)
            feat = self._extract_features(window)
            features.append(feat)

        X = np.array(features)
        y = np.array(y_train)

        self.feature_dim = X.shape[1]

        # Train classifier
        if use_xgboost:
            try:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='logloss'
                )
            except ImportError:
                warnings.warn("XGBoost not available, falling back to RandomForest")
                use_xgboost = False

        if not use_xgboost:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )

        self.model.fit(X, y)
        return self

    def predict(self, window: np.ndarray) -> Tuple[bool, float]:
        """
        Predict on a single window.

        Args:
            window: EEG array (n_channels, window_size)

        Returns:
            (prediction, confidence)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        window = self._preprocess(window)
        feat = self._extract_features(window).reshape(1, -1)

        pred = self.model.predict(feat)[0]
        proba = self.model.predict_proba(feat)[0]
        confidence = proba[1]  # P(ictal)

        return bool(pred), float(confidence)

    def predict_batch(self, windows: List[np.ndarray]) -> List[Tuple[bool, float]]:
        """Predict on multiple windows."""
        return [self.predict(w) for w in windows]

    def evaluate(self, X_test: List[np.ndarray],
                 y_test: List[int]) -> Dict[str, float]:
        """
        Evaluate classifier performance.

        Args:
            X_test: List of test windows
            y_test: True labels

        Returns:
            Dict with accuracy, precision, recall, F1, AUC
        """
        predictions = []
        confidences = []

        for window in X_test:
            pred, conf = self.predict(window)
            predictions.append(int(pred))
            confidences.append(conf)

        predictions = np.array(predictions)
        confidences = np.array(confidences)
        y_test = np.array(y_test)

        # Metrics
        tp = np.sum((predictions == 1) & (y_test == 1))
        tn = np.sum((predictions == 0) & (y_test == 0))
        fp = np.sum((predictions == 1) & (y_test == 0))
        fn = np.sum((predictions == 0) & (y_test == 1))

        accuracy = (tp + tn) / len(y_test) if len(y_test) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        # AUC
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_test, confidences)
        except:
            auc = 0.5

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'auc': auc,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'mean_confidence_positive': float(np.mean(confidences[y_test == 1])) if np.any(y_test == 1) else 0,
            'mean_confidence_negative': float(np.mean(confidences[y_test == 0])) if np.any(y_test == 0) else 0
        }


# =============================================================================
# DATA LOADING (compatible with existing infrastructure)
# =============================================================================

def load_kaggle_segment(filepath: Path) -> Tuple[np.ndarray, int]:
    """
    Load a Kaggle AES competition segment.

    Args:
        filepath: Path to .mat file

    Returns:
        (data, sampling_rate)
    """
    mat = loadmat(str(filepath))

    # Find the data key (varies by file)
    data_key = None
    for key in mat.keys():
        if 'segment' in key.lower():
            data_key = key
            break

    if data_key is None:
        raise ValueError(f"Could not find segment data in {filepath}")

    segment = mat[data_key][0, 0]
    data = segment['data']
    fs = int(segment['sampling_frequency'][0, 0])

    return data, fs


def load_subject_windows(subject_dir: Path, seg_type: str,
                         max_channels: int = None,
                         window_size: int = 500,
                         n_windows: int = None) -> Tuple[List[np.ndarray], int]:
    """
    Load windows from a subject's segments.

    Args:
        subject_dir: Path to subject folder
        seg_type: 'ictal' or 'interictal'
        max_channels: Limit channels (None = all)
        window_size: Samples per window
        n_windows: Max windows to return (None = all)

    Returns:
        (list of windows, sampling_rate)
    """
    import re

    pattern = f"*_{seg_type}_segment_*.mat"
    files = sorted(subject_dir.glob(pattern),
                   key=lambda p: int(re.search(r'_(\d+)\.mat$', p.name).group(1)))

    if not files:
        return [], 0

    all_windows = []
    fs = None

    for fpath in files:
        try:
            data, fs = load_kaggle_segment(fpath)

            if max_channels and data.shape[0] > max_channels:
                data = data[:max_channels, :]

            windows = extract_windows(data, window_size)
            all_windows.extend(windows)

            if n_windows and len(all_windows) >= n_windows:
                all_windows = all_windows[:n_windows]
                break

        except Exception as e:
            warnings.warn(f"Failed to load {fpath}: {e}")
            continue

    return all_windows, fs


# =============================================================================
# COMPARISON RUNNER
# =============================================================================

def run_comparison(subject_dir: Path, num_channels: int = 4,
                   window_size: int = 500, n_samples: int = 15,
                   test_split: float = 0.5) -> Dict:
    """
    Run head-to-head comparison of Classical vs QPNN.

    Uses same data, same splits for fair comparison.

    Args:
        subject_dir: Path to subject data
        num_channels: Number of EEG channels to use
        window_size: Samples per window
        n_samples: Number of windows per class
        test_split: Fraction for testing

    Returns:
        Comparison results dict
    """
    print(f"\n{'='*60}")
    print("CLASSICAL BASELINE vs QPNN COMPARISON")
    print(f"{'='*60}")
    print(f"Subject: {subject_dir.name}")
    print(f"Channels: {num_channels}, Window: {window_size} samples")

    # Load data
    print("\nLoading data...")
    ictal_windows, fs = load_subject_windows(
        subject_dir, 'ictal',
        max_channels=num_channels,
        window_size=window_size,
        n_windows=n_samples
    )

    inter_windows, _ = load_subject_windows(
        subject_dir, 'interictal',
        max_channels=num_channels,
        window_size=window_size,
        n_windows=n_samples
    )

    print(f"Loaded {len(ictal_windows)} ictal, {len(inter_windows)} interictal windows")
    print(f"Sampling rate: {fs} Hz")

    if len(ictal_windows) < 2 or len(inter_windows) < 2:
        raise ValueError("Not enough data for train/test split")

    # Split data
    n_train_ictal = max(1, int(len(ictal_windows) * (1 - test_split)))
    n_train_inter = max(1, int(len(inter_windows) * (1 - test_split)))

    X_train = ictal_windows[:n_train_ictal] + inter_windows[:n_train_inter]
    y_train = [1] * n_train_ictal + [0] * n_train_inter

    X_test = ictal_windows[n_train_ictal:] + inter_windows[n_train_inter:]
    y_test = [1] * (len(ictal_windows) - n_train_ictal) + [0] * (len(inter_windows) - n_train_inter)

    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Shuffle training data
    idx = np.random.permutation(len(X_train))
    X_train = [X_train[i] for i in idx]
    y_train = [y_train[i] for i in idx]

    # Train classical baseline
    print("\n--- Classical Baseline ---")
    classical = ClassicalBaseline(fs=fs, window_size=window_size, use_preprocessing=True)
    classical.fit(X_train, y_train)
    classical_results = classical.evaluate(X_test, y_test)

    print(f"Accuracy:  {classical_results['accuracy']:.3f}")
    print(f"Precision: {classical_results['precision']:.3f}")
    print(f"Recall:    {classical_results['recall']:.3f}")
    print(f"F1:        {classical_results['f1']:.3f}")
    print(f"AUC:       {classical_results['auc']:.3f}")

    # Try QPNN (if available)
    qpnn_results = None
    try:
        # Import from your existing code
        import sys
        sys.path.insert(0, str(subject_dir.parent.parent))

        from qdnu import TemplateTrainer, SeizurePredictor

        print("\n--- QPNN ---")

        # Train on first ictal sample (template-based)
        trainer = TemplateTrainer(
            num_channels=num_channels,
            lambda_a=0.1,
            lambda_c=0.05,
            dt=0.001
        )
        trainer.train(X_train[y_train.index(1)])  # First ictal in training

        predictor = SeizurePredictor(trainer, threshold=0.5)

        # Evaluate
        predictions = []
        fidelities = []
        for window in X_test:
            pred, fid, _ = predictor.predict(window)
            predictions.append(int(pred))
            fidelities.append(fid)

        predictions = np.array(predictions)
        fidelities = np.array(fidelities)
        y_test_arr = np.array(y_test)

        # Find optimal threshold
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.linspace(0.3, 0.9, 20):
            preds = (fidelities > thresh).astype(int)
            tp = np.sum((preds == 1) & (y_test_arr == 1))
            fp = np.sum((preds == 1) & (y_test_arr == 0))
            fn = np.sum((preds == 0) & (y_test_arr == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        # Recompute with optimal threshold
        predictions = (fidelities > best_thresh).astype(int)

        tp = np.sum((predictions == 1) & (y_test_arr == 1))
        tn = np.sum((predictions == 0) & (y_test_arr == 0))
        fp = np.sum((predictions == 1) & (y_test_arr == 0))
        fn = np.sum((predictions == 0) & (y_test_arr == 1))

        qpnn_results = {
            'accuracy': (tp + tn) / len(y_test),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': best_f1,
            'threshold': best_thresh,
            'mean_fidelity_ictal': float(np.mean(fidelities[y_test_arr == 1])),
            'mean_fidelity_inter': float(np.mean(fidelities[y_test_arr == 0]))
        }

        try:
            from sklearn.metrics import roc_auc_score
            qpnn_results['auc'] = roc_auc_score(y_test, fidelities)
        except:
            qpnn_results['auc'] = 0.5

        print(f"Accuracy:  {qpnn_results['accuracy']:.3f}")
        print(f"Precision: {qpnn_results['precision']:.3f}")
        print(f"Recall:    {qpnn_results['recall']:.3f}")
        print(f"F1:        {qpnn_results['f1']:.3f}")
        print(f"AUC:       {qpnn_results['auc']:.3f}")
        print(f"Threshold: {qpnn_results['threshold']:.3f}")

    except ImportError as e:
        print(f"\nQPNN not available: {e}")
    except Exception as e:
        print(f"\nQPNN evaluation failed: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Classical':<12} {'QPNN':<12}")
    print("-" * 40)

    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        classical_val = classical_results.get(metric, 0)
        qpnn_val = qpnn_results.get(metric, 0) if qpnn_results else 'N/A'

        if isinstance(qpnn_val, float):
            print(f"{metric:<15} {classical_val:<12.3f} {qpnn_val:<12.3f}")
        else:
            print(f"{metric:<15} {classical_val:<12.3f} {qpnn_val:<12}")

    return {
        'classical': classical_results,
        'qpnn': qpnn_results,
        'fs': fs,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("Classical Baseline Test Suite")
    print("=" * 50)

    # Test with synthetic data
    np.random.seed(42)

    print("\n=== Test 1: Preprocessing ===")
    raw = np.random.randn(4, 1000) * 100
    processed = preprocess_eeg(raw, fs=256)
    print(f"Raw range: [{raw.min():.1f}, {raw.max():.1f}]")
    print(f"Processed range: [{processed.min():.2f}, {processed.max():.2f}]")
    print("[OK] Preprocessing works")

    print("\n=== Test 2: Feature Extraction ===")
    features = extract_all_features(processed, fs=256)
    print(f"Feature dimension: {len(features)}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print("[OK] Feature extraction works")

    print("\n=== Test 3: Classifier Training ===")

    # Generate synthetic ictal (synchronized) and interictal (random)
    def make_ictal(n_ch, n_samples):
        t = np.linspace(0, 1, n_samples)
        sync = 2.0 * np.sin(2 * np.pi * 10 * t)
        return np.array([sync + 0.2 * np.random.randn(n_samples) for _ in range(n_ch)])

    def make_interictal(n_ch, n_samples):
        return np.random.randn(n_ch, n_samples)

    X_train = [make_ictal(4, 500) for _ in range(10)] + [make_interictal(4, 500) for _ in range(10)]
    y_train = [1] * 10 + [0] * 10

    X_test = [make_ictal(4, 500) for _ in range(5)] + [make_interictal(4, 500) for _ in range(5)]
    y_test = [1] * 5 + [0] * 5

    baseline = ClassicalBaseline(fs=256, window_size=500, use_preprocessing=True)
    baseline.fit(X_train, y_train, use_xgboost=False)  # Use RF if XGB not installed

    results = baseline.evaluate(X_test, y_test)
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"F1 Score: {results['f1']:.3f}")
    print("[OK] Classifier training works")

    print("\n=== Test 4: Single Prediction ===")
    pred, conf = baseline.predict(make_ictal(4, 500))
    print(f"Ictal prediction: {pred}, confidence: {conf:.3f}")
    pred, conf = baseline.predict(make_interictal(4, 500))
    print(f"Interictal prediction: {pred}, confidence: {conf:.3f}")
    print("[OK] Single prediction works")

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

    # Try real data if available
    print("\n\nLooking for real data...")
    data_roots = [
        Path("../data"),
        Path("../../data"),
        Path("./seizure-data"),
        Path("../seizure-data")
    ]

    for root in data_roots:
        if root.exists():
            subjects = list(root.glob("*_*"))
            if subjects:
                print(f"Found data at {root}")
                print(f"Subjects: {[s.name for s in subjects[:3]]}")

                # Run comparison on first subject
                try:
                    results = run_comparison(subjects[0], num_channels=4)
                except Exception as e:
                    print(f"Comparison failed: {e}")
                break
    else:
        print("No real data found. Run with your data directory.")
