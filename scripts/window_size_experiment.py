#!/usr/bin/env python3
"""
================================================================================
PROMPT 015 - Window Size Impact on Classical Baselines
================================================================================

Tests how sub-window size affects classical baseline performance.

Current pipeline: 1.95-second windows (499 samples at 256 Hz)
Kaggle winners: 20-30 second windows

This experiment measures:
1. How much AUC is left on the table from window size alone
2. Whether the SPD manifold argument holds (longer windows = better covariance)

Window sizes tested: [1, 2, 5, 10, 15, 20, 30] seconds

Output:
  - results/window_analysis/window_size_impact.json
  - results/window_analysis/window_size_report.md

================================================================================
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
from scipy.linalg import eigvalsh
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import LedoitWolf
from xgboost import XGBClassifier

# Pyriemann imports
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

sys.path.insert(0, str(Path(__file__).parent.parent))

from sagemaker.train_chbmit import (
    EXCLUDE_SUBJECTS,
    WINDOW_SEC,
    FS,
    EEGSegment,
    preprocess_eeg,
    read_edf_segment,
    extract_segments_for_subject,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT = Path("H:/Data/PythonDNU/EEG/chbmit")
OUTPUT_DIR = Path("results/window_analysis")

# Window sizes to test (in seconds)
WINDOW_SIZES = [1, 2, 5, 10, 15, 20, 30]

# Segment duration (fixed)
SEGMENT_DURATION_SEC = 30

# CH8 channels (same as all prior experiments)
CHANNELS = [
    "FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
    "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"
]

# Priority subjects (same as PROMPT 006/008)
PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']

# XGBoost config
N_BAGS = 5

# Condition number threshold for "ill-conditioned"
CONDITION_THRESHOLD = 1000

# Current baseline
CURRENT_TIER2_AUC = 0.7234
CURRENT_WINDOW_SEC = 1.95


# =============================================================================
# XGBoost MODEL
# =============================================================================

def make_xgb_model(seed: int = 42) -> XGBClassifier:
    """Create XGBoost classifier with paper's hyperparameters."""
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=seed,
        eval_metric='logloss',
        n_jobs=-1,
    )


# =============================================================================
# DATA LOADING
# =============================================================================

def load_segment_data(segment: EEGSegment, channels: List[str]) -> Optional[np.ndarray]:
    """Load and preprocess EEG data for a segment."""
    edf_path = DATA_ROOT / segment.subject / segment.source_file

    if not edf_path.exists():
        return None

    data, fs = read_edf_segment(
        str(edf_path),
        segment.start_sec,
        segment.end_sec,
        target_channels=channels
    )

    if data is None or data.size == 0:
        return None

    # Need at least half of the expected 30s segment
    min_samples = int(SEGMENT_DURATION_SEC * fs * 0.5)
    if data.shape[1] < min_samples:
        return None

    data = preprocess_eeg(data, fs=int(fs))
    return data


def load_subject_segments(subject_id: str) -> List[Tuple[np.ndarray, str]]:
    """Load all segments for a subject."""
    subject_dir = DATA_ROOT / subject_id

    if not subject_dir.exists():
        return []

    segments = extract_segments_for_subject(subject_dir)
    results = []

    for seg in segments:
        data = load_segment_data(seg, CHANNELS)
        if data is not None:
            results.append((data, seg.label))

    return results


# =============================================================================
# TIER 2: CORRELATION EIGENVALUES + XGBOOST
# =============================================================================

def extract_tier2_subwindow_features(subwindow: np.ndarray, fs: int = FS) -> np.ndarray:
    """Extract Tier 2 features from a single sub-window."""
    n_channels = subwindow.shape[0]

    # FFT magnitude (1-40 Hz)
    fft_result = np.fft.rfft(subwindow, axis=1)
    fft_mag = np.abs(fft_result)[:, 1:41]
    fft_mag = np.log10(fft_mag + 1e-6)

    # Frequency-domain correlation eigenvalues
    try:
        C_freq = np.corrcoef(fft_mag)
        if np.isnan(C_freq).any():
            eigenvalues_freq = np.zeros(n_channels)
        else:
            eigenvalues_freq = np.linalg.eigvalsh(C_freq)
    except:
        eigenvalues_freq = np.zeros(n_channels)

    # Time-domain correlation eigenvalues
    try:
        C_time = np.corrcoef(subwindow)
        if np.isnan(C_time).any():
            eigenvalues_time = np.zeros(n_channels)
        else:
            eigenvalues_time = np.linalg.eigvalsh(C_time)
    except:
        eigenvalues_time = np.zeros(n_channels)

    return np.concatenate([fft_mag.flatten(), eigenvalues_freq, eigenvalues_time])


def extract_tier2_features_variable_window(
    segment: np.ndarray,
    window_sec: float,
    fs: int = FS
) -> Dict:
    """
    Extract Tier 2 features with variable sub-window size.

    Returns dict with 'max', 'mean', 'std', 'n_subwindows' keys
    """
    n_channels, n_samples = segment.shape
    samples_per_subwindow = int(window_sec * fs)
    n_subwindows = n_samples // samples_per_subwindow

    if n_subwindows == 0:
        # Use entire segment if window > segment
        features = extract_tier2_subwindow_features(segment, fs)
        return {
            'max': features,
            'mean': features,
            'std': None,
            'n_subwindows': 1
        }

    subwindow_features = []
    for i in range(n_subwindows):
        start = i * samples_per_subwindow
        end = start + samples_per_subwindow
        subwindow = segment[:, start:end]
        features = extract_tier2_subwindow_features(subwindow, fs)
        subwindow_features.append(features)

    subwindow_features = np.array(subwindow_features)

    # Compute all aggregations
    max_features = np.max(subwindow_features, axis=0)
    mean_features = np.mean(subwindow_features, axis=0)
    std_features = np.std(subwindow_features, axis=0) if n_subwindows >= 2 else None

    return {
        'max': max_features,
        'mean': mean_features,
        'std': std_features,
        'n_subwindows': n_subwindows
    }


# =============================================================================
# TIER 3: RIEMANNIAN TANGENT SPACE + LDA
# =============================================================================

def compute_covariance_with_condition(
    segment: np.ndarray,
    window_sec: float,
    fs: int = FS
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    """
    Compute covariance matrices for sub-windows with condition number tracking.

    Returns: (covariances, raw_conditions, regularized_conditions)
    """
    n_channels, n_samples = segment.shape
    samples_per_subwindow = int(window_sec * fs)
    n_subwindows = n_samples // samples_per_subwindow

    if n_subwindows == 0:
        n_subwindows = 1
        samples_per_subwindow = n_samples

    covariances = []
    raw_conditions = []
    reg_conditions = []

    for i in range(n_subwindows):
        start = i * samples_per_subwindow
        end = min(start + samples_per_subwindow, n_samples)
        subwindow = segment[:, start:end]

        if subwindow.shape[1] < n_channels:
            continue

        # Raw covariance
        try:
            cov_raw = np.cov(subwindow)
            eigs_raw = eigvalsh(cov_raw)
            eigs_raw = np.maximum(eigs_raw, 1e-10)  # Avoid division by zero
            cond_raw = eigs_raw[-1] / eigs_raw[0]
            raw_conditions.append(cond_raw)
        except:
            raw_conditions.append(np.inf)

        # Ledoit-Wolf regularized covariance
        try:
            lw = LedoitWolf()
            cov_reg = lw.fit(subwindow.T).covariance_
            eigs_reg = eigvalsh(cov_reg)
            eigs_reg = np.maximum(eigs_reg, 1e-10)
            cond_reg = eigs_reg[-1] / eigs_reg[0]
            reg_conditions.append(cond_reg)

            # Check SPD validity
            if np.all(eigs_reg > 0) and not np.isnan(cov_reg).any():
                covariances.append(cov_reg)
        except:
            reg_conditions.append(np.inf)

    return covariances, raw_conditions, reg_conditions


def extract_riemannian_features_variable_window(
    segment: np.ndarray,
    window_sec: float,
    fs: int = FS
) -> Dict:
    """
    Extract Riemannian tangent space features with variable window size.
    """
    covariances, raw_conds, reg_conds = compute_covariance_with_condition(
        segment, window_sec, fs
    )

    return {
        'covariances': covariances,
        'raw_conditions': raw_conds,
        'reg_conditions': reg_conds,
        'n_subwindows': len(covariances)
    }


# =============================================================================
# LOSO CV
# =============================================================================

def run_tier2_loso_for_window(
    subject_data: Dict[str, List[Tuple[np.ndarray, str]]],
    window_sec: float
) -> Dict:
    """Run Tier 2 LOSO CV for a specific window size."""
    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}

    # Extract features for all subjects
    subject_features = {}
    for subject_id, segments in subject_data.items():
        features_list = {'max': [], 'mean': [], 'std': []}
        labels = []
        n_subwindows_list = []

        for segment_data, label in segments:
            result = extract_tier2_features_variable_window(segment_data, window_sec)

            max_feat = result['max']
            mean_feat = result['mean']
            std_feat = result['std']

            # Skip if features contain NaN/Inf
            if np.isnan(max_feat).any() or np.isinf(max_feat).any():
                continue

            features_list['max'].append(max_feat)
            features_list['mean'].append(mean_feat)
            if std_feat is not None:
                features_list['std'].append(std_feat)
            n_subwindows_list.append(result['n_subwindows'])
            labels.append(label_map.get(label, 0))

        if len(features_list['max']) > 0 and len(set(labels)) == 2:
            subject_features[subject_id] = {
                'max': np.array(features_list['max']),
                'mean': np.array(features_list['mean']),
                'std': np.array(features_list['std']) if len(features_list['std']) == len(labels) else None,
                'labels': np.array(labels),
                'n_subwindows': np.mean(n_subwindows_list) if n_subwindows_list else 1
            }

    if len(subject_features) < 2:
        return {'error': 'Not enough subjects'}

    # Run LOSO for each aggregation
    results = {}
    for agg in ['max', 'mean', 'std']:
        if agg == 'std' and any(sf['std'] is None for sf in subject_features.values()):
            results[agg] = {'overall_auc': None, 'per_subject': {}}
            continue

        per_subject = {}
        all_y_true = []
        all_y_prob = []

        for test_subject in subject_features.keys():
            # Train data
            X_train_list = []
            y_train_list = []
            for train_subject, sf in subject_features.items():
                if train_subject != test_subject:
                    X_train_list.append(sf[agg])
                    y_train_list.append(sf['labels'])

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)
            X_test = subject_features[test_subject][agg]
            y_test = subject_features[test_subject]['labels']

            # Skip if not enough samples
            if len(X_train) < 10 or len(X_test) < 2:
                continue

            # 5-bag ensemble
            y_probs = []
            for bag_idx in range(N_BAGS):
                rng = np.random.RandomState(42 + bag_idx)
                idx = rng.choice(len(X_train), size=len(X_train), replace=True)
                model = make_xgb_model(seed=42 + bag_idx)
                model.fit(X_train[idx], y_train[idx])
                y_prob = model.predict_proba(X_test)[:, 1]
                y_probs.append(y_prob)

            y_prob_mean = np.mean(y_probs, axis=0)

            try:
                auc = roc_auc_score(y_test, y_prob_mean)
            except:
                auc = 0.5

            per_subject[test_subject] = {'auc': auc, 'n_samples': len(y_test)}
            all_y_true.extend(y_test)
            all_y_prob.extend(y_prob_mean)

        try:
            overall_auc = roc_auc_score(all_y_true, all_y_prob)
        except:
            overall_auc = 0.5

        results[agg] = {
            'overall_auc': overall_auc,
            'per_subject': per_subject
        }

    # Average n_subwindows
    avg_subwindows = np.mean([sf['n_subwindows'] for sf in subject_features.values()])

    return {
        'aggregation': results,
        'n_subwindows_per_segment': int(round(avg_subwindows))
    }


def run_tier3_loso_for_window(
    subject_data: Dict[str, List[Tuple[np.ndarray, str]]],
    window_sec: float
) -> Dict:
    """Run Tier 3 Riemannian LOSO CV for a specific window size."""
    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}

    # Extract covariances for all subjects
    subject_covs = {}
    all_raw_conditions = []
    all_reg_conditions = []

    for subject_id, segments in subject_data.items():
        covs_list = []
        labels = []

        for segment_data, label in segments:
            result = extract_riemannian_features_variable_window(segment_data, window_sec)

            all_raw_conditions.extend(result['raw_conditions'])
            all_reg_conditions.extend(result['reg_conditions'])

            # Use mean covariance across sub-windows for this segment
            if result['covariances']:
                if len(result['covariances']) == 1:
                    covs_list.append(result['covariances'][0])
                else:
                    # Use pyriemann-style mean (geometric mean approximation)
                    mean_cov = np.mean(result['covariances'], axis=0)
                    covs_list.append(mean_cov)
                labels.append(label_map.get(label, 0))

        if len(covs_list) > 0 and len(set(labels)) == 2:
            subject_covs[subject_id] = {
                'covs': np.array(covs_list),
                'labels': np.array(labels)
            }

    if len(subject_covs) < 2:
        return {
            'error': 'Not enough subjects',
            'covariance_quality': {}
        }

    # Compute condition number statistics
    raw_conds_finite = [c for c in all_raw_conditions if np.isfinite(c)]
    reg_conds_finite = [c for c in all_reg_conditions if np.isfinite(c)]

    cov_quality = {
        'median_condition_raw': float(np.median(raw_conds_finite)) if raw_conds_finite else None,
        'median_condition_regularized': float(np.median(reg_conds_finite)) if reg_conds_finite else None,
        'pct_ill_conditioned': 100 * sum(1 for c in raw_conds_finite if c > CONDITION_THRESHOLD) / len(raw_conds_finite) if raw_conds_finite else None
    }

    # Run LOSO with TangentSpace + LDA
    per_subject = {}
    all_y_true = []
    all_y_prob = []

    for test_subject in subject_covs.keys():
        # Train data
        X_train_list = []
        y_train_list = []
        for train_subject, sc in subject_covs.items():
            if train_subject != test_subject:
                X_train_list.append(sc['covs'])
                y_train_list.append(sc['labels'])

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = subject_covs[test_subject]['covs']
        y_test = subject_covs[test_subject]['labels']

        try:
            # Tangent space projection
            ts = TangentSpace(metric='riemann')
            ts.fit(X_train, y_train)
            X_train_ts = ts.transform(X_train)
            X_test_ts = ts.transform(X_test)

            # LDA classifier
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train_ts, y_train)
            y_prob = lda.predict_proba(X_test_ts)[:, 1]

            auc = roc_auc_score(y_test, y_prob)
        except Exception as e:
            logger.debug(f"Tier 3 failed for {test_subject}: {e}")
            auc = 0.5
            y_prob = np.full(len(y_test), 0.5)

        per_subject[test_subject] = {'auc': auc, 'n_samples': len(y_test)}
        all_y_true.extend(y_test)
        all_y_prob.extend(y_prob)

    try:
        overall_auc = roc_auc_score(all_y_true, all_y_prob)
    except:
        overall_auc = 0.5

    return {
        'overall_auc': overall_auc,
        'per_subject': per_subject,
        'covariance_quality': cov_quality
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_experiment():
    """Run the full window size experiment."""
    logger.info("=" * 70)
    logger.info("PROMPT 015 - Window Size Impact on Classical Baselines")
    logger.info("=" * 70)
    logger.info(f"Window sizes: {WINDOW_SIZES}")
    logger.info(f"Subjects: {PRIORITY_SUBJECTS}")

    # Load data for all subjects
    logger.info("Loading subject data...")
    subject_data = {}
    for subject_id in PRIORITY_SUBJECTS:
        segments = load_subject_segments(subject_id)
        if segments:
            subject_data[subject_id] = segments
            n_ictal = sum(1 for _, label in segments if label in ('ictal', 'preictal'))
            n_inter = sum(1 for _, label in segments if label == 'interictal')
            logger.info(f"  {subject_id}: {len(segments)} segments ({n_ictal} ictal, {n_inter} interictal)")

    if len(subject_data) < 2:
        logger.error("Not enough subjects with data")
        return

    # Run experiment for each window size
    results = {
        'experiment_config': {
            'window_sizes_seconds': WINDOW_SIZES,
            'channels': CHANNELS,
            'subjects': list(subject_data.keys()),
            'segment_duration_seconds': SEGMENT_DURATION_SEC
        },
        'tier2_max': {},
        'tier3_riemannian': {},
        'covariance_quality': {}
    }

    for window_sec in WINDOW_SIZES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Window size: {window_sec}s")
        logger.info(f"{'='*60}")

        n_subwindows = SEGMENT_DURATION_SEC // window_sec
        logger.info(f"  Sub-windows per segment: {n_subwindows}")

        # Tier 2
        logger.info("  Running Tier 2 (Correlation Eigenvalues + XGBoost)...")
        tier2_result = run_tier2_loso_for_window(subject_data, window_sec)

        results['tier2_max'][str(window_sec)] = {
            'overall_auc': tier2_result['aggregation']['max']['overall_auc'],
            'per_subject': tier2_result['aggregation']['max']['per_subject'],
            'n_subwindows_per_segment': tier2_result['n_subwindows_per_segment'],
            'aggregation': {
                'max': tier2_result['aggregation']['max']['overall_auc'],
                'mean': tier2_result['aggregation']['mean']['overall_auc'],
                'std': tier2_result['aggregation']['std']['overall_auc']
            }
        }

        logger.info(f"    MAX AUC: {tier2_result['aggregation']['max']['overall_auc']:.4f}")
        logger.info(f"    MEAN AUC: {tier2_result['aggregation']['mean']['overall_auc']:.4f}")
        if tier2_result['aggregation']['std']['overall_auc']:
            logger.info(f"    STD AUC: {tier2_result['aggregation']['std']['overall_auc']:.4f}")

        # Tier 3
        logger.info("  Running Tier 3 (Riemannian TangentSpace + LDA)...")
        tier3_result = run_tier3_loso_for_window(subject_data, window_sec)

        if 'error' not in tier3_result:
            results['tier3_riemannian'][str(window_sec)] = {
                'overall_auc': tier3_result['overall_auc'],
                'per_subject': tier3_result['per_subject'],
                'n_subwindows_per_segment': n_subwindows
            }
            results['covariance_quality'][str(window_sec)] = tier3_result['covariance_quality']

            logger.info(f"    AUC: {tier3_result['overall_auc']:.4f}")
            cq = tier3_result['covariance_quality']
            if cq['median_condition_raw']:
                logger.info(f"    Condition (raw): {cq['median_condition_raw']:.1f}")
            if cq['median_condition_regularized']:
                logger.info(f"    Condition (reg): {cq['median_condition_regularized']:.1f}")
        else:
            logger.warning(f"    Tier 3 failed: {tier3_result['error']}")

    # Compute quantum projection
    tier2_aucs = [(float(w), r['overall_auc']) for w, r in results['tier2_max'].items()
                   if r['overall_auc'] is not None]
    if tier2_aucs:
        best_window, best_auc = max(tier2_aucs, key=lambda x: x[1])
        results['quantum_projection'] = {
            'current_tier2_auc': CURRENT_TIER2_AUC,
            'current_window_seconds': CURRENT_WINDOW_SEC,
            'best_tier2_auc': best_auc,
            'best_window_seconds': best_window,
            'projected_improvement': best_auc - CURRENT_TIER2_AUC
        }

    # Save JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "window_size_impact.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved: {json_path}")

    # Generate markdown report
    report = generate_report(results)
    md_path = OUTPUT_DIR / "window_size_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"Saved: {md_path}")

    # Print summary
    print_summary(results)

    return results


def generate_report(results: Dict) -> str:
    """Generate markdown report."""
    md = f"""# Window Size Impact on Seizure Detection Performance

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

- **Segment duration**: {results['experiment_config']['segment_duration_seconds']}s
- **Channels**: {len(results['experiment_config']['channels'])} (CH8)
- **Subjects**: {', '.join(results['experiment_config']['subjects'])}
- **Window sizes tested**: {results['experiment_config']['window_sizes_seconds']}

---

## Tier 2 MAX (Correlation Eigenvalues + XGBoost)

| Window (s) | Sub-windows | MAX AUC | MEAN AUC | STD AUC |
|------------|-------------|---------|----------|---------|
"""

    for ws in results['experiment_config']['window_sizes_seconds']:
        ws_str = str(ws)
        if ws_str in results['tier2_max']:
            r = results['tier2_max'][ws_str]
            max_auc = f"{r['aggregation']['max']:.4f}" if r['aggregation']['max'] else "N/A"
            mean_auc = f"{r['aggregation']['mean']:.4f}" if r['aggregation']['mean'] else "N/A"
            std_auc = f"{r['aggregation']['std']:.4f}" if r['aggregation']['std'] else "N/A"
            md += f"| {ws} | {r['n_subwindows_per_segment']} | {max_auc} | {mean_auc} | {std_auc} |\n"

    md += """
---

## Tier 3 Riemannian (Tangent Space + LDA)

| Window (s) | Sub-windows | AUC |
|------------|-------------|-----|
"""

    for ws in results['experiment_config']['window_sizes_seconds']:
        ws_str = str(ws)
        if ws_str in results['tier3_riemannian']:
            r = results['tier3_riemannian'][ws_str]
            auc = f"{r['overall_auc']:.4f}" if r['overall_auc'] else "N/A"
            md += f"| {ws} | {r['n_subwindows_per_segment']} | {auc} |\n"

    md += """
---

## Covariance Matrix Condition Number

| Window (s) | Median Condition (raw) | Median Condition (reg) | % Ill-conditioned |
|------------|------------------------|------------------------|-------------------|
"""

    for ws in results['experiment_config']['window_sizes_seconds']:
        ws_str = str(ws)
        if ws_str in results['covariance_quality']:
            cq = results['covariance_quality'][ws_str]
            raw = f"{cq['median_condition_raw']:.1f}" if cq['median_condition_raw'] else "N/A"
            reg = f"{cq['median_condition_regularized']:.1f}" if cq['median_condition_regularized'] else "N/A"
            ill = f"{cq['pct_ill_conditioned']:.1f}%" if cq['pct_ill_conditioned'] is not None else "N/A"
            md += f"| {ws} | {raw} | {reg} | {ill} |\n"

    # Key findings
    md += """
---

## Key Findings

"""

    # Find best window for each tier
    tier2_aucs = [(float(w), r['aggregation']['max']) for w, r in results['tier2_max'].items()
                   if r['aggregation']['max'] is not None]
    tier3_aucs = [(float(w), r['overall_auc']) for w, r in results['tier3_riemannian'].items()
                   if r['overall_auc'] is not None]

    if tier2_aucs:
        best_t2_window, best_t2_auc = max(tier2_aucs, key=lambda x: x[1])
        worst_t2_window, worst_t2_auc = min(tier2_aucs, key=lambda x: x[1])
        improvement = best_t2_auc - worst_t2_auc

        md += f"### 1. Tier 2 Performance by Window Size\n\n"
        md += f"- **Best window**: {best_t2_window}s (AUC = {best_t2_auc:.4f})\n"
        md += f"- **Worst window**: {worst_t2_window}s (AUC = {worst_t2_auc:.4f})\n"
        md += f"- **Improvement**: {improvement:.4f} ({100*improvement/worst_t2_auc:.1f}%)\n\n"

    if tier3_aucs:
        best_t3_window, best_t3_auc = max(tier3_aucs, key=lambda x: x[1])
        worst_t3_window, worst_t3_auc = min(tier3_aucs, key=lambda x: x[1])
        improvement = best_t3_auc - worst_t3_auc

        md += f"### 2. Tier 3 Performance by Window Size\n\n"
        md += f"- **Best window**: {best_t3_window}s (AUC = {best_t3_auc:.4f})\n"
        md += f"- **Worst window**: {worst_t3_window}s (AUC = {worst_t3_auc:.4f})\n"
        md += f"- **Improvement**: {improvement:.4f}\n\n"

    # Compare Tier 2 vs Tier 3 improvement
    if tier2_aucs and tier3_aucs:
        t2_improvement = max(tier2_aucs, key=lambda x: x[1])[1] - min(tier2_aucs, key=lambda x: x[1])[1]
        t3_improvement = max(tier3_aucs, key=lambda x: x[1])[1] - min(tier3_aucs, key=lambda x: x[1])[1]

        md += f"### 3. Tier 2 vs Tier 3 Sensitivity to Window Size\n\n"
        if t3_improvement > t2_improvement:
            md += f"- Tier 3 improves **more** with window size (+{t3_improvement:.4f}) than Tier 2 (+{t2_improvement:.4f})\n"
            md += f"- This supports the SPD stability argument: covariance quality is the bottleneck for Riemannian methods\n\n"
        else:
            md += f"- Tier 2 improves **more** with window size (+{t2_improvement:.4f}) than Tier 3 (+{t3_improvement:.4f})\n"
            md += f"- Eigenvalue features benefit equally from longer windows\n\n"

    # Quantum projection
    if 'quantum_projection' in results:
        qp = results['quantum_projection']
        md += f"### 4. Quantum Improvement Projection\n\n"
        md += f"- Current Tier 2 AUC (at {qp['current_window_seconds']}s): {qp['current_tier2_auc']:.4f}\n"
        md += f"- Best Tier 2 AUC (at {qp['best_window_seconds']}s): {qp['best_tier2_auc']:.4f}\n"
        md += f"- **Projected improvement from window size alone**: {qp['projected_improvement']:+.4f}\n\n"

        if qp['projected_improvement'] > 0.01:
            md += f"If quantum encoding benefits proportionally from longer windows, this suggests "
            md += f"up to +{qp['projected_improvement']:.3f} AUC improvement may be achievable by "
            md += f"using {qp['best_window_seconds']}s windows instead of {qp['current_window_seconds']}s.\n\n"

    # SPD manifold argument
    md += """---

## SPD Manifold Argument Assessment

"""

    # Check if condition number drops with window size
    cond_numbers = [(float(w), cq['median_condition_regularized'])
                    for w, cq in results['covariance_quality'].items()
                    if cq.get('median_condition_regularized') is not None]

    if len(cond_numbers) >= 2:
        sorted_conds = sorted(cond_numbers, key=lambda x: x[0])
        short_window_cond = sorted_conds[0][1]
        long_window_cond = sorted_conds[-1][1]

        if short_window_cond > long_window_cond:
            md += f"**Condition number decreases with window size**: {short_window_cond:.1f} (at {sorted_conds[0][0]}s) -> {long_window_cond:.1f} (at {sorted_conds[-1][0]}s)\n\n"
            md += "This supports the hypothesis that longer windows produce better-conditioned SPD matrices.\n\n"
        else:
            md += f"**Condition number does NOT decrease with window size**: {short_window_cond:.1f} (at {sorted_conds[0][0]}s) -> {long_window_cond:.1f} (at {sorted_conds[-1][0]}s)\n\n"

    # Check if Riemannian improves more than eigenvalues
    if tier2_aucs and tier3_aucs:
        t2_relative_improvement = (max(tier2_aucs, key=lambda x: x[1])[1] - min(tier2_aucs, key=lambda x: x[1])[1]) / min(tier2_aucs, key=lambda x: x[1])[1]
        t3_relative_improvement = (max(tier3_aucs, key=lambda x: x[1])[1] - min(tier3_aucs, key=lambda x: x[1])[1]) / max(0.01, min(tier3_aucs, key=lambda x: x[1])[1])

        if t3_relative_improvement > t2_relative_improvement * 1.5:
            md += "**SUPPORTED**: Riemannian methods benefit disproportionately from longer windows.\n"
            md += "Current quantum performance may be limited by covariance estimation quality from short windows, "
            md += "not fundamental encoding limitations. Longer windows should produce cleaner density matrix "
            md += "encodings for the quantum circuit.\n"
        else:
            md += "**NOT STRONGLY SUPPORTED**: Eigenvalue features benefit similarly from longer windows.\n"
            md += "The quantum encoding bottleneck may not be primarily a window size / covariance quality issue.\n"

    return md


def print_summary(results: Dict):
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nTier 2 MAX AUC by Window Size:")
    for ws in results['experiment_config']['window_sizes_seconds']:
        ws_str = str(ws)
        if ws_str in results['tier2_max'] and results['tier2_max'][ws_str]['aggregation']['max']:
            auc = results['tier2_max'][ws_str]['aggregation']['max']
            print(f"  {ws}s: {auc:.4f}")

    print("\nTier 3 Riemannian AUC by Window Size:")
    for ws in results['experiment_config']['window_sizes_seconds']:
        ws_str = str(ws)
        if ws_str in results['tier3_riemannian'] and results['tier3_riemannian'][ws_str]['overall_auc']:
            auc = results['tier3_riemannian'][ws_str]['overall_auc']
            print(f"  {ws}s: {auc:.4f}")

    if 'quantum_projection' in results:
        qp = results['quantum_projection']
        print(f"\nQuantum Projection:")
        print(f"  Current baseline ({qp['current_window_seconds']}s): {qp['current_tier2_auc']:.4f}")
        print(f"  Best result ({qp['best_window_seconds']}s): {qp['best_tier2_auc']:.4f}")
        print(f"  Projected improvement: {qp['projected_improvement']:+.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run PROMPT 015 experiment."""
    results = run_experiment()
    return results


if __name__ == "__main__":
    main()
