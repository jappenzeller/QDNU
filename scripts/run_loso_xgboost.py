#!/usr/bin/env python3
"""
================================================================================
CHB-MIT LOSO CV - V4 (XGBoost with Bagging)
================================================================================

Local script that reuses the channel-harmonized feature extraction from
sagemaker/train_chbmit.py and runs LOSO CV with XGBoost + 5-bag ensemble.

This script:
1. Extracts features using the same pipeline as the SageMaker job (or loads from cache)
2. Runs LOSO CV with XGBoost + 5-bag ensemble
3. Saves results and prints comparison table

================================================================================
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
import logging

# Add parent directory to path to import from sagemaker module
sys.path.insert(0, str(Path(__file__).parent.parent))

from sagemaker.train_chbmit import (
    COMMON_CHANNELS,
    EXCLUDE_SUBJECTS,
    FS,
    WINDOW_SEC,
    EEGSegment,
    FeatureSegment,
    SegmentResult,
    preprocess_eeg,
    extract_all_features,
    normalize_channel_label,
    read_edf_segment,
    parse_summary_file,
    extract_segments_for_subject,
    analyze_segment,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Data location
DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
OUTPUT_DIR = Path("analysis_results/loso_v2_xgboost")
CACHE_FILE = Path("analysis_results/feature_cache.npz")

# XGBoost parameters
N_BAGS = 5


# =============================================================================
# XGBOOST MODEL
# =============================================================================

def make_xgb_model(seed: int = 42):
    """Create XGBoost classifier with optimal parameters."""
    from xgboost import XGBClassifier

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
# FEATURE EXTRACTION
# =============================================================================

def extract_all_features_cached() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features for all subjects, using cache if available."""

    if CACHE_FILE.exists():
        logger.info(f"Loading cached features from {CACHE_FILE}")
        data = np.load(CACHE_FILE, allow_pickle=True)
        return data['features'], data['labels'], data['subjects']

    logger.info("Extracting features from EDF files...")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Using {len(COMMON_CHANNELS)} common channels")

    # Find subjects
    all_subjects = sorted([d for d in DATA_DIR.iterdir()
                           if d.is_dir() and d.name.startswith('chb')])
    subjects = [d for d in all_subjects if d.name not in EXCLUDE_SUBJECTS]
    logger.info(f"Found {len(all_subjects)} subjects, using {len(subjects)} (excluding {EXCLUDE_SUBJECTS})")

    all_features = []
    all_labels = []
    all_subjects_arr = []

    for subject_dir in subjects:
        logger.info(f"\nProcessing {subject_dir.name}")

        segments = extract_segments_for_subject(subject_dir)

        ictal = sum(1 for s in segments if s.label == 'ictal')
        preictal = sum(1 for s in segments if s.label == 'preictal')
        interictal = sum(1 for s in segments if s.label == 'interictal')
        logger.info(f"  Segments: {ictal} ictal, {preictal} preictal, {interictal} interictal")

        subject_count = 0
        for seg in segments:
            result = analyze_segment(seg, DATA_DIR, target_channels=COMMON_CHANNELS)
            if result:
                seg_result, feat_seg = result
                all_features.append(feat_seg.features)
                all_labels.append(feat_seg.label)
                all_subjects_arr.append(feat_seg.subject)
                subject_count += 1

        logger.info(f"  Extracted: {subject_count} segments")

    features = np.array(all_features)
    labels = np.array(all_labels)
    subjects_arr = np.array(all_subjects_arr)

    logger.info(f"\nTotal: {len(features)} segments, {features.shape[1]} features")

    # Save cache
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE_FILE, features=features, labels=labels, subjects=subjects_arr)
    logger.info(f"Saved feature cache to {CACHE_FILE}")

    return features, labels, subjects_arr


# =============================================================================
# LOSO CV WITH XGBOOST + BAGGING
# =============================================================================

def run_loso_cv_xgboost(features: np.ndarray, labels: np.ndarray,
                        subjects: np.ndarray) -> Dict:
    """Run LOSO CV with XGBoost + 5-bag ensemble."""
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, roc_curve)

    # Label mapping: ictal + preictal = 1, interictal = 0
    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}
    y = np.array([label_map.get(l, 0) for l in labels])

    unique_subjects = sorted(set(subjects))

    results = {'per_subject': {}, 'overall': {}, 'model_params': {}}
    all_preds, all_true, all_proba = [], [], []
    all_feature_importances = []

    logger.info(f"\nLOSO CV with XGBoost + {N_BAGS}-bag ensemble")
    logger.info(f"Subjects: {len(unique_subjects)}, Segments: {len(features)}")
    logger.info("-" * 60)

    for test_subject in unique_subjects:
        # Split data
        train_mask = subjects != test_subject
        test_mask = subjects == test_subject

        X_train = features[train_mask]
        y_train = y[train_mask]
        X_test = features[test_mask]
        y_test = y[test_mask]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.info(f"  {test_subject}: skipped (single class)")
            continue

        # Train 5-bag ensemble
        models = []
        for bag in range(N_BAGS):
            rng = np.random.RandomState(42 + bag)
            idx = rng.choice(len(X_train), size=len(X_train), replace=True)
            model = make_xgb_model(seed=42 + bag)
            model.fit(X_train[idx], y_train[idx])
            models.append(model)

        # Predict: average probabilities across bags
        y_proba = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
        y_pred = (y_proba >= 0.5).astype(int)

        # Collect feature importances
        importances = np.mean([m.feature_importances_ for m in models], axis=0)
        all_feature_importances.append(importances)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = 0.5

        # Optimal threshold (Youden's J)
        try:
            fpr, tpr, thresholds = roc_curve(y_test, y_proba)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred_opt = (y_proba >= optimal_threshold).astype(int)
            acc_opt = accuracy_score(y_test, y_pred_opt)
        except:
            optimal_threshold = 0.5
            acc_opt = acc

        results['per_subject'][test_subject] = {
            'n_train': int(len(X_train)),
            'n_test': int(len(X_test)),
            'accuracy': float(acc),
            'accuracy_optimized': float(acc_opt),
            'optimal_threshold': float(optimal_threshold),
            'precision': float(prec),
            'recall': float(rec),
            'f1': float(f1),
            'auc': float(auc),
        }

        all_preds.extend(y_pred.tolist())
        all_true.extend(y_test.tolist())
        all_proba.extend(y_proba.tolist())

        logger.info(f"  {test_subject}: Acc={acc:.3f} (opt={acc_opt:.3f}) F1={f1:.3f} AUC={auc:.3f}")

    # Overall metrics
    if all_preds:
        results['overall'] = {
            'n_segments': len(all_true),
            'accuracy': float(accuracy_score(all_true, all_preds)),
            'precision': float(precision_score(all_true, all_preds, zero_division=0)),
            'recall': float(recall_score(all_true, all_preds, zero_division=0)),
            'f1': float(f1_score(all_true, all_preds, zero_division=0)),
        }
        try:
            results['overall']['auc'] = float(roc_auc_score(all_true, all_proba))
        except ValueError:
            results['overall']['auc'] = 0.5

    # Average feature importances
    avg_importances = np.mean(all_feature_importances, axis=0)
    top_indices = np.argsort(avg_importances)[::-1][:20]
    results['top_features'] = [
        {'index': int(i), 'importance': float(avg_importances[i])}
        for i in top_indices
    ]

    results['model_params'] = {
        'classifier': 'XGBoost',
        'n_estimators': 200,
        'n_bags': N_BAGS,
        'max_depth': 6,
        'learning_rate': 0.1,
        'window_sec': WINDOW_SEC,
        'n_features': features.shape[1],
        'n_channels': len(COMMON_CHANNELS),
        'feature_set': 'ClassicalBaselineV4_XGBoost'
    }

    return results


# =============================================================================
# COMPARISON
# =============================================================================

def print_comparison(v4_results: Dict):
    """Print comparison table across versions."""

    # V1 results (11 subjects, basic RF) - from earlier runs
    v1 = {
        'accuracy': 0.6875,
        'auc': 0.7072,
        'f1': 0.7680,
        'precision': 0.7095,
        'recall': 0.8371,
        'features': 552,
        'classifier': 'RF-100',
        'subjects': 11,
    }

    # V3 results (22 subjects, full features, RF)
    v3 = {
        'accuracy': 0.7099,
        'auc': 0.7990,
        'f1': 0.7403,
        'precision': 0.7256,
        'recall': 0.7556,
        'features': 2016,
        'classifier': 'RF-300',
        'subjects': 22,
    }

    # V4 results (22 subjects, full features, XGBoost + bagging)
    overall = v4_results.get('overall', {})
    params = v4_results.get('model_params', {})
    v4 = {
        'accuracy': overall.get('accuracy', 0),
        'auc': overall.get('auc', 0),
        'f1': overall.get('f1', 0),
        'precision': overall.get('precision', 0),
        'recall': overall.get('recall', 0),
        'features': params.get('n_features', 2016),
        'classifier': f"XGB-{params.get('n_estimators', 200)}x{params.get('n_bags', 5)}bags",
        'subjects': len(v4_results.get('per_subject', {})),
    }

    print("\n" + "=" * 70)
    print("CLASSICAL BASELINE PROGRESSION")
    print("=" * 70)
    print(f"{'Metric':<15} {'V1 (11subj)':<15} {'V3 (22subj/RF)':<17} {'V4 (22subj/XGB)':<17}")
    print("-" * 70)
    print(f"{'Accuracy':<15} {v1['accuracy']*100:>6.2f}%        {v3['accuracy']*100:>6.2f}%          {v4['accuracy']*100:>6.2f}%")
    print(f"{'AUC':<15} {v1['auc']*100:>6.2f}%        {v3['auc']*100:>6.2f}%          {v4['auc']*100:>6.2f}%")
    print(f"{'F1':<15} {v1['f1']*100:>6.2f}%        {v3['f1']*100:>6.2f}%          {v4['f1']*100:>6.2f}%")
    print(f"{'Precision':<15} {v1['precision']*100:>6.2f}%        {v3['precision']*100:>6.2f}%          {v4['precision']*100:>6.2f}%")
    print(f"{'Recall':<15} {v1['recall']*100:>6.2f}%        {v3['recall']*100:>6.2f}%          {v4['recall']*100:>6.2f}%")
    print("-" * 70)
    print(f"{'Features':<15} {v1['features']:<15} {v3['features']:<17} {v4['features']:<17}")
    print(f"{'Classifier':<15} {v1['classifier']:<15} {v3['classifier']:<17} {v4['classifier']:<17}")
    print(f"{'Subjects':<15} {v1['subjects']:<15} {v3['subjects']:<17} {v4['subjects']:<17}")
    print("=" * 70)

    # Print improvement
    acc_imp = (v4['accuracy'] - v3['accuracy']) * 100
    auc_imp = (v4['auc'] - v3['auc']) * 100
    print(f"\nV3 -> V4 improvement: Accuracy {acc_imp:+.2f}%, AUC {auc_imp:+.2f}%")

    return {'V1': v1, 'V3': v3, 'V4': v4}


def print_top_features(results: Dict, n: int = 20):
    """Print top N most important features."""

    # Feature index ranges (for 18 channels)
    n_ch = 18
    feature_ranges = [
        (0, n_ch * 6, 'Band power (coarse)'),
        (n_ch * 6, n_ch * 6 + n_ch * 20, 'Fine spectral (2Hz)'),
        (n_ch * 26, n_ch * 26 + n_ch * 46, 'FFT magnitudes'),
        (n_ch * 72, n_ch * 72 + n_ch * 6, 'Statistics'),
        (n_ch * 78, n_ch * 78 + n_ch * 3, 'Hjorth'),
        (n_ch * 81, n_ch * 81 + n_ch, 'Spectral entropy'),
        (n_ch * 82, n_ch * 82 + n_ch * 6, 'AR error'),
        (n_ch * 88, n_ch * 88 + n_ch * 3, 'Zero crossings'),
        (n_ch * 91, n_ch * 91 + n_ch, 'Nonlinear energy'),
        (n_ch * 92, n_ch * 92 + n_ch, 'RMS'),
        (n_ch * 93, n_ch * 93 + 153, 'Cross-correlations'),
        (n_ch * 93 + 153, n_ch * 93 + 153 + n_ch, 'Corr eigenvalues'),
        (n_ch * 93 + 153 + n_ch, n_ch * 93 + 153 + n_ch + 153, 'Freq cross-corr'),
        (n_ch * 93 + 153 + n_ch + 153, 2016, 'Freq corr eigenvalues'),
    ]

    def get_feature_name(idx: int) -> str:
        for start, end, name in feature_ranges:
            if start <= idx < end:
                return f"{name} [{idx - start}]"
        return f"Feature {idx}"

    print("\n" + "=" * 50)
    print(f"TOP {n} MOST IMPORTANT FEATURES")
    print("=" * 50)

    top_features = results.get('top_features', [])[:n]
    for i, feat in enumerate(top_features):
        idx = feat['index']
        imp = feat['importance']
        name = get_feature_name(idx)
        print(f"{i+1:2d}. [{idx:4d}] {name:<30} {imp:.4f}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function."""
    logger.info("=" * 70)
    logger.info("CHB-MIT LOSO CV - V4 (XGBoost with 5-bag Bagging)")
    logger.info("=" * 70)
    logger.info(f"Started: {datetime.now()}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract or load features
    features, labels, subjects = extract_all_features_cached()
    logger.info(f"\nFeatures shape: {features.shape}")
    logger.info(f"Labels: {len(labels)}")
    logger.info(f"Unique subjects: {len(set(subjects))}")

    # Handle NaN/Inf
    features = np.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)

    # Run LOSO CV
    logger.info("\n" + "=" * 70)
    logger.info("LOSO CROSS-VALIDATION WITH XGBOOST")
    logger.info("=" * 70)

    results = run_loso_cv_xgboost(features, labels, subjects)

    # Save results
    with open(OUTPUT_DIR / 'loso_classification.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    n_ictal = sum(1 for l in labels if l == 'ictal')
    n_preictal = sum(1 for l in labels if l == 'preictal')
    n_interictal = sum(1 for l in labels if l == 'interictal')

    summary = {
        'n_subjects': len(set(subjects)),
        'n_segments': len(labels),
        'n_ictal': n_ictal,
        'n_preictal': n_preictal,
        'n_interictal': n_interictal,
        'loso_results': results.get('overall', {}),
        'model_params': results.get('model_params', {}),
        'completed': datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print comparison
    comparison = print_comparison(results)

    with open(OUTPUT_DIR / 'comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)

    # Print top features
    print_top_features(results)

    # Final summary
    logger.info(f"\n{'=' * 70}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'=' * 70}")
    overall = results.get('overall', {})
    logger.info(f"Accuracy:  {overall.get('accuracy', 0):.3f}")
    logger.info(f"AUC:       {overall.get('auc', 0):.3f}")
    logger.info(f"F1 Score:  {overall.get('f1', 0):.3f}")
    logger.info(f"Precision: {overall.get('precision', 0):.3f}")
    logger.info(f"Recall:    {overall.get('recall', 0):.3f}")

    logger.info(f"\nResults saved to: {OUTPUT_DIR}")
    logger.info(f"Completed: {datetime.now()}")


if __name__ == '__main__':
    main()
