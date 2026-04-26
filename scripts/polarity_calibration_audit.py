#!/usr/bin/env python3
"""
================================================================================
PROMPT 022 — Polarity Calibration Validity Audit
================================================================================

RUNS BEFORE ANY HARDWARE SPEND.

Tests whether the current calibrated AUC numbers (Tier 3 0.659, Tier 2A 0.628,
A-Gate 0.637) survive clean (leakage-free) polarity calibration.

Two clean calibration approaches:
  Approach A: Inner CV on training fold (majority vote of training polarities)
  Approach B: Label-free sign from covariance geometry (eigenvalue trace)

Plus: Tier 3 + XGBoost experiment to test if polarity is geometric or LDA artifact.

Output: results/calibration/clean_calibration_audit.json
================================================================================
"""

import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sagemaker.train_chbmit import (
    FS, WINDOW_SEC, preprocess_eeg,
    extract_segments_for_subject, EEGSegment,
)

from sagemaker.train_full_cohort import (
    compute_covariance_ledoit_wolf, extract_eigenvalues, make_spd,
    read_edf_segment, normalize_channel_label,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
OUTPUT_DIR = PROJECT_ROOT / "results" / "calibration"

PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']

QUANTUM_CHANNELS = [
    "FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
    "FP2-F8", "F8-T8", "FP2-F4", "F4-C4",
]

# A-Gate hardware results from PROMPT 006
AGATE_HW_RESULTS = PROJECT_ROOT / "results" / "hardware_validation" / "ch8_heron_loso_results.json"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_subject_features(subject_id: str) -> Optional[Dict]:
    """Load EEG data and compute features for one subject.

    Returns dict with tier2a_X, tier3_covs, y, eigenvalue_sums_per_segment.
    """
    subject_dir = DATA_DIR / subject_id
    if not subject_dir.exists():
        return None

    segments = extract_segments_for_subject(subject_dir)
    if not segments:
        return None

    label_map = {'interictal': 0, 'preictal': 1, 'ictal': 1}

    tier2a_features = []
    tier3_covs = []
    labels = []
    eigenvalue_sums = []  # For Approach B: trace of covariance
    segment_labels_raw = []  # 'ictal', 'interictal', etc.

    for seg in segments:
        edf_path = subject_dir / seg.source_file
        if not edf_path.exists():
            continue

        data, fs = read_edf_segment(str(edf_path), seg.start_sec, seg.end_sec,
                                     target_channels=QUANTUM_CHANNELS)
        if data is None or data.size == 0:
            continue

        min_samples = int(WINDOW_SEC * fs * 0.5)
        if data.shape[1] < min_samples:
            continue

        data = preprocess_eeg(data, fs=int(fs))

        # Covariance and eigenvalues
        cov = compute_covariance_ledoit_wolf(data)
        eigvals = extract_eigenvalues(cov)
        cov_spd = make_spd(cov)

        # Eigenvalue sum (trace) for Approach B
        raw_eigvals = np.real(np.linalg.eigvals(cov))
        eig_sum = float(np.sum(raw_eigvals))

        tier2a_features.append(eigvals)
        tier3_covs.append(cov_spd)
        eigenvalue_sums.append(eig_sum)

        label = label_map.get(seg.label, 0)
        labels.append(label)
        segment_labels_raw.append(seg.label)

    if not tier2a_features:
        return None

    y = np.array(labels)
    if len(np.unique(y)) < 2:
        return None

    return {
        'tier2a_X': np.array(tier2a_features),
        'tier3_covs': np.array(tier3_covs),
        'y': y,
        'eigenvalue_sums': np.array(eigenvalue_sums),
        'segment_labels': segment_labels_raw,
        'n_seizure': int(np.sum(y)),
        'n_segments': len(y),
    }


def load_all_subjects() -> Dict[str, Dict]:
    """Load features for all priority subjects."""
    patient_data = {}
    for subj in PRIORITY_SUBJECTS:
        logger.info(f"Loading {subj}...")
        data = load_subject_features(subj)
        if data is not None:
            patient_data[subj] = data
            logger.info(f"  {subj}: {data['n_segments']} segments, {data['n_seizure']} seizure")
        else:
            logger.warning(f"  {subj}: FAILED to load")
    return patient_data


# =============================================================================
# CLASSIFIERS
# =============================================================================

def run_tier3_lda(patient_data: Dict, subjects: List[str]) -> Dict[str, Dict]:
    """Run Tier 3 (TS+LDA) LOSO, return per-subject raw AUC and score arrays."""
    from pyriemann.tangentspace import TangentSpace

    results = {}
    for test_subj in subjects:
        train_subjs = [s for s in subjects if s != test_subj]

        X_train = np.vstack([patient_data[s]['tier3_covs'] for s in train_subjs])
        y_train = np.concatenate([patient_data[s]['y'] for s in train_subjs])
        X_test = patient_data[test_subj]['tier3_covs']
        y_test = patient_data[test_subj]['y']

        try:
            ts = TangentSpace()
            X_tr_ts = ts.fit_transform(X_train)
            X_te_ts = ts.transform(X_test)

            lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            lda.fit(X_tr_ts, y_train)
            y_proba = lda.predict_proba(X_te_ts)[:, 1]
            raw_auc = roc_auc_score(y_test, y_proba)
        except Exception as e:
            logger.warning(f"  Tier3 LDA {test_subj} failed: {e}")
            raw_auc = 0.5
            y_proba = np.full(len(y_test), 0.5)

        results[test_subj] = {
            'raw_auc': float(raw_auc),
            'scores': y_proba.tolist(),
            'y_true': y_test.tolist(),
        }
    return results


def run_tier2a_xgb(patient_data: Dict, subjects: List[str]) -> Dict[str, Dict]:
    """Run Tier 2A (Eigenvalues + XGBoost) LOSO."""
    results = {}
    for test_subj in subjects:
        train_subjs = [s for s in subjects if s != test_subj]

        X_train = np.vstack([patient_data[s]['tier2a_X'] for s in train_subjs])
        y_train = np.concatenate([patient_data[s]['y'] for s in train_subjs])
        X_test = patient_data[test_subj]['tier2a_X']
        y_test = patient_data[test_subj]['y']

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42,
            eval_metric='logloss', n_jobs=-1,
        )
        model.fit(X_train_s, y_train)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        try:
            raw_auc = roc_auc_score(y_test, y_proba)
        except:
            raw_auc = 0.5

        results[test_subj] = {
            'raw_auc': float(raw_auc),
            'scores': y_proba.tolist(),
            'y_true': y_test.tolist(),
        }
    return results


def run_tier3_xgb(patient_data: Dict, subjects: List[str]) -> Dict[str, Dict]:
    """Run Tier 3 + XGBoost (tangent space features + XGBoost) LOSO.

    Purpose: test whether polarity is geometric or an LDA artifact.
    """
    from pyriemann.tangentspace import TangentSpace

    results = {}
    for test_subj in subjects:
        train_subjs = [s for s in subjects if s != test_subj]

        X_train_cov = np.vstack([patient_data[s]['tier3_covs'] for s in train_subjs])
        y_train = np.concatenate([patient_data[s]['y'] for s in train_subjs])
        X_test_cov = patient_data[test_subj]['tier3_covs']
        y_test = patient_data[test_subj]['y']

        try:
            ts = TangentSpace()
            X_train = ts.fit_transform(X_train_cov)
            X_test = ts.transform(X_test_cov)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Reasonable defaults per prompt spec
            model = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42, eval_metric='logloss', n_jobs=-1,
            )
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]
            raw_auc = roc_auc_score(y_test, y_proba)
        except Exception as e:
            logger.warning(f"  Tier3 XGB {test_subj} failed: {e}")
            raw_auc = 0.5
            y_proba = np.full(len(y_test), 0.5)

        results[test_subj] = {
            'raw_auc': float(raw_auc),
            'scores': y_proba.tolist(),
            'y_true': y_test.tolist(),
        }
    return results


# =============================================================================
# APPROACH A: INNER CV ON TRAINING FOLD
# =============================================================================

def approach_a_calibrate(classifier_results: Dict[str, Dict],
                         classifier_name: str,
                         patient_data: Dict,
                         subjects: List[str],
                         run_classifier_fn) -> Dict[str, Dict]:
    """
    Approach A: Determine polarity from inner CV on training fold.

    For each LOSO fold (test = S):
      1. Within the 6 training patients, run LOO inner loop
      2. For each inner held-out training patient, compute their raw AUC
      3. Determine each training patient's polarity from inner loop
      4. Majority vote → assign test patient's polarity direction
      5. Apply direction to test scores and compute calibrated AUC
    """
    results = {}

    for test_subj in subjects:
        train_subjs = [s for s in subjects if s != test_subj]

        # Inner LOO: for each training patient, train on other 5, test on 1
        inner_polarities = []
        inner_details = []

        for inner_test in train_subjs:
            inner_train = [s for s in train_subjs if s != inner_test]

            if len(inner_train) < 2:
                continue

            # Run the classifier on this inner fold
            inner_results = run_classifier_fn(patient_data, inner_train + [inner_test])
            inner_auc = inner_results.get(inner_test, {}).get('raw_auc', 0.5)

            inner_pol = 'inverted' if inner_auc < 0.5 else 'standard'
            inner_polarities.append(inner_pol)
            inner_details.append({'subject': inner_test, 'inner_auc': inner_auc, 'polarity': inner_pol})

        # Majority vote
        n_inverted = sum(1 for p in inner_polarities if p == 'inverted')
        n_standard = len(inner_polarities) - n_inverted
        majority_polarity = 'inverted' if n_inverted > n_standard else 'standard'

        # Apply to test subject
        test_raw_auc = classifier_results[test_subj]['raw_auc']
        if majority_polarity == 'inverted':
            calibrated_auc = 1.0 - test_raw_auc
        else:
            calibrated_auc = test_raw_auc

        # Oracle for comparison
        oracle_polarity = 'inverted' if test_raw_auc < 0.5 else 'standard'
        oracle_auc = max(test_raw_auc, 1.0 - test_raw_auc)

        results[test_subj] = {
            'raw_auc': test_raw_auc,
            'approach_a_polarity': majority_polarity,
            'approach_a_auc': float(calibrated_auc),
            'oracle_polarity': oracle_polarity,
            'oracle_auc': float(oracle_auc),
            'agrees_with_oracle': majority_polarity == oracle_polarity,
            'n_inner_inverted': n_inverted,
            'n_inner_standard': n_standard,
            'inner_details': inner_details,
        }

    return results


# =============================================================================
# APPROACH B: LABEL-FREE SIGN FROM COVARIANCE GEOMETRY
# =============================================================================

def approach_b_calibrate(classifier_results: Dict[str, Dict],
                         patient_data: Dict,
                         subjects: List[str]) -> Dict[str, Dict]:
    """
    Approach B: Label-free sign from covariance eigenvalue trace.

    For each LOSO fold (test = S):
      1. From TRAINING patients, compute mean eigenvalue sum for ictal vs interictal
      2. If mean(Σλ_ictal) > mean(Σλ_interictal) → "standard" convention
      3. Apply sign convention to test patient scores
    """
    results = {}

    for test_subj in subjects:
        train_subjs = [s for s in subjects if s != test_subj]

        # Compute mean eigenvalue sum for ictal vs interictal across training patients
        train_ictal_sums = []
        train_inter_sums = []
        n_train_patients_with_ictal = 0

        for train_s in train_subjs:
            pdata = patient_data[train_s]
            eig_sums = pdata['eigenvalue_sums']
            y = pdata['y']

            ictal_mask = y == 1
            inter_mask = y == 0

            if np.any(ictal_mask):
                n_train_patients_with_ictal += 1
                train_ictal_sums.extend(eig_sums[ictal_mask].tolist())
            if np.any(inter_mask):
                train_inter_sums.extend(eig_sums[inter_mask].tolist())

        # Determine geometric convention from training data
        mean_ictal_trace = np.mean(train_ictal_sums) if train_ictal_sums else 0
        mean_inter_trace = np.mean(train_inter_sums) if train_inter_sums else 0

        # Standard convention: ictal has larger trace (more total variance)
        geometric_convention = 'standard' if mean_ictal_trace > mean_inter_trace else 'inverted'

        # Edge case flag
        low_ictal_flag = n_train_patients_with_ictal < 3

        # Apply to test subject
        test_raw_auc = classifier_results[test_subj]['raw_auc']
        if geometric_convention == 'inverted':
            calibrated_auc = 1.0 - test_raw_auc
        else:
            calibrated_auc = test_raw_auc

        oracle_polarity = 'inverted' if test_raw_auc < 0.5 else 'standard'
        oracle_auc = max(test_raw_auc, 1.0 - test_raw_auc)

        results[test_subj] = {
            'raw_auc': test_raw_auc,
            'approach_b_polarity': geometric_convention,
            'approach_b_auc': float(calibrated_auc),
            'oracle_polarity': oracle_polarity,
            'oracle_auc': float(oracle_auc),
            'agrees_with_oracle': geometric_convention == oracle_polarity,
            'mean_ictal_trace': float(mean_ictal_trace),
            'mean_inter_trace': float(mean_inter_trace),
            'n_train_patients_with_ictal': n_train_patients_with_ictal,
            'low_ictal_flag': low_ictal_flag,
        }

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PROMPT 022 — Polarity Calibration Validity Audit")
    print("=" * 70)
    print(f"Subjects: {PRIORITY_SUBJECTS}")
    print()

    # =========================================================================
    # Step 1: Load all subject data
    # =========================================================================
    print("[1] Loading subject data and computing features...")
    patient_data = load_all_subjects()

    valid_subjects = sorted([s for s in PRIORITY_SUBJECTS if s in patient_data])
    print(f"\nLoaded {len(valid_subjects)} subjects: {valid_subjects}")

    if len(valid_subjects) < 7:
        missing = [s for s in PRIORITY_SUBJECTS if s not in valid_subjects]
        print(f"WARNING: Missing subjects: {missing}")

    # =========================================================================
    # Step 2: Run all classifiers
    # =========================================================================
    print("\n[2] Running classifiers...")

    print("  Running Tier 3 (TS+LDA)...")
    tier3_lda_results = run_tier3_lda(patient_data, valid_subjects)

    print("  Running Tier 2A (Eigenvalues+XGBoost)...")
    tier2a_results = run_tier2a_xgb(patient_data, valid_subjects)

    print("  Running Tier 3 + XGBoost (polarity geometry test)...")
    tier3_xgb_results = run_tier3_xgb(patient_data, valid_subjects)

    # Load A-Gate hardware results
    agate_results = {}
    if AGATE_HW_RESULTS.exists():
        with open(AGATE_HW_RESULTS) as f:
            hw = json.load(f)
        for subj, data in hw['ch8']['per_subject'].items():
            agate_results[subj] = {
                'raw_auc': data['auc'],
                'scores': [],  # Not available from existing results
                'y_true': [],
            }
        print("  Loaded A-Gate hardware results from PROMPT 006")

    all_classifiers = {
        'Tier3_LDA': (tier3_lda_results, run_tier3_lda),
        'Tier2A_Eigen': (tier2a_results, run_tier2a_xgb),
        'Tier3_XGB': (tier3_xgb_results, run_tier3_xgb),
        'AGate_Sim': (agate_results, None),  # No re-run capability for hardware
    }

    # =========================================================================
    # Step 3: Apply Approach A and B calibration to each classifier
    # =========================================================================
    print("\n[3] Applying clean calibration methods...")

    calibration_results = {}

    for clf_name, (clf_results, run_fn) in all_classifiers.items():
        if not clf_results:
            continue

        # Filter to valid subjects that have results
        clf_subjects = [s for s in valid_subjects if s in clf_results]

        print(f"\n  {clf_name} ({len(clf_subjects)} subjects):")

        # Approach A: Inner CV (only if we have a re-runnable classifier)
        if run_fn is not None:
            approach_a = approach_a_calibrate(
                clf_results, clf_name, patient_data, clf_subjects, run_fn
            )
            a_auc = np.mean([r['approach_a_auc'] for r in approach_a.values()])
            a_agree = sum(1 for r in approach_a.values() if r['agrees_with_oracle'])
            print(f"    Approach A: cal AUC = {a_auc:.4f}, oracle agreement = {a_agree}/{len(approach_a)}")
        else:
            approach_a = {}
            # For A-Gate, we can't re-run inner CV, just compute oracle
            for subj in clf_subjects:
                raw = clf_results[subj]['raw_auc']
                oracle_pol = 'inverted' if raw < 0.5 else 'standard'
                approach_a[subj] = {
                    'raw_auc': raw,
                    'approach_a_polarity': 'N/A',
                    'approach_a_auc': float('nan'),
                    'oracle_polarity': oracle_pol,
                    'oracle_auc': float(max(raw, 1.0 - raw)),
                    'agrees_with_oracle': False,  # N/A
                }

        # Approach B: Covariance geometry
        approach_b = approach_b_calibrate(clf_results, patient_data, clf_subjects)
        b_auc = np.mean([r['approach_b_auc'] for r in approach_b.values()])
        b_agree = sum(1 for r in approach_b.values() if r['agrees_with_oracle'])
        print(f"    Approach B: cal AUC = {b_auc:.4f}, oracle agreement = {b_agree}/{len(approach_b)}")

        # Oracle
        oracle_auc = np.mean([max(clf_results[s]['raw_auc'], 1.0 - clf_results[s]['raw_auc'])
                              for s in clf_subjects])
        print(f"    Oracle:     cal AUC = {oracle_auc:.4f}")

        calibration_results[clf_name] = {
            'approach_a': approach_a,
            'approach_b': approach_b,
            'overall_oracle_auc': float(oracle_auc),
            'overall_approach_a_auc': float(np.mean([r['approach_a_auc'] for r in approach_a.values()
                                                      if not np.isnan(r['approach_a_auc'])])) if approach_a else None,
            'overall_approach_b_auc': float(b_auc),
            'approach_a_agreement': f"{a_agree}/{len(approach_a)}" if run_fn else "N/A",
            'approach_b_agreement': f"{b_agree}/{len(approach_b)}",
        }

    # =========================================================================
    # Step 4: Print GATE tables
    # =========================================================================
    print("\n" + "=" * 90)
    print("TABLE 1 — Polarity Concordance")
    print("=" * 90)

    header = f"{'Subject':<10} {'Oracle pol.':<14} {'Approach A':<14} {'Approach B':<14} {'Quantum HW':<14}"
    print(header)
    print("-" * 90)

    # Get quantum HW polarity from A-Gate results
    quantum_hw_pol = {}
    for subj in valid_subjects:
        if subj in agate_results:
            raw = agate_results[subj]['raw_auc']
            quantum_hw_pol[subj] = 'inverted' if raw < 0.5 else 'standard'

    # Use Tier 3 LDA as the primary for the concordance table
    for subj in valid_subjects:
        t3_results = calibration_results.get('Tier3_LDA', {})
        a_data = t3_results.get('approach_a', {}).get(subj, {})
        b_data = t3_results.get('approach_b', {}).get(subj, {})

        oracle_pol = a_data.get('oracle_polarity', '?')
        a_pol = a_data.get('approach_a_polarity', '?')
        b_pol = b_data.get('approach_b_polarity', '?')
        hw_pol = quantum_hw_pol.get(subj, '?')

        print(f"{subj:<10} {oracle_pol:<14} {a_pol:<14} {b_pol:<14} {hw_pol:<14}")

    print("\n" + "=" * 90)
    print("TABLE 2 — Calibrated AUC Comparison")
    print("=" * 90)

    header2 = f"{'Classifier':<16} {'Oracle cal.':<16} {'Approach A':<16} {'Approach B':<16}"
    print(header2)
    print("-" * 90)

    for clf_name in ['Tier3_LDA', 'Tier2A_Eigen', 'Tier3_XGB', 'AGate_Sim']:
        cr = calibration_results.get(clf_name, {})
        oracle = cr.get('overall_oracle_auc', float('nan'))
        a_auc = cr.get('overall_approach_a_auc', float('nan'))
        b_auc = cr.get('overall_approach_b_auc', float('nan'))

        oracle_s = f"{oracle:.4f}" if not np.isnan(oracle) else "N/A"
        a_s = f"{a_auc:.4f}" if a_auc is not None and not np.isnan(a_auc) else "N/A"
        b_s = f"{b_auc:.4f}" if not np.isnan(b_auc) else "N/A"

        print(f"{clf_name:<16} {oracle_s:<16} {a_s:<16} {b_s:<16}")

    print("\n" + "=" * 90)
    print("TABLE 3 — Tier 3 XGBoost Per-Subject (Raw)")
    print("=" * 90)

    header3 = f"{'Subject':<10} {'Raw AUC':<12} {'Polarity':<12} {'Inverted in LDA too?':<22}"
    print(header3)
    print("-" * 90)

    for subj in valid_subjects:
        xgb_raw = tier3_xgb_results.get(subj, {}).get('raw_auc', float('nan'))
        xgb_pol = 'inverted' if xgb_raw < 0.5 else 'standard'
        lda_raw = tier3_lda_results.get(subj, {}).get('raw_auc', float('nan'))
        lda_pol = 'inverted' if lda_raw < 0.5 else 'standard'
        both_inverted = 'YES' if xgb_pol == 'inverted' and lda_pol == 'inverted' else 'no'
        if xgb_pol == 'inverted' and lda_pol != 'inverted':
            both_inverted = 'XGB only'
        elif lda_pol == 'inverted' and xgb_pol != 'inverted':
            both_inverted = 'LDA only'

        print(f"{subj:<10} {xgb_raw:<12.4f} {xgb_pol:<12} {both_inverted:<22}")

    # =========================================================================
    # Step 5: Decision on pre-registered prediction
    # =========================================================================
    print("\n" + "=" * 90)
    print("DECISION: PRE-REGISTERED PREDICTION VALIDITY")
    print("=" * 90)

    t3_oracle = calibration_results.get('Tier3_LDA', {}).get('overall_oracle_auc', 0)
    t3_b = calibration_results.get('Tier3_LDA', {}).get('overall_approach_b_auc', 0)

    delta = abs(t3_oracle - t3_b)
    print(f"Tier 3 LDA Oracle calibrated AUC: {t3_oracle:.4f}")
    print(f"Tier 3 LDA Approach B calibrated AUC: {t3_b:.4f}")
    print(f"Delta: {delta:.4f}")

    if t3_b >= t3_oracle - 0.03:
        print("VERDICT: Approach B recovers oracle AUC within ±0.03.")
        print("Pre-registered prediction (0.63-0.70) STANDS.")
        verdict = "PREDICTION_VALID"
    elif t3_b >= 0.60:
        print("VERDICT: Approach B AUC dropped but still ≥ 0.60.")
        print("Pre-registered prediction may need minor revision.")
        verdict = "PREDICTION_MARGINAL"
    else:
        print("VERDICT: Approach B AUC dropped below 0.60.")
        print("Pre-registered prediction range MUST BE REVISED before hardware runs.")
        verdict = "PREDICTION_INVALID"

    # =========================================================================
    # Step 6: Save full report
    # =========================================================================
    report = {
        'prompt': '022',
        'task': 'Polarity Calibration Validity Audit',
        'timestamp': datetime.now().isoformat(),
        'verdict': verdict,
        'subjects': valid_subjects,
        'classifiers': {},
        'calibration_results': {},
        'tier3_xgb_experiment': {},
    }

    # Classifier raw results (without score arrays for compactness)
    for clf_name, (clf_results, _) in all_classifiers.items():
        report['classifiers'][clf_name] = {
            subj: {'raw_auc': r['raw_auc']}
            for subj, r in clf_results.items()
        }

    # Calibration results (strip inner_details and score arrays)
    for clf_name, cr in calibration_results.items():
        clean_a = {}
        for subj, r in cr.get('approach_a', {}).items():
            clean_a[subj] = {k: v for k, v in r.items() if k != 'inner_details'}
        clean_b = {}
        for subj, r in cr.get('approach_b', {}).items():
            clean_b[subj] = r

        report['calibration_results'][clf_name] = {
            'approach_a': clean_a,
            'approach_b': clean_b,
            'overall_oracle_auc': cr.get('overall_oracle_auc'),
            'overall_approach_a_auc': cr.get('overall_approach_a_auc'),
            'overall_approach_b_auc': cr.get('overall_approach_b_auc'),
            'approach_a_agreement': cr.get('approach_a_agreement'),
            'approach_b_agreement': cr.get('approach_b_agreement'),
        }

    # XGBoost experiment
    for subj in valid_subjects:
        xgb_raw = tier3_xgb_results.get(subj, {}).get('raw_auc', None)
        lda_raw = tier3_lda_results.get(subj, {}).get('raw_auc', None)
        report['tier3_xgb_experiment'][subj] = {
            'xgb_raw_auc': xgb_raw,
            'lda_raw_auc': lda_raw,
            'xgb_polarity': 'inverted' if xgb_raw and xgb_raw < 0.5 else 'standard',
            'lda_polarity': 'inverted' if lda_raw and lda_raw < 0.5 else 'standard',
            'both_inverted': (xgb_raw is not None and xgb_raw < 0.5 and
                              lda_raw is not None and lda_raw < 0.5),
        }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'clean_calibration_audit.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to: {output_path}")


if __name__ == '__main__':
    main()
