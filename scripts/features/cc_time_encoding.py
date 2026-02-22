#!/usr/bin/env python3
"""
CC_time (Time-Domain Correlation) Feature Extraction

Implements time-domain correlation features using Pearson correlation
on raw EEG channel data. This is simpler than CC_freq but uses normalized
correlation (Pearson CC) rather than raw covariance.

Note: This differs from existing Tier 2 MAX pooling covariance features
which use raw covariance matrices. CC_time uses Pearson correlation
(normalized), which may capture different information.

Procedure:
1. Compute Pearson correlation directly on raw channel data
2. Extract upper triangle
3. Compute eigenvalues, sort by magnitude descending
4. Return dict with cc_upper, cc_eigenvalues

Reference: A-002 from research prompts
"""

import numpy as np
from typing import Dict


def extract_cc_time(
    data: np.ndarray,
    epsilon: float = 1e-10
) -> Dict[str, np.ndarray]:
    """
    Extract time-domain correlation (CC_time) features.

    Args:
        data: EEG data array of shape (n_channels, n_samples)
        epsilon: Small constant for numerical stability

    Returns:
        Dictionary with keys:
            - 'cc_upper': Upper triangle of CC_time matrix, shape (n_channels*(n_channels-1)/2,)
            - 'cc_eigenvalues': Sorted eigenvalues (descending magnitude), shape (n_channels,)
            - 'cc_matrix': Full correlation matrix, shape (n_channels, n_channels)
    """
    n_channels, n_samples = data.shape

    # Step 1: Compute Pearson correlation directly on time-domain data
    # corrcoef treats each row as a variable, columns as observations
    try:
        CC_time = np.corrcoef(data)  # shape: (n_channels, n_channels)
    except Exception:
        # Fallback: return identity matrix if correlation fails
        CC_time = np.eye(n_channels)

    # Handle NaN values (can occur with constant channels)
    if np.isnan(CC_time).any():
        CC_time = np.nan_to_num(CC_time, nan=0.0)
        np.fill_diagonal(CC_time, 1.0)

    # Step 2: Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices(n_channels, k=1)
    cc_upper = CC_time[triu_indices]

    # Step 3: Compute eigenvalues
    try:
        # Use eigvalsh for symmetric matrices (numerically stable)
        eigenvalues = np.linalg.eigvalsh(CC_time)
        # Sort by magnitude descending
        eigenvalues_sorted = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
    except np.linalg.LinAlgError:
        # Fallback: return ones if eigenvalue decomposition fails
        eigenvalues_sorted = np.ones(n_channels)

    return {
        'cc_upper': cc_upper,
        'cc_eigenvalues': eigenvalues_sorted,
        'cc_matrix': CC_time,
    }


def extract_cov_time(
    data: np.ndarray,
    epsilon: float = 1e-10
) -> Dict[str, np.ndarray]:
    """
    Extract time-domain covariance features (for comparison with CC_time).

    This is equivalent to the existing Tier 2 covariance approach but
    returns eigenvalues in the same format as CC_time for comparison.

    Args:
        data: EEG data array of shape (n_channels, n_samples)
        epsilon: Small constant for numerical stability

    Returns:
        Dictionary with keys:
            - 'cov_upper': Upper triangle of covariance matrix
            - 'cov_eigenvalues': Sorted eigenvalues (descending magnitude)
            - 'cov_matrix': Full covariance matrix
    """
    n_channels, n_samples = data.shape

    # Compute raw covariance
    try:
        cov_matrix = np.cov(data)  # shape: (n_channels, n_channels)
    except Exception:
        cov_matrix = np.eye(n_channels)

    # Handle NaN values
    if np.isnan(cov_matrix).any():
        cov_matrix = np.nan_to_num(cov_matrix, nan=0.0)

    # Extract upper triangle
    triu_indices = np.triu_indices(n_channels, k=1)
    cov_upper = cov_matrix[triu_indices]

    # Compute eigenvalues
    try:
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        eigenvalues_sorted = eigenvalues[np.argsort(np.abs(eigenvalues))[::-1]]
    except np.linalg.LinAlgError:
        eigenvalues_sorted = np.ones(n_channels)

    return {
        'cov_upper': cov_upper,
        'cov_eigenvalues': eigenvalues_sorted,
        'cov_matrix': cov_matrix,
    }


def validate_eigenvalue_sum(eigenvalues: np.ndarray, n_channels: int, tol: float = 1e-6) -> bool:
    """
    Validate that eigenvalues of a correlation matrix sum to n_channels.

    For a correlation matrix (diagonal = 1), trace = n_channels,
    and trace equals sum of eigenvalues.

    Args:
        eigenvalues: Array of eigenvalues
        n_channels: Number of channels
        tol: Tolerance for comparison

    Returns:
        True if eigenvalues sum to n_channels within tolerance
    """
    return np.abs(np.sum(eigenvalues) - n_channels) < tol


def compare_cc_vs_cov(data: np.ndarray) -> Dict[str, float]:
    """
    Compare CC_time (Pearson correlation) vs raw covariance features.

    This helps verify whether the normalization matters for classification.

    Args:
        data: EEG data array of shape (n_channels, n_samples)

    Returns:
        Dictionary with comparison metrics
    """
    cc_result = extract_cc_time(data)
    cov_result = extract_cov_time(data)

    # Correlation between CC eigenvalues and cov eigenvalues
    # (normalized to compare relative magnitudes)
    cc_eig_norm = cc_result['cc_eigenvalues'] / np.sum(cc_result['cc_eigenvalues'])
    cov_eig_norm = cov_result['cov_eigenvalues'] / np.sum(cov_result['cov_eigenvalues'])

    eigen_correlation = np.corrcoef(cc_eig_norm, cov_eig_norm)[0, 1]

    # Correlation between upper triangle values
    upper_correlation = np.corrcoef(cc_result['cc_upper'], cov_result['cov_upper'])[0, 1]

    return {
        'eigen_correlation': eigen_correlation,
        'upper_correlation': upper_correlation,
        'cc_max_eigen': cc_result['cc_eigenvalues'][0],
        'cov_max_eigen': cov_result['cov_eigenvalues'][0],
        'cc_trace': np.sum(cc_result['cc_eigenvalues']),
        'cov_trace': np.sum(cov_result['cov_eigenvalues']),
    }


# =============================================================================
# Validation / Test
# =============================================================================

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add parent directories to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("=" * 60)
    print("CC_time Feature Extraction Validation")
    print("=" * 60)

    # Try to load real CHB-MIT data
    try:
        from sagemaker.train_chbmit import (
            extract_segments_for_subject,
            read_edf_segment,
            preprocess_eeg,
            FS,
        )

        DATA_ROOT = Path("H:/Data/PythonDNU/EEG/chbmit")
        CHANNELS = ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3", "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]

        if DATA_ROOT.exists():
            subject_dir = DATA_ROOT / "chb01"

            print(f"\nLoading segments from {subject_dir}...")
            segments = extract_segments_for_subject(subject_dir)

            # Find one ictal and one interictal segment
            ictal_seg = None
            interictal_seg = None

            for seg in segments:
                if seg.label == 'ictal' and ictal_seg is None:
                    ictal_seg = seg
                elif seg.label == 'interictal' and interictal_seg is None:
                    interictal_seg = seg
                if ictal_seg and interictal_seg:
                    break

            if ictal_seg and interictal_seg:
                print(f"Found ictal: {ictal_seg.source_file}")
                print(f"Found interictal: {interictal_seg.source_file}")

                # Load ictal data
                ictal_data, fs = read_edf_segment(
                    str(subject_dir / ictal_seg.source_file),
                    ictal_seg.start_sec,
                    ictal_seg.end_sec,
                    target_channels=CHANNELS
                )
                ictal_data = preprocess_eeg(ictal_data, fs=int(fs))

                # Load interictal data
                inter_data, fs = read_edf_segment(
                    str(subject_dir / interictal_seg.source_file),
                    interictal_seg.start_sec,
                    interictal_seg.end_sec,
                    target_channels=CHANNELS
                )
                inter_data = preprocess_eeg(inter_data, fs=int(fs))

                print(f"\n--- ICTAL Segment (chb01) ---")
                print(f"Data shape: {ictal_data.shape}")

                result_ictal = extract_cc_time(ictal_data)
                print(f"\nCC_time Results:")
                print(f"  cc_upper shape: {result_ictal['cc_upper'].shape}")
                print(f"  cc_upper range: [{result_ictal['cc_upper'].min():.4f}, {result_ictal['cc_upper'].max():.4f}]")
                print(f"  cc_eigenvalues: {result_ictal['cc_eigenvalues']}")
                print(f"  Eigenvalue sum: {np.sum(result_ictal['cc_eigenvalues']):.6f} (should be 8)")
                print(f"  Validation: {'PASSED' if validate_eigenvalue_sum(result_ictal['cc_eigenvalues'], 8) else 'FAILED'}")

                print(f"\n--- INTERICTAL Segment (chb01) ---")
                print(f"Data shape: {inter_data.shape}")

                result_inter = extract_cc_time(inter_data)
                print(f"\nCC_time Results:")
                print(f"  cc_upper shape: {result_inter['cc_upper'].shape}")
                print(f"  cc_upper range: [{result_inter['cc_upper'].min():.4f}, {result_inter['cc_upper'].max():.4f}]")
                print(f"  cc_eigenvalues: {result_inter['cc_eigenvalues']}")
                print(f"  Eigenvalue sum: {np.sum(result_inter['cc_eigenvalues']):.6f} (should be 8)")
                print(f"  Validation: {'PASSED' if validate_eigenvalue_sum(result_inter['cc_eigenvalues'], 8) else 'FAILED'}")

                # Compare CC vs Covariance
                print(f"\n--- CC_time vs Covariance Comparison (Ictal) ---")
                comparison = compare_cc_vs_cov(ictal_data)
                print(f"  Eigenvalue correlation: {comparison['eigen_correlation']:.4f}")
                print(f"  Upper triangle correlation: {comparison['upper_correlation']:.4f}")
                print(f"  CC max eigenvalue: {comparison['cc_max_eigen']:.4f}")
                print(f"  Cov max eigenvalue: {comparison['cov_max_eigen']:.4f}")
                print(f"  CC trace: {comparison['cc_trace']:.4f}")
                print(f"  Cov trace: {comparison['cov_trace']:.4f}")

                print(f"\n--- CC_time vs Covariance Comparison (Interictal) ---")
                comparison = compare_cc_vs_cov(inter_data)
                print(f"  Eigenvalue correlation: {comparison['eigen_correlation']:.4f}")
                print(f"  Upper triangle correlation: {comparison['upper_correlation']:.4f}")

            else:
                print("Could not find both segment types")
        else:
            raise FileNotFoundError("Data directory not found")

    except Exception as e:
        print(f"\nUsing synthetic data: {e}")

        # Synthetic data test
        np.random.seed(42)
        n_channels = 8
        n_samples = 256 * 10
        fs = 256

        t = np.arange(n_samples) / fs

        # Ictal: synchronized activity
        ictal_data = np.array([
            np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 0.5))
            + 0.3 * np.random.randn(n_samples)
            for _ in range(n_channels)
        ])

        # Interictal: independent random
        inter_data = np.random.randn(n_channels, n_samples)

        print(f"\n--- Synthetic ICTAL ---")
        result = extract_cc_time(ictal_data)
        print(f"cc_upper shape: {result['cc_upper'].shape}")
        print(f"cc_eigenvalues: {result['cc_eigenvalues']}")
        print(f"Eigenvalue sum: {np.sum(result['cc_eigenvalues']):.6f}")
        print(f"Validation: {'PASSED' if validate_eigenvalue_sum(result['cc_eigenvalues'], n_channels) else 'FAILED'}")

        print(f"\n--- Synthetic INTERICTAL ---")
        result = extract_cc_time(inter_data)
        print(f"cc_upper shape: {result['cc_upper'].shape}")
        print(f"cc_eigenvalues: {result['cc_eigenvalues']}")
        print(f"Eigenvalue sum: {np.sum(result['cc_eigenvalues']):.6f}")
        print(f"Validation: {'PASSED' if validate_eigenvalue_sum(result['cc_eigenvalues'], n_channels) else 'FAILED'}")

        print(f"\n--- CC vs Covariance Comparison ---")
        comp = compare_cc_vs_cov(ictal_data)
        print(f"Eigenvalue correlation: {comp['eigen_correlation']:.4f}")
        print(f"Upper triangle correlation: {comp['upper_correlation']:.4f}")

    print("\n" + "=" * 60)
    print("Validation complete")
    print("=" * 60)
