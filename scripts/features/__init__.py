"""
Feature Extraction Module

This module contains various EEG feature extraction implementations
for seizure prediction benchmarks.

Modules:
- cc_freq_encoding: Frequency-domain correlation features (CC_freq)
- cc_time_encoding: Time-domain correlation features (CC_time)
- log_fft_encoding: Log10 FFT magnitude features
"""

from .cc_freq_encoding import extract_cc_freq, validate_eigenvalue_sum
from .cc_time_encoding import extract_cc_time, extract_cov_time, compare_cc_vs_cov
from .log_fft_encoding import extract_log_fft, extract_log_fft_simple, extract_log_fft_per_band

__all__ = [
    'extract_cc_freq',
    'extract_cc_time',
    'extract_cov_time',
    'compare_cc_vs_cov',
    'validate_eigenvalue_sum',
    'extract_log_fft',
    'extract_log_fft_simple',
    'extract_log_fft_per_band',
]
