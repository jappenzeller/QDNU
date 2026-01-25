"""
EEG Data Loading and Processing

Handles loading and preprocessing of EEG data for QDNU.
"""

from .eeg_loader import (
    load_for_qdnu,
    list_available_subjects,
    load_segment,
    load_subject_data,
    get_eeg_windows,
)

__all__ = [
    'load_for_qdnu',
    'list_available_subjects',
    'load_segment',
    'load_subject_data',
    'get_eeg_windows',
]
