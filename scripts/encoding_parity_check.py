#!/usr/bin/env python3
"""
================================================================================
PROMPT 023 - Pre-Flight: Encoding Parity Check
================================================================================

Verifies that extract_plv_params() produces identical (a, b, c) parameter tuples
across all four hardware scripts (006, 013, 014, 017) for the same input data.

Output: results/preflight/encoding_parity_report.json
================================================================================
"""

import sys
import json
import importlib.util
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
OUTPUT_DIR = PROJECT_ROOT / "results" / "preflight"
DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'QA1'))

from sagemaker.train_chbmit import (
    FS,
    WINDOW_SEC,
    normalize_channel_label,
    extract_segments_for_subject as _extract_segments_for_subject,
    EEGSegment,
)

# ============================================================================
# Script definitions: name -> (file, function_name)
# ============================================================================

SCRIPTS = {
    "PROMPT_006": SCRIPTS_DIR / "quantum_heron_validation.py",
    "PROMPT_013": SCRIPTS_DIR / "hardware_scaling_ch12_plv.py",
    "PROMPT_014": SCRIPTS_DIR / "hardware_scaling_ch16_plv.py",
    "PROMPT_017": SCRIPTS_DIR / "quantum_20s_hardware.py",
}


def load_extract_plv_params(script_path: Path, module_name: str):
    """Dynamically load extract_plv_params from a script."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    mod = importlib.util.module_from_spec(spec)
    # Prevent the module from executing its main block
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod.extract_plv_params


def extract_segments_for_subject(subject_id: str):
    """Load segments for a subject."""
    subject_dir = DATA_DIR / subject_id
    return _extract_segments_for_subject(subject_dir)


def load_eeg_segment(seg: EEGSegment, subject_dir: Path, channels: List[str]):
    """Load EEG data for a segment."""
    import pyedflib

    edf_path = subject_dir / seg.source_file
    if not edf_path.exists():
        return None

    with pyedflib.EdfReader(str(edf_path)) as f:
        n_signals = f.signals_in_file
        file_channels = [f.getLabel(i) for i in range(n_signals)]
        file_channels_norm = [normalize_channel_label(ch) for ch in file_channels]

        channel_indices = []
        for ch in channels:
            ch_norm = normalize_channel_label(ch)
            if ch_norm in file_channels_norm:
                idx = file_channels_norm.index(ch_norm)
                channel_indices.append(idx)
            else:
                return None

        fs = f.getSampleFrequency(0)
        start_sample = int(seg.start_sec * fs)
        n_samples = int(WINDOW_SEC * fs)

        data = np.zeros((len(channels), n_samples))
        for i, ch_idx in enumerate(channel_indices):
            signal = f.readSignal(ch_idx)
            end_sample = min(start_sample + n_samples, len(signal))
            actual_samples = end_sample - start_sample
            if actual_samples < n_samples:
                return None
            data[i] = signal[start_sample:end_sample]

        return data


def main():
    print("=" * 70)
    print("PROMPT 023 - Encoding Parity Check")
    print("=" * 70)

    # Step 1: Load all four extract_plv_params functions
    print("\n[1] Loading extract_plv_params from all four scripts...")
    funcs = {}
    for name, path in SCRIPTS.items():
        if not path.exists():
            print(f"  FAIL: {name} script not found at {path}")
            return
        funcs[name] = load_extract_plv_params(path, f"_parity_{name}")
        print(f"  Loaded {name}: {path.name}")

    # Step 2: Load chb01 EEG data
    print("\n[2] Loading chb01 EEG segments...")
    subject_id = "chb01"
    subject_dir = DATA_DIR / subject_id
    segments = extract_segments_for_subject(subject_id)

    # Use CH8 channels (common across all scripts for parity test)
    # The function itself is channel-count agnostic — it takes whatever shape is passed
    channels_ch8 = ["FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
                    "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"]

    # Find first ictal and first interictal segment
    ictal_seg = None
    interictal_seg = None
    for seg in segments:
        if seg.label == 'ictal' and ictal_seg is None:
            ictal_seg = seg
        elif seg.label == 'interictal' and interictal_seg is None:
            interictal_seg = seg
        if ictal_seg and interictal_seg:
            break

    if ictal_seg is None or interictal_seg is None:
        print("  FAIL: Could not find both ictal and interictal segments for chb01")
        return

    print(f"  Ictal segment: {ictal_seg.source_file} [{ictal_seg.start_sec}s - {ictal_seg.end_sec}s]")
    print(f"  Interictal segment: {interictal_seg.source_file} [{interictal_seg.start_sec}s - {interictal_seg.end_sec}s]")

    # Load EEG data
    ictal_data = load_eeg_segment(ictal_seg, subject_dir, channels_ch8)
    interictal_data = load_eeg_segment(interictal_seg, subject_dir, channels_ch8)

    if ictal_data is None or interictal_data is None:
        print("  FAIL: Could not load EEG data for one or both segments")
        return

    print(f"  Ictal shape: {ictal_data.shape}")
    print(f"  Interictal shape: {interictal_data.shape}")

    # Step 3: Run extract_plv_params from each script on both segments
    print("\n[3] Running extract_plv_params from all four scripts...")
    results: Dict[str, Dict[str, Any]] = {}
    all_pass = True

    for segment_name, eeg_data in [("ictal", ictal_data), ("interictal", interictal_data)]:
        print(f"\n  --- {segment_name.upper()} segment ---")
        segment_params = {}

        for script_name, func in funcs.items():
            params = func(eeg_data, fs=256.0, band=(4, 13))
            segment_params[script_name] = params
            # Print channel 0 for visual check
            a, b, c = params[0]
            print(f"  {script_name}: ch0 = (a={a:.10f}, b={b:.10f}, c={c:.10f})")

        # Step 4: Check exact equality across all scripts
        reference_name = "PROMPT_006"
        ref_params = segment_params[reference_name]

        for script_name in ["PROMPT_013", "PROMPT_014", "PROMPT_017"]:
            test_params = segment_params[script_name]
            for ch_idx in range(len(ref_params)):
                ref_a, ref_b, ref_c = ref_params[ch_idx]
                test_a, test_b, test_c = test_params[ch_idx]

                if ref_a != test_a or ref_b != test_b or ref_c != test_c:
                    all_pass = False
                    print(f"  MISMATCH: {script_name} ch{ch_idx} vs {reference_name}")
                    print(f"    a: {ref_a} vs {test_a} (delta={abs(ref_a - test_a):.2e})")
                    print(f"    b: {ref_b} vs {test_b} (delta={abs(ref_b - test_b):.2e})")
                    print(f"    c: {ref_c} vs {test_c} (delta={abs(ref_c - test_c):.2e})")

        results[segment_name] = {
            "shape": list(eeg_data.shape),
            "params_per_script": {
                name: [(float(a), float(b), float(c)) for a, b, c in params]
                for name, params in segment_params.items()
            }
        }

    # Step 5: Build report
    status = "PASS" if all_pass else "FAIL"
    print(f"\n{'=' * 70}")
    print(f"RESULT: {status}")
    print(f"{'=' * 70}")

    # Print gate values for channel 0 ictal (for visual verification)
    ch0_ictal = results["ictal"]["params_per_script"]["PROMPT_006"][0]
    a, b, c = ch0_ictal
    print(f"\nGATE CHECK - chb01 ictal segment 1, channel 0:")
    print(f"  a = {a:.6f}  (expected range: 0.05-0.95)")
    print(f"  b = {b:.6f}  (expected range: 0-{2*np.pi:.4f})")
    print(f"  c = {c:.6f}  (expected range: 0.05-0.95)")
    print(f"  a in [0.05, 0.95]: {0.05 <= a <= 0.95}")
    print(f"  b in [0, 2pi]:     {0 <= b <= 2*np.pi}")
    print(f"  c in [0.05, 0.95]: {0.05 <= c <= 0.95}")

    report = {
        "prompt": "023",
        "task": "Encoding Parity Check",
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "subject": "chb01",
        "scripts_compared": list(SCRIPTS.keys()),
        "script_files": {k: str(v) for k, v in SCRIPTS.items()},
        "segments": results,
        "channel_0_ictal_gate_check": {
            "a": a,
            "b": b,
            "c": c,
            "a_in_range": bool(0.05 <= a <= 0.95),
            "b_in_range": bool(0 <= b <= 2 * np.pi),
            "c_in_range": bool(0.05 <= c <= 0.95),
        },
        "all_scripts_match": all_pass,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "encoding_parity_report.json"
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
