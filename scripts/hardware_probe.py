#!/usr/bin/env python3
"""
================================================================================
PROMPT 025 — Hardware Probe: Single Subject (chb01) on ibm_torino
================================================================================

Minimal credit burn: runs chb01 ONLY through all three configurations on actual
ibm_torino hardware. Verifies connectivity, job submission, and result retrieval.

Output: results/preflight/hardware_probe_chb01.json
================================================================================
"""

import sys
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'QA1'))

from scipy.signal import hilbert, butter, filtfilt

from sagemaker.train_chbmit import (
    FS, WINDOW_SEC, normalize_channel_label,
    extract_segments_for_subject as _extract_segments_for_subject,
    EEGSegment,
)
from QA1.multichannel_circuit import create_multichannel_circuit

import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("H:/Data/PythonDNU/EEG/chbmit")
OUTPUT_DIR = PROJECT_ROOT / "results" / "preflight"
CLASSICAL_SCALING = PROJECT_ROOT / "results" / "scaling" / "classical_channel_scaling.json"

SHOTS = 1024
BAND = (4, 13)
SUBJECT = 'chb01'

WINDOW_SECONDS_20S = 20.0
WINDOW_SAMPLES_20S = int(WINDOW_SECONDS_20S * FS)

CONFIGS = {
    'CH8_20s': {'channel_key': 'ch8', 'use_20s': True, 'expected_qubits': 17},
    'CH12': {'channel_key': 'ch12', 'use_20s': False, 'expected_qubits': 25},
    'CH16': {'channel_key': 'ch16', 'use_20s': False, 'expected_qubits': 33},
}


# =============================================================================
# PLV ENCODING (verified by PROMPT 023)
# =============================================================================

def extract_plv_params(eeg_segment, fs=256.0, band=(4, 13)):
    n_channels, n_samples = eeg_segment.shape
    nyq = fs / 2
    low, high = band[0] / nyq, min(band[1] / nyq, 0.99)
    try:
        b_filt, a_filt = butter(4, [low, high], btype='band')
    except ValueError:
        return [(0.5, np.pi, 0.5)] * n_channels

    phases = np.zeros((n_channels, n_samples))
    envelopes = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        try:
            filtered = filtfilt(b_filt, a_filt, eeg_segment[ch].astype(np.float64))
            analytic = hilbert(filtered)
            phases[ch] = np.angle(analytic)
            envelopes[ch] = np.abs(analytic)
        except Exception:
            envelopes[ch] = np.ones(n_samples)

    mean_analytic = np.mean(envelopes * np.exp(1j * phases), axis=0)
    global_phase = np.angle(mean_analytic)
    all_amps = [np.mean(envelopes[ch]) for ch in range(n_channels)]
    amp_min, amp_max = min(all_amps), max(all_amps)
    amp_range = amp_max - amp_min if amp_max > amp_min else 1.0

    params_list = []
    for ch in range(n_channels):
        a_norm = 0.05 + 0.9 * (all_amps[ch] - amp_min) / amp_range
        mean_sin = np.mean(np.sin(phases[ch]))
        mean_cos = np.mean(np.cos(phases[ch]))
        b_phase = np.arctan2(mean_sin, mean_cos) % (2 * np.pi)
        phase_diff = phases[ch] - global_phase
        plv = np.abs(np.mean(np.exp(1j * phase_diff)))
        c_val = np.clip(plv, 0.05, 0.95)
        params_list.append((a_norm, b_phase, c_val))
    return params_list


# =============================================================================
# DATA LOADING
# =============================================================================

def load_channels(config_key):
    if CLASSICAL_SCALING.exists():
        with open(CLASSICAL_SCALING) as f:
            data = json.load(f)
        if config_key in data:
            return data[config_key]['channels']
    defaults = {
        'ch8': ["FP1-F7","F7-T7","FP1-F3","F3-C3","FP2-F8","F8-T8","FP2-F4","F4-C4"],
        'ch12': ["FP1-F7","F7-T7","FP1-F3","F3-C3","FP2-F8","F8-T8","FP2-F4","F4-C4",
                 "C3-P3","C4-P4","CZ-PZ","FZ-CZ"],
        'ch16': ["FP1-F7","F7-T7","FP1-F3","F3-C3","FP2-F8","F8-T8","FP2-F4","F4-C4",
                 "C3-P3","C4-P4","CZ-PZ","FZ-CZ","P3-O1","P4-O2","P7-O1","P8-O2"],
    }
    return defaults.get(config_key, defaults['ch8'])


def load_eeg_segment(seg, subject_dir, channels):
    try:
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
                    channel_indices.append(file_channels_norm.index(ch_norm))
                else:
                    return None
            fs = f.getSampleFrequency(0)
            start_sample = int(seg.start_sec * fs)
            n_samples = int(WINDOW_SEC * fs)
            data = np.zeros((len(channels), n_samples))
            for i, ch_idx in enumerate(channel_indices):
                signal = f.readSignal(ch_idx)
                end_sample = min(start_sample + n_samples, len(signal))
                if end_sample - start_sample < n_samples:
                    return None
                data[i] = signal[start_sample:end_sample]
            return data
    except Exception:
        return None


# =============================================================================
# MAIN
# =============================================================================

def main():
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    print("=" * 70)
    print("PROMPT 025 — Hardware Probe: chb01 on ibm_torino")
    print("=" * 70)
    print(f"Subject: {SUBJECT}")
    print(f"Shots: {SHOTS}")
    print()

    # Connect to IBM
    print("[1] Connecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.backend('ibm_torino')
    print(f"    Backend: {backend.name} ({backend.num_qubits} qubits)")

    # Load chb01 segments
    print("\n[2] Loading chb01 EEG data...")
    subject_dir = DATA_DIR / SUBJECT
    segments = _extract_segments_for_subject(subject_dir)

    # Find first ictal and first interictal
    ictal_seg = None
    interictal_seg = None
    for seg in segments:
        if seg.label == 'ictal' and ictal_seg is None:
            ictal_seg = seg
        elif seg.label == 'interictal' and interictal_seg is None:
            interictal_seg = seg
        if ictal_seg and interictal_seg:
            break

    print(f"    Ictal: {ictal_seg.source_file} [{ictal_seg.start_sec}s-{ictal_seg.end_sec}s]")
    print(f"    Interictal: {interictal_seg.source_file} [{interictal_seg.start_sec}s-{interictal_seg.end_sec}s]")

    # Run each config
    results = {}

    for config_name, config in CONFIGS.items():
        print(f"\n[3] Config: {config_name}")
        channels = load_channels(config['channel_key'])
        n_channels = len(channels)
        n_qubits = 2 * n_channels + 1
        use_20s = config['use_20s']

        print(f"    Channels: {n_channels}, Qubits: {n_qubits}")

        # Load EEG for both segments
        ictal_data = load_eeg_segment(ictal_seg, subject_dir, channels)
        interictal_data = load_eeg_segment(interictal_seg, subject_dir, channels)

        if ictal_data is None or interictal_data is None:
            print(f"    FAIL: Could not load EEG data for {config_name}")
            results[config_name] = {'error': 'EEG load failed'}
            continue

        # Apply window if needed
        if use_20s:
            ictal_data = ictal_data[:, :WINDOW_SAMPLES_20S]
            interictal_data = interictal_data[:, :WINDOW_SAMPLES_20S]

        # Extract PLV params
        ictal_params = extract_plv_params(ictal_data, fs=FS, band=BAND)
        interictal_params = extract_plv_params(interictal_data, fs=FS, band=BAND)

        print(f"    Ictal params ch0: a={ictal_params[0][0]:.4f}, b={ictal_params[0][1]:.4f}, c={ictal_params[0][2]:.4f}")

        # Build circuits
        ictal_circuit = create_multichannel_circuit(ictal_params)
        ictal_circuit.measure_all()
        interictal_circuit = create_multichannel_circuit(interictal_params)
        interictal_circuit.measure_all()

        print(f"    Circuits built: {ictal_circuit.num_qubits} qubits each")

        # Transpile
        print(f"    Transpiling (optimization_level=3)...")
        pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
        transpiled = pm.run([ictal_circuit, interictal_circuit])

        ops = transpiled[0].count_ops()
        two_q = sum(ops.get(g, 0) for g in ['cx', 'cz', 'ecr'])
        print(f"    Transpiled: depth={transpiled[0].depth()}, 2Q gates={two_q}")

        # Submit to hardware
        print(f"    Submitting to {backend.name} ({SHOTS} shots)...")
        t_submit = time.time()

        sampler = SamplerV2(mode=backend)
        job = sampler.run(transpiled, shots=SHOTS)
        job_id = job.job_id()
        print(f"    Job ID: {job_id}")

        # Wait for results
        print(f"    Waiting for results...")
        result = job.result()
        t_done = time.time()

        # Extract counts
        counts_list = []
        for i in range(2):
            pub_result = result[i]
            try:
                counts = pub_result.data.meas.get_counts()
            except:
                try:
                    counts = pub_result.data.c.get_counts()
                except:
                    counts = {}
            counts_list.append(dict(counts))

        ictal_counts = counts_list[0]
        interictal_counts = counts_list[1]

        # Sanity checks
        ictal_total = sum(ictal_counts.values())
        interictal_total = sum(interictal_counts.values())

        # Bitstring length check
        if ictal_counts:
            bitstring_len = len(next(iter(ictal_counts.keys())))
        else:
            bitstring_len = 0

        # Top 5 bitstrings
        top5_ictal = sorted(ictal_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_interictal = sorted(interictal_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Compute score (Hellinger fidelity difference)
        def hellinger_fid(p, q):
            tp = sum(p.values())
            tq = sum(q.values())
            if tp == 0 or tq == 0:
                return 0.0
            all_keys = set(p.keys()) | set(q.keys())
            s = sum(np.sqrt(p.get(k, 0) / tp * q.get(k, 0) / tq) for k in all_keys)
            return s ** 2

        # Self-fidelity as sanity check
        fid_same = hellinger_fid(ictal_counts, ictal_counts)
        fid_cross = hellinger_fid(ictal_counts, interictal_counts)
        score = fid_same - fid_cross  # Positive if ictal is self-similar

        wall_time = t_done - t_submit

        print(f"\n    --- {config_name} Results ---")
        print(f"    Job ID: {job_id}")
        print(f"    Bitstring length: {bitstring_len} (expected {n_qubits})")
        print(f"    Ictal total counts: {ictal_total} (expected {SHOTS})")
        print(f"    Interictal total counts: {interictal_total} (expected {SHOTS})")
        print(f"    Score (fid_self - fid_cross): {score:.6f}")
        print(f"    Wall time: {wall_time:.1f}s")
        print(f"    Top 5 ictal bitstrings:")
        for bs, cnt in top5_ictal:
            print(f"      {bs}: {cnt}")

        config_result = {
            'job_id': job_id,
            'n_qubits': n_qubits,
            'n_channels': n_channels,
            'bitstring_length': bitstring_len,
            'bitstring_length_correct': bitstring_len == n_qubits,
            'ictal_total_counts': ictal_total,
            'interictal_total_counts': interictal_total,
            'counts_exact': ictal_total == SHOTS and interictal_total == SHOTS,
            'score': float(score),
            'score_is_real': not np.isnan(score),
            'wall_time_sec': round(wall_time, 1),
            'transpiled_depth': transpiled[0].depth(),
            'transpiled_2q_gates': two_q,
            'top5_ictal': top5_ictal,
            'top5_interictal': top5_interictal,
            'ictal_counts': ictal_counts,
            'interictal_counts': interictal_counts,
        }

        # Validation
        checks_passed = (
            bitstring_len == n_qubits and
            ictal_total == SHOTS and
            interictal_total == SHOTS and
            not np.isnan(score)
        )
        config_result['checks_passed'] = checks_passed
        print(f"    All checks passed: {checks_passed}")

        results[config_name] = config_result

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = True
    for config_name, r in results.items():
        if 'error' in r:
            print(f"  {config_name}: ERROR - {r['error']}")
            all_pass = False
        else:
            status = "PASS" if r['checks_passed'] else "FAIL"
            if not r['checks_passed']:
                all_pass = False
            print(f"  {config_name}: {status} | Job={r['job_id']} | "
                  f"bits={r['bitstring_length']}/{r['n_qubits']} | "
                  f"counts={r['ictal_total_counts']}/{SHOTS} | "
                  f"2Q={r['transpiled_2q_gates']} | "
                  f"time={r['wall_time_sec']}s")

    overall = "PASS" if all_pass else "FAIL"
    print(f"\nOverall: {overall}")

    # Save report
    report = {
        'prompt': '025',
        'task': 'Hardware Probe: chb01 on ibm_torino',
        'timestamp': datetime.now().isoformat(),
        'status': overall,
        'backend': backend.name,
        'subject': SUBJECT,
        'shots': SHOTS,
        'configs': {k: {kk: vv for kk, vv in v.items()
                        if kk not in ('ictal_counts', 'interictal_counts')}
                    for k, v in results.items()},
        'raw_counts': {k: {'ictal': v.get('ictal_counts', {}),
                           'interictal': v.get('interictal_counts', {})}
                       for k, v in results.items() if 'error' not in v},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'hardware_probe_chb01.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to: {output_path}")


if __name__ == '__main__':
    main()
