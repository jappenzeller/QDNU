#!/usr/bin/env python3
"""
================================================================================
PROMPT 030 — Apply Dual Calibration Reporting to Hardware Results
================================================================================

Applies dual-reporting calibration (raw + oracle-calibrated, honestly labeled)
to all hardware results from PROMPT 026-028.

Output: results/calibration/hardware_calibration_report.json
================================================================================
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']

# Tier 2A clean-validated comparator from PROMPT 022
TIER2A_CLEAN_AUC = 0.6831
TIER2A_ORACLE_AUC = 0.6892


def load_results():
    """Load per-subject results from all three hardware runs."""
    configs = {}

    # CH8-20s (PROMPT 026)
    path = PROJECT_ROOT / "results" / "window_analysis" / "quantum_20s_hardware.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        configs['CH8_20s'] = {
            subj: d.get('raw_auc', d.get('auc'))
            for subj, d in data['per_subject'].items()
        }

    # CH12 and CH16 (PROMPT 027/028)
    path = PROJECT_ROOT / "results" / "projection" / "hardware_scaling.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        for ch_key, config_name in [('ch12', 'CH12'), ('ch16', 'CH16')]:
            if ch_key in data and 'per_subject' in data[ch_key]:
                configs[config_name] = {
                    subj: d.get('raw_auc', d.get('auc'))
                    for subj, d in data[ch_key]['per_subject'].items()
                }

    # CH8-1.95s (PROMPT 006 baseline)
    path = PROJECT_ROOT / "results" / "hardware_validation" / "ch8_heron_loso_results.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        configs['CH8_1.95s'] = {
            subj: d['auc']
            for subj, d in data['ch8']['per_subject'].items()
        }

    return configs


def main():
    print("=" * 100)
    print("PROMPT 030 -- Dual Calibration Reporting on Hardware Results")
    print("=" * 100)

    configs = load_results()
    print(f"Configs loaded: {list(configs.keys())}\n")

    report = {
        'prompt': '030',
        'task': 'Dual Calibration Reporting',
        'timestamp': datetime.now().isoformat(),
        'calibration_method': 'oracle (max(raw, 1-raw), labeled as assumes-known-polarity)',
        'tier2a_clean_reference': TIER2A_CLEAN_AUC,
        'tier2a_oracle_reference': TIER2A_ORACLE_AUC,
        'configs': {},
    }

    # Per-config results
    all_config_rows = []

    for config_name in ['CH8_1.95s', 'CH8_20s', 'CH12', 'CH16']:
        if config_name not in configs:
            continue

        per_subj = configs[config_name]
        raw_aucs = []
        oracle_aucs = []
        n_inverted = 0
        per_subject_detail = {}

        chb14_inv = False
        chb21_inv = False
        chb03_inv = False

        for subj in PRIORITY_SUBJECTS:
            if subj not in per_subj:
                continue
            raw = per_subj[subj]
            oracle = max(raw, 1.0 - raw)
            polarity = 'inverted' if raw < 0.5 else ('chance' if abs(raw - 0.5) < 0.001 else 'standard')

            if polarity == 'inverted':
                n_inverted += 1
            if subj == 'chb14' and polarity == 'inverted':
                chb14_inv = True
            if subj == 'chb21' and polarity == 'inverted':
                chb21_inv = True
            if subj == 'chb03' and polarity == 'inverted':
                chb03_inv = True

            raw_aucs.append(raw)
            oracle_aucs.append(oracle)

            per_subject_detail[subj] = {
                'raw_auc': round(raw, 4),
                'oracle_calibrated_auc': round(oracle, 4),
                'polarity': polarity,
            }

        overall_raw = float(np.mean(raw_aucs)) if raw_aucs else 0.5
        overall_oracle = float(np.mean(oracle_aucs)) if oracle_aucs else 0.5

        config_result = {
            'overall_raw_auc': round(overall_raw, 4),
            'overall_oracle_calibrated_auc': round(overall_oracle, 4),
            'n_inverted': n_inverted,
            'chb14_inverted': chb14_inv,
            'chb21_inverted': chb21_inv,
            'chb03_inverted': chb03_inv,
            'per_subject': per_subject_detail,
        }

        report['configs'][config_name] = config_result

        all_config_rows.append({
            'name': config_name,
            'raw': overall_raw,
            'oracle': overall_oracle,
            'n_inv': n_inverted,
            'chb14': chb14_inv,
            'chb21': chb21_inv,
            'chb03': chb03_inv,
        })

    # Print gate table
    print("=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = (f"{'Config':<12} {'Raw AUC':<10} {'Oracle-cal.':<13} {'N_inv':<8} "
              f"{'chb14 inv?':<12} {'chb21 inv?':<12} {'chb03 inv?':<12}")
    print(header)
    print("-" * 100)

    for r in all_config_rows:
        print(f"{r['name']:<12} {r['raw']:<10.4f} {r['oracle']:<13.4f} {r['n_inv']:<8} "
              f"{'YES' if r['chb14'] else 'no':<12} "
              f"{'YES' if r['chb21'] else 'no':<12} "
              f"{'YES' if r['chb03'] else 'no':<12}")

    print(f"{'Tier2A clean':<12} {'--':<10} {TIER2A_CLEAN_AUC:<13.4f} {'--':<8} "
          f"{'--':<12} {'--':<12} {'--':<12}")

    # Per-subject detail table
    print("\n" + "=" * 100)
    print("PER-SUBJECT RAW AUC ACROSS ALL CONFIGS")
    print("=" * 100)

    header2 = f"{'Subject':<10}"
    for cn in ['CH8_1.95s', 'CH8_20s', 'CH12', 'CH16']:
        header2 += f" {cn:<12}"
    print(header2)
    print("-" * 100)

    for subj in PRIORITY_SUBJECTS:
        row = f"{subj:<10}"
        for cn in ['CH8_1.95s', 'CH8_20s', 'CH12', 'CH16']:
            cfg = report['configs'].get(cn, {})
            ps = cfg.get('per_subject', {}).get(subj, {})
            raw = ps.get('raw_auc')
            if raw is not None:
                pol = ps.get('polarity', '?')[0]
                row += f" {raw:.3f}({pol}){'':<4}"
            else:
                row += f" {'--':<12}"
        print(row)

    # Save
    out_dir = PROJECT_ROOT / "results" / "calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hardware_calibration_report.json"
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {out_path}")


if __name__ == '__main__':
    main()
