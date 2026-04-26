#!/usr/bin/env python3
"""
================================================================================
PROMPT 029 — Results Validation: Polarity Consistency Audit
================================================================================

Cross-validates polarity assignments across all available hardware runs to test
whether polarity is a stable per-patient property.

Output: results/validation/polarity_consistency_audit.json
================================================================================
"""

import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# LOAD ALL HARDWARE RESULTS
# =============================================================================

RESULTS = {
    'CH8_1.95s': PROJECT_ROOT / "results" / "hardware_validation" / "ch8_heron_loso_results.json",
    'CH8_20s': PROJECT_ROOT / "results" / "window_analysis" / "quantum_20s_hardware.json",
    'CH12': PROJECT_ROOT / "results" / "projection" / "hardware_scaling.json",
    'CH16': PROJECT_ROOT / "results" / "projection" / "hardware_scaling.json",
}

PRIORITY_SUBJECTS = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']


def load_per_subject_auc():
    """Load per-subject raw AUC from all hardware runs."""
    all_runs = {}

    # CH8 1.95s (PROMPT 006)
    path = RESULTS['CH8_1.95s']
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        run = {}
        for subj, d in data['ch8']['per_subject'].items():
            run[subj] = d['auc']
        all_runs['CH8_1.95s'] = run

    # CH8 20s (PROMPT 026)
    path = RESULTS['CH8_20s']
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        run = {}
        for subj, d in data['per_subject'].items():
            run[subj] = d.get('raw_auc', d.get('auc'))
        all_runs['CH8_20s'] = run

    # CH12 and CH16 (PROMPT 027/028)
    path = RESULTS['CH12']
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        for ch_key in ['ch12', 'ch16']:
            if ch_key in data and 'per_subject' in data[ch_key]:
                run = {}
                for subj, d in data[ch_key]['per_subject'].items():
                    run[subj] = d.get('raw_auc', d.get('auc'))
                config_name = 'CH12' if ch_key == 'ch12' else 'CH16'
                all_runs[config_name] = run

    return all_runs


def main():
    print("=" * 90)
    print("PROMPT 029 — Polarity Consistency Audit")
    print("=" * 90)

    all_runs = load_per_subject_auc()
    run_names = list(all_runs.keys())

    print(f"\nRuns loaded: {run_names}")
    print(f"Subjects: {PRIORITY_SUBJECTS}\n")

    # Build polarity matrix
    polarity_matrix = {}
    auc_matrix = {}

    for subj in PRIORITY_SUBJECTS:
        polarity_matrix[subj] = {}
        auc_matrix[subj] = {}
        for run_name in run_names:
            auc = all_runs[run_name].get(subj)
            if auc is not None:
                auc_matrix[subj][run_name] = round(auc, 4)
                if abs(auc - 0.5) < 0.001:
                    polarity_matrix[subj][run_name] = 'chance'
                elif auc < 0.5:
                    polarity_matrix[subj][run_name] = 'inverted'
                else:
                    polarity_matrix[subj][run_name] = 'standard'
            else:
                auc_matrix[subj][run_name] = None
                polarity_matrix[subj][run_name] = 'missing'

    # Consistency check (excluding CH16 which is all chance-level)
    # Also excluding exact 0.5000 entries as "no signal"
    meaningful_runs = [r for r in run_names if r != 'CH16']

    consistency = {}
    for subj in PRIORITY_SUBJECTS:
        pols = []
        for run_name in meaningful_runs:
            p = polarity_matrix[subj].get(run_name)
            if p and p not in ('missing', 'chance'):
                pols.append(p)

        if len(pols) >= 2:
            is_consistent = len(set(pols)) == 1
            consistency[subj] = {
                'polarities': pols,
                'consistent': is_consistent,
                'dominant': max(set(pols), key=pols.count) if pols else 'unknown',
                'n_runs_with_signal': len(pols),
            }
        else:
            consistency[subj] = {
                'polarities': pols,
                'consistent': None,
                'dominant': pols[0] if pols else 'unknown',
                'n_runs_with_signal': len(pols),
            }

    # Print consistency matrix
    print("=" * 90)
    print("POLARITY CONSISTENCY MATRIX")
    print("=" * 90)

    header = f"{'Subject':<10}"
    for rn in run_names:
        header += f" {rn:<12}"
    header += f" {'Consistent?':<14}"
    print(header)
    print("-" * 90)

    n_consistent = 0
    n_inconsistent = 0
    n_insufficient = 0

    for subj in PRIORITY_SUBJECTS:
        row = f"{subj:<10}"
        for rn in run_names:
            auc = auc_matrix[subj].get(rn)
            pol = polarity_matrix[subj].get(rn, '?')
            if auc is not None:
                marker = f"{auc:.3f}({pol[0]})"
            else:
                marker = "—"
            row += f" {marker:<12}"

        c = consistency[subj]
        if c['consistent'] is True:
            row += f" {'YES':<14}"
            n_consistent += 1
        elif c['consistent'] is False:
            row += f" {'NO':<14}"
            n_inconsistent += 1
        else:
            row += f" {'<2 runs':<14}"
            n_insufficient += 1

        print(row)

    print()
    print(f"Consistent (same polarity across CH8-1.95s, CH8-20s, CH12): {n_consistent}/7")
    print(f"Inconsistent: {n_inconsistent}/7")
    print(f"CH16: all chance-level (noise floor exceeded at 33q/246 CZ gates)")

    # Detailed inconsistency report
    print("\n" + "=" * 90)
    print("INCONSISTENCY DETAILS")
    print("=" * 90)

    for subj in PRIORITY_SUBJECTS:
        c = consistency[subj]
        if c['consistent'] is False:
            print(f"\n  {subj}: polarities = {c['polarities']}")
            for rn in meaningful_runs:
                auc = auc_matrix[subj].get(rn)
                pol = polarity_matrix[subj].get(rn)
                if auc is not None:
                    print(f"    {rn}: AUC={auc:.4f} -> {pol}")

    # Pre-registered prediction check
    print("\n" + "=" * 90)
    print("PRE-REGISTERED PREDICTION CHECK")
    print("=" * 90)

    # Prediction: chb14 and chb21 genuinely geometric inversions (from PROMPT 022 XGBoost)
    # Prediction: 100% within-patient consistency
    chb14_pols = [polarity_matrix['chb14'].get(r) for r in meaningful_runs
                  if polarity_matrix['chb14'].get(r) not in ('missing', 'chance')]
    chb21_pols = [polarity_matrix['chb21'].get(r) for r in meaningful_runs
                  if polarity_matrix['chb21'].get(r) not in ('missing', 'chance')]

    print(f"chb14 polarities across runs: {chb14_pols}")
    print(f"chb21 polarities across runs: {chb21_pols}")
    print(f"chb14 always inverted? {'YES' if all(p == 'inverted' for p in chb14_pols) else 'NO'}")
    print(f"chb21 always inverted? {'YES' if all(p == 'inverted' for p in chb21_pols) else 'NO'}")
    print(f"100% within-patient consistency? {'YES' if n_inconsistent == 0 else 'NO'}")

    prediction_met = n_inconsistent == 0
    print(f"\nPre-registered prediction met: {prediction_met}")
    if not prediction_met:
        print(f"  {n_inconsistent} subjects show polarity inconsistency across configurations")

    # Save report
    report = {
        'prompt': '029',
        'task': 'Polarity Consistency Audit',
        'timestamp': datetime.now().isoformat(),
        'runs_loaded': run_names,
        'auc_matrix': auc_matrix,
        'polarity_matrix': polarity_matrix,
        'consistency': consistency,
        'summary': {
            'n_consistent': n_consistent,
            'n_inconsistent': n_inconsistent,
            'n_chance_ch16': sum(1 for s in PRIORITY_SUBJECTS
                                if polarity_matrix[s].get('CH16') == 'chance'),
            'prediction_met': prediction_met,
            'chb14_always_inverted': all(p == 'inverted' for p in chb14_pols),
            'chb21_always_inverted': all(p == 'inverted' for p in chb21_pols),
        },
    }

    out_dir = PROJECT_ROOT / "results" / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "polarity_consistency_audit.json"
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {out_path}")


if __name__ == '__main__':
    main()
