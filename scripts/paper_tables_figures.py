#!/usr/bin/env python3
"""
================================================================================
PROMPT 032 — Paper Tables and Figures
================================================================================

Generates publication-ready tables and figures for polarity_paper_v2.docx.

Output: results/paper_artifacts/
================================================================================
"""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "paper_artifacts"

# Color scheme
C_HW = '#1f77b4'       # blue - hardware
C_SIM = '#ff7f0e'      # orange - simulation
C_T2A = '#2ca02c'      # green - Tier 2A
C_T3 = '#d62728'       # red - Tier 3
C_RAW = '#9467bd'       # purple - raw
C_ORACLE = '#1f77b4'    # blue - oracle calibrated
C_NOISE = '#7f7f7f'     # gray - noise floor


def load_sources():
    s = {}
    for name, path in [
        ('calibration', RESULTS_DIR / "calibration" / "hardware_calibration_report.json"),
        ('projection', RESULTS_DIR / "projection" / "hardware_projection.json"),
        ('consistency', RESULTS_DIR / "validation" / "polarity_consistency_audit.json"),
        ('clean_audit', RESULTS_DIR / "calibration" / "clean_calibration_audit.json"),
        ('hw_scaling', RESULTS_DIR / "projection" / "hardware_scaling.json"),
        ('probe', RESULTS_DIR / "preflight" / "hardware_probe_chb01.json"),
    ]:
        if path.exists():
            with open(path) as f:
                s[name] = json.load(f)
    return s


# =============================================================================
# TABLE 1: Hardware Scaling Results
# =============================================================================

def table1_hardware_scaling(sources):
    cal = sources.get('calibration', {}).get('configs', {})
    probe = sources.get('probe', {}).get('configs', {})

    rows = []
    configs = [
        ('CH8-1.95s', 'CH8_1.95s', 8, 17, None),
        ('CH8-20s', 'CH8_20s', 8, 17, 'CH8_20s'),
        ('CH12', 'CH12', 12, 25, 'CH12'),
        ('CH16', 'CH16', 16, 33, 'CH16'),
    ]

    for label, key, ch, qubits, probe_key in configs:
        c = cal.get(key, {})
        raw = c.get('overall_raw_auc', '')
        oracle = c.get('overall_oracle_calibrated_auc', '')
        n_inv = c.get('n_inverted', '')

        # CZ gates from probe
        cz = ''
        if probe_key and probe_key in probe:
            cz = probe[probe_key].get('transpiled_2q_gates', '')
        elif key == 'CH8_1.95s' and 'CH8_20s' in probe:
            cz = probe['CH8_20s'].get('transpiled_2q_gates', '')  # same circuit structure

        chb14 = 'Y' if c.get('chb14_inverted') else 'N'
        chb21 = 'Y' if c.get('chb21_inverted') else 'N'

        rows.append([label, ch, qubits, cz,
                     f"{raw:.4f}" if isinstance(raw, float) else str(raw),
                     f"{oracle:.4f}" if isinstance(oracle, float) else str(oracle),
                     n_inv, f"{chb14}/{chb21}"])

    # Write CSV
    csv_path = OUTPUT_DIR / "table1_hardware_scaling.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Config', 'Channels', 'Qubits', 'CZ_Gates', 'Raw_AUC',
                     'Oracle_Cal_AUC', 'N_Inverted', 'chb14_chb21_inv'])
        w.writerows(rows)

    print(f"  Table 1 saved: {csv_path}")

    # Print
    print("\n  TABLE 1: Hardware Scaling Results")
    print(f"  {'Config':<12} {'Ch':<4} {'Q':<4} {'CZ':<6} {'Raw AUC':<10} {'Oracle-cal.':<12} {'N_inv':<6} {'14/21'}")
    print("  " + "-" * 70)
    for r in rows:
        print(f"  {r[0]:<12} {r[1]:<4} {r[2]:<4} {str(r[3]):<6} {r[4]:<10} {r[5]:<12} {str(r[6]):<6} {r[7]}")


# =============================================================================
# TABLE 2: Polarity Consistency
# =============================================================================

def table2_polarity_consistency(sources):
    con = sources.get('consistency', {})
    pm = con.get('polarity_matrix', {})
    cs = con.get('consistency', {})

    subjects = ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']
    runs = ['CH8_1.95s', 'CH8_20s', 'CH12', 'CH16']

    rows = []
    for subj in subjects:
        row = [subj]
        for rn in runs:
            row.append(pm.get(subj, {}).get(rn, '?'))
        consistent = cs.get(subj, {}).get('consistent')
        row.append('YES' if consistent else ('NO' if consistent is False else '?'))

        # Note from PROMPT 022
        if subj == 'chb21':
            row.append('geometric (XGB confirmed)')
        elif subj == 'chb03':
            row.append('consistent on A-Gate')
        elif subj in ('chb05', 'chb07', 'chb14'):
            row.append('config-dependent')
        else:
            row.append('')

        rows.append(row)

    csv_path = OUTPUT_DIR / "table2_polarity_consistency.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Subject'] + runs + ['Consistent', 'Note'])
        w.writerows(rows)

    print(f"\n  Table 2 saved: {csv_path}")

    print("\n  TABLE 2: Polarity Consistency")
    print(f"  {'Subject':<10} {'CH8-1.95s':<12} {'CH8-20s':<12} {'CH12':<12} {'CH16':<12} {'Cons?':<8} {'Note'}")
    print("  " + "-" * 80)
    for r in rows:
        print(f"  {r[0]:<10} {r[1]:<12} {r[2]:<12} {r[3]:<12} {r[4]:<12} {r[5]:<8} {r[6]}")


# =============================================================================
# TABLE 3: Calibration Validity Summary
# =============================================================================

def table3_calibration_validity(sources):
    audit = sources.get('clean_audit', {})
    cal_results = audit.get('calibration_results', {})

    rows = []
    for clf_name, display in [('Tier3_LDA', 'Tier 3 LDA'),
                               ('Tier2A_Eigen', 'Tier 2A Eigen'),
                               ('Tier3_XGB', 'Tier 3 XGB'),
                               ('AGate_Sim', 'A-Gate Sim')]:
        cr = cal_results.get(clf_name, {})
        oracle = cr.get('overall_oracle_auc', '')
        a_auc = cr.get('overall_approach_a_auc', '')
        b_auc = cr.get('overall_approach_b_auc', '')
        b_agree = cr.get('approach_b_agreement', '')

        clean_valid = 'YES' if clf_name == 'Tier2A_Eigen' else 'NO'

        rows.append([display,
                     f"{oracle:.4f}" if isinstance(oracle, float) else str(oracle),
                     f"{b_auc:.4f}" if isinstance(b_auc, float) else str(b_auc),
                     str(b_agree), clean_valid])

    csv_path = OUTPUT_DIR / "table3_calibration_validity.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Classifier', 'Oracle_Cal_AUC', 'Approach_B_AUC', 'B_Agreement', 'Clean_Validated'])
        w.writerows(rows)

    print(f"\n  Table 3 saved: {csv_path}")

    print("\n  TABLE 3: Calibration Validity")
    print(f"  {'Classifier':<16} {'Oracle':<10} {'Appr.B':<10} {'B agree':<10} {'Clean?'}")
    print("  " + "-" * 60)
    for r in rows:
        print(f"  {r[0]:<16} {r[1]:<10} {r[2]:<10} {r[3]:<10} {r[4]}")


# =============================================================================
# FIGURE 4: Channel Scaling AUC Plot
# =============================================================================

def figure4_channel_scaling(sources):
    cal = sources.get('calibration', {}).get('configs', {})

    # Hardware points
    hw_channels = []
    hw_raw = []
    hw_oracle = []

    for key, ch in [('CH8_1.95s', 8), ('CH12', 12), ('CH16', 16)]:
        c = cal.get(key, {})
        if c:
            hw_channels.append(ch)
            hw_raw.append(c['overall_raw_auc'])
            hw_oracle.append(c['overall_oracle_calibrated_auc'])

    # Simulation points
    proj = sources.get('projection', {})
    sim_channels = []
    sim_auc = []
    for pt in proj.get('data_points', []):
        if pt['backend'] == 'simulation' and pt['channels'] in (4, 8, 12):
            sim_channels.append(pt['channels'])
            sim_auc.append(pt['raw_auc'])

    sim_channels, sim_auc = zip(*sorted(zip(sim_channels, sim_auc))) if sim_channels else ([], [])

    fig, ax = plt.subplots(figsize=(8, 5))

    # Hardware
    ax.plot(hw_channels, hw_raw, 'o--', color=C_RAW, label='Hardware raw AUC', markersize=8, alpha=0.7)
    ax.plot(hw_channels, hw_oracle, 's-', color=C_ORACLE, label='Hardware oracle-cal. AUC', markersize=8, linewidth=2)

    # Simulation
    if sim_channels:
        ax.plot(sim_channels, sim_auc, '^:', color=C_SIM, label='Simulation AUC', markersize=8, alpha=0.7)

    # Comparator lines
    ax.axhline(y=0.6831, color=C_T2A, linestyle='--', linewidth=1.5, alpha=0.8,
               label='Tier 2A clean-validated (0.683)')
    ax.axhline(y=0.5, color=C_NOISE, linestyle=':', linewidth=1, alpha=0.5, label='Chance level')

    ax.set_xlabel('Number of EEG Channels', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('A-Gate Channel Scaling on IBM Torino', fontsize=13)
    ax.set_xticks([4, 8, 12, 16])
    ax.set_ylim(0.4, 0.75)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    png_path = OUTPUT_DIR / "figure4_channel_scaling.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # CSV
    csv_path = OUTPUT_DIR / "figure4_channel_scaling.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Channels', 'HW_Raw_AUC', 'HW_Oracle_Cal_AUC', 'Sim_AUC'])
        for ch in [4, 8, 12, 16]:
            hw_r = cal.get(f'CH{ch}_1.95s' if ch == 8 else f'CH{ch}', {}).get('overall_raw_auc', '')
            hw_o = cal.get(f'CH{ch}_1.95s' if ch == 8 else f'CH{ch}', {}).get('overall_oracle_calibrated_auc', '')
            sim_v = dict(zip(sim_channels, sim_auc)).get(ch, '')
            w.writerow([ch,
                        f"{hw_r:.4f}" if isinstance(hw_r, float) else '',
                        f"{hw_o:.4f}" if isinstance(hw_o, float) else '',
                        f"{sim_v:.4f}" if isinstance(sim_v, float) else ''])

    print(f"\n  Figure 4 saved: {png_path}")
    print(f"  Figure 4 CSV: {csv_path}")


# =============================================================================
# FIGURE 5: Window Size Effect
# =============================================================================

def figure5_window_effect(sources):
    cal = sources.get('calibration', {}).get('configs', {})

    ch8_195 = cal.get('CH8_1.95s', {})
    ch8_20s = cal.get('CH8_20s', {})

    labels = ['CH8 @ 1.95s', 'CH8 @ 20s']
    raw_vals = [ch8_195.get('overall_raw_auc', 0), ch8_20s.get('overall_raw_auc', 0)]
    oracle_vals = [ch8_195.get('overall_oracle_calibrated_auc', 0),
                   ch8_20s.get('overall_oracle_calibrated_auc', 0)]

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 5))

    bars1 = ax.bar(x - width/2, raw_vals, width, label='Raw AUC', color=C_RAW, alpha=0.7)
    bars2 = ax.bar(x + width/2, oracle_vals, width, label='Oracle-cal. AUC', color=C_ORACLE, alpha=0.9)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    # Classical reference
    ax.axhline(y=0.6831, color=C_T2A, linestyle='--', linewidth=1.5, alpha=0.8,
               label='Tier 2A clean-validated (0.683)')
    ax.axhline(y=0.5, color=C_NOISE, linestyle=':', linewidth=1, alpha=0.5)

    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Window Size Effect on A-Gate (CH8, IBM Torino)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0.3, 0.75)
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    png_path = OUTPUT_DIR / "figure5_window_effect.png"
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # CSV
    csv_path = OUTPUT_DIR / "figure5_window_effect.csv"
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Config', 'Raw_AUC', 'Oracle_Cal_AUC'])
        w.writerow(['CH8_1.95s', f"{raw_vals[0]:.4f}", f"{oracle_vals[0]:.4f}"])
        w.writerow(['CH8_20s', f"{raw_vals[1]:.4f}", f"{oracle_vals[1]:.4f}"])

    print(f"\n  Figure 5 saved: {png_path}")
    print(f"  Figure 5 CSV: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PROMPT 032 -- Paper Tables and Figures")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = load_sources()
    print(f"Sources loaded: {list(sources.keys())}\n")

    table1_hardware_scaling(sources)
    table2_polarity_consistency(sources)
    table3_calibration_validity(sources)
    figure4_channel_scaling(sources)
    figure5_window_effect(sources)

    print("\n" + "=" * 70)
    print("All 5 assets generated.")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == '__main__':
    main()
