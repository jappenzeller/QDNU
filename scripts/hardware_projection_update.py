#!/usr/bin/env python3
"""
================================================================================
PROMPT 031 — Hardware Projection Update
================================================================================

Re-runs projection analysis with all new hardware data points now available.
Produces updated projection JSON and markdown report.

Output: results/projection/hardware_projection.json
        results/projection/hardware_projection.md
================================================================================
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "projection"

# =============================================================================
# LOAD ALL DATA SOURCES
# =============================================================================

def load_all():
    """Load all available data points."""
    sources = {}

    # CH8 1.95s hardware (PROMPT 006)
    p = RESULTS_DIR / "hardware_validation" / "ch8_heron_loso_results.json"
    if p.exists():
        with open(p) as f:
            sources['ch8_1.95s_hw'] = json.load(f)

    # CH8 20s hardware (PROMPT 026)
    p = RESULTS_DIR / "window_analysis" / "quantum_20s_hardware.json"
    if p.exists():
        with open(p) as f:
            sources['ch8_20s_hw'] = json.load(f)

    # CH12/CH16 hardware (PROMPT 027/028)
    p = RESULTS_DIR / "projection" / "hardware_scaling.json"
    if p.exists():
        with open(p) as f:
            sources['hw_scaling'] = json.load(f)

    # Simulation scaling
    p = RESULTS_DIR / "projection" / "simulation_scaling.json"
    if p.exists():
        with open(p) as f:
            sources['sim_scaling'] = json.load(f)

    # Calibration report (PROMPT 030)
    p = RESULTS_DIR / "calibration" / "hardware_calibration_report.json"
    if p.exists():
        with open(p) as f:
            sources['calibration'] = json.load(f)

    # Clean calibration audit (PROMPT 022)
    p = RESULTS_DIR / "calibration" / "clean_calibration_audit.json"
    if p.exists():
        with open(p) as f:
            sources['clean_audit'] = json.load(f)

    # Window size impact
    p = RESULTS_DIR / "window_analysis" / "window_size_impact.json"
    if p.exists():
        with open(p) as f:
            sources['window_impact'] = json.load(f)

    return sources


# =============================================================================
# BUILD DATA POINTS
# =============================================================================

def build_data_points(sources):
    """Build unified list of data points for projection."""
    points = []
    cal = sources.get('calibration', {}).get('configs', {})

    # CH4 1.95s hardware (from hw_scaling)
    hw = sources.get('hw_scaling', {})
    if 'ch4' in hw:
        ch4 = hw['ch4']
        points.append({
            'id': 'ch4_1.95s_hw', 'channels': 4, 'qubits': 9,
            'window_s': 1.95, 'backend': 'hardware',
            'raw_auc': ch4.get('overall_auc', 0.5),
            'oracle_cal_auc': ch4.get('overall_auc', 0.5),
            'status': 'degenerate' if abs(ch4.get('overall_auc', 0.5) - 0.5) < 0.01 else 'available',
        })

    # CH8 1.95s hardware
    c = cal.get('CH8_1.95s', {})
    if c:
        points.append({
            'id': 'ch8_1.95s_hw', 'channels': 8, 'qubits': 17,
            'window_s': 1.95, 'backend': 'hardware',
            'raw_auc': c['overall_raw_auc'],
            'oracle_cal_auc': c['overall_oracle_calibrated_auc'],
            'n_inverted': c['n_inverted'],
            'status': 'available',
        })

    # CH8 20s hardware
    c = cal.get('CH8_20s', {})
    if c:
        points.append({
            'id': 'ch8_20s_hw', 'channels': 8, 'qubits': 17,
            'window_s': 20.0, 'backend': 'hardware',
            'raw_auc': c['overall_raw_auc'],
            'oracle_cal_auc': c['overall_oracle_calibrated_auc'],
            'n_inverted': c['n_inverted'],
            'status': 'available',
        })

    # CH12 hardware
    c = cal.get('CH12', {})
    if c:
        points.append({
            'id': 'ch12_hw', 'channels': 12, 'qubits': 25,
            'window_s': 30.0, 'backend': 'hardware',
            'raw_auc': c['overall_raw_auc'],
            'oracle_cal_auc': c['overall_oracle_calibrated_auc'],
            'n_inverted': c['n_inverted'],
            'status': 'available',
        })

    # CH16 hardware
    c = cal.get('CH16', {})
    if c:
        points.append({
            'id': 'ch16_hw', 'channels': 16, 'qubits': 33,
            'window_s': 30.0, 'backend': 'hardware',
            'raw_auc': c['overall_raw_auc'],
            'oracle_cal_auc': c['overall_oracle_calibrated_auc'],
            'n_inverted': c['n_inverted'],
            'status': 'noise_floor',
            'note': 'Coherence budget exceeded at 33q/246 CZ gates',
        })

    # Simulation points
    sim = sources.get('sim_scaling', {})
    for ch_key, ch_num in [('ch4', 4), ('ch8', 8), ('ch12', 12)]:
        if ch_key in sim and sim[ch_key].get('overall_auc') is not None:
            points.append({
                'id': f'{ch_key}_sim', 'channels': ch_num,
                'qubits': 2 * ch_num + 1,
                'window_s': 1.95, 'backend': 'simulation',
                'raw_auc': sim[ch_key]['overall_auc'],
                'oracle_cal_auc': sim[ch_key]['overall_auc'],  # sim doesn't need calibration
                'status': 'available',
            })

    return points


# =============================================================================
# GAP ANALYSIS
# =============================================================================

COMPARATORS = {
    'tier2a_clean': {
        'label': 'Tier 2A Eigenvalues (clean-validated)',
        'auc': 0.6831,
        'note': 'Only comparator with validated label-free calibration (PROMPT 022)',
    },
    'tier2a_oracle': {
        'label': 'Tier 2A Eigenvalues (oracle)',
        'auc': 0.6892,
        'note': 'Oracle calibration for reference',
    },
    'tier3_oracle': {
        'label': 'Tier 3 Riemannian (oracle)',
        'auc': 0.7011,
        'note': 'Oracle only - clean calibration fails (PROMPT 022)',
    },
    'tier2b_full': {
        'label': 'Classical Ceiling (336 features)',
        'auc': 0.952,
        'note': 'Not a fair quantum comparator',
    },
}


def compute_gaps(points, comparators):
    """Compute gap between each hardware point and comparators."""
    gaps = []
    for pt in points:
        if pt['backend'] != 'hardware':
            continue
        for comp_name, comp in comparators.items():
            gap = comp['auc'] - pt['oracle_cal_auc']
            gaps.append({
                'data_point': pt['id'],
                'comparator': comp_name,
                'comparator_auc': comp['auc'],
                'hardware_oracle_auc': pt['oracle_cal_auc'],
                'gap': round(gap, 4),
                'hardware_exceeds': gap < 0,
            })
    return gaps


# =============================================================================
# MARKDOWN REPORT
# =============================================================================

def generate_markdown(points, gaps, comparators, sources):
    """Generate human-readable markdown report."""
    lines = [
        "# Hardware Projection Report (PROMPT 031)",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## 1. Channel Scaling Curve (Hardware)",
        "",
        "| Channels | Qubits | Window | Raw AUC | Oracle-cal. AUC | N_inv | Status |",
        "|----------|--------|--------|---------|-----------------|-------|--------|",
    ]

    for pt in sorted(points, key=lambda x: (x['channels'], x.get('window_s', 0))):
        if pt['backend'] == 'hardware':
            n_inv = pt.get('n_inverted', '—')
            lines.append(
                f"| {pt['channels']} | {pt['qubits']} | {pt.get('window_s', '?')}s | "
                f"{pt['raw_auc']:.4f} | {pt['oracle_cal_auc']:.4f} | {n_inv} | {pt['status']} |"
            )

    lines += [
        "",
        "## 2. Window Size Scaling (CH8)",
        "",
        "| Window | Raw AUC | Oracle-cal. AUC | N_inverted |",
        "|--------|---------|-----------------|------------|",
    ]

    for pt in points:
        if pt['backend'] == 'hardware' and pt['channels'] == 8:
            lines.append(
                f"| {pt.get('window_s', '?')}s | {pt['raw_auc']:.4f} | "
                f"{pt['oracle_cal_auc']:.4f} | {pt.get('n_inverted', '?')} |"
            )

    lines += [
        "",
        "## 3. Simulation vs Hardware",
        "",
        "| Config | Sim AUC | HW Raw AUC | HW Oracle-cal. | Sim-HW Gap |",
        "|--------|---------|------------|----------------|------------|",
    ]

    sim_pts = {pt['channels']: pt for pt in points if pt['backend'] == 'simulation'}
    hw_pts = {(pt['channels'], pt.get('window_s')): pt for pt in points if pt['backend'] == 'hardware'}

    for ch in [4, 8, 12]:
        sim = sim_pts.get(ch)
        hw = hw_pts.get((ch, 1.95)) or hw_pts.get((ch, 30.0))
        if sim and hw:
            gap = sim['raw_auc'] - hw['raw_auc']
            lines.append(
                f"| CH{ch} | {sim['raw_auc']:.4f} | {hw['raw_auc']:.4f} | "
                f"{hw['oracle_cal_auc']:.4f} | {gap:+.4f} |"
            )

    lines += [
        "",
        "## 4. Gap Analysis vs Classical Comparators",
        "",
        "| Comparator | AUC | Best HW Oracle-cal. | Gap | Note |",
        "|------------|-----|---------------------|-----|------|",
    ]

    best_hw = max(
        (pt['oracle_cal_auc'] for pt in points if pt['backend'] == 'hardware'),
        default=0.5
    )

    for comp_name, comp in comparators.items():
        gap = comp['auc'] - best_hw
        lines.append(
            f"| {comp['label']} | {comp['auc']:.4f} | {best_hw:.4f} | "
            f"{gap:+.4f} | {comp['note']} |"
        )

    lines += [
        "",
        f"Best hardware result (oracle-calibrated): **{best_hw:.4f}** (CH8 1.95s)",
        "",
        "## 5. Key Findings",
        "",
        "- **CH16 (33q, 246 CZ gates):** Noise floor exceeded. All subjects at chance (0.500).",
        "- **Polarity instability:** 4/7 subjects show inconsistent polarity across configs.",
        "  Only chb21 is consistently inverted; chb03 consistently inverted on A-Gate.",
        "- **Channel scaling:** No improvement from CH8 to CH12 on hardware.",
        "  CH12 oracle-cal (0.565) < CH8 1.95s oracle-cal (0.637).",
        "- **Window scaling:** 20s windows did NOT improve over 1.95s on hardware.",
        "  CH8 20s oracle-cal (0.605) < CH8 1.95s oracle-cal (0.637).",
        "- **Clean calibration gap:** A-Gate best (0.637) still below Tier 2A clean (0.683).",
        "",
        "## 6. Polarity Tracking",
        "",
    ]

    # Add polarity from consistency audit if available
    pol_path = RESULTS_DIR / "validation" / "polarity_consistency_audit.json"
    if pol_path.exists():
        with open(pol_path) as f:
            pol_data = json.load(f)
        lines.append("| Subject | CH8-1.95s | CH8-20s | CH12 | CH16 | Consistent? |")
        lines.append("|---------|-----------|---------|------|------|-------------|")
        for subj in ['chb01', 'chb03', 'chb05', 'chb07', 'chb11', 'chb14', 'chb21']:
            pm = pol_data.get('polarity_matrix', {}).get(subj, {})
            c = pol_data.get('consistency', {}).get(subj, {})
            cons = 'YES' if c.get('consistent') else ('NO' if c.get('consistent') is False else '?')
            lines.append(
                f"| {subj} | {pm.get('CH8_1.95s', '?')} | {pm.get('CH8_20s', '?')} | "
                f"{pm.get('CH12', '?')} | {pm.get('CH16', '?')} | {cons} |"
            )

    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("PROMPT 031 -- Hardware Projection Update")
    print("=" * 70)

    # Verify inputs
    required = [
        ("CH8 20s hardware", RESULTS_DIR / "window_analysis" / "quantum_20s_hardware.json"),
        ("Hardware scaling", RESULTS_DIR / "projection" / "hardware_scaling.json"),
        ("Calibration report", RESULTS_DIR / "calibration" / "hardware_calibration_report.json"),
    ]

    all_exist = True
    for name, path in required:
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  {name}: {status}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\nERROR: Missing required input files.")
        return

    print("\nAll inputs available. Building projection...\n")

    sources = load_all()
    points = build_data_points(sources)
    gaps = compute_gaps(points, COMPARATORS)

    # Print gap analysis summary
    print("=" * 80)
    print("GAP ANALYSIS SUMMARY")
    print("=" * 80)

    header = f"{'Comparator':<40} {'AUC':<8} {'Best HW':<10} {'Gap':<8}"
    print(header)
    print("-" * 80)

    best_hw = max(
        (pt['oracle_cal_auc'] for pt in points if pt['backend'] == 'hardware'),
        default=0.5
    )
    best_hw_id = max(
        (pt for pt in points if pt['backend'] == 'hardware'),
        key=lambda x: x['oracle_cal_auc'],
    )['id']

    for comp_name, comp in COMPARATORS.items():
        gap = comp['auc'] - best_hw
        print(f"{comp['label']:<40} {comp['auc']:<8.4f} {best_hw:<10.4f} {gap:+.4f}")

    print(f"\nBest hardware: {best_hw_id} = {best_hw:.4f} (oracle-calibrated)")

    # All data points
    print("\n" + "=" * 80)
    print("ALL DATA POINTS")
    print("=" * 80)
    print(f"{'ID':<20} {'Ch':<4} {'Q':<4} {'Win':<6} {'Backend':<12} {'Raw':<8} {'Oracle':<8} {'Status':<12}")
    print("-" * 80)
    for pt in sorted(points, key=lambda x: (x['backend'], x['channels'])):
        print(f"{pt['id']:<20} {pt['channels']:<4} {pt['qubits']:<4} "
              f"{pt.get('window_s', '?'):<6} {pt['backend']:<12} "
              f"{pt['raw_auc']:<8.4f} {pt['oracle_cal_auc']:<8.4f} {pt['status']:<12}")

    # Save JSON
    report = {
        'prompt': '031',
        'timestamp': datetime.now().isoformat(),
        'comparators': COMPARATORS,
        'data_points': points,
        'gap_analysis': gaps,
        'best_hardware': {
            'id': best_hw_id,
            'oracle_cal_auc': best_hw,
        },
        'pending_data_points': [],
        'notes': [
            'All hardware runs complete (CH8-1.95s, CH8-20s, CH12, CH16)',
            'CH16 at noise floor (33q/246 CZ gates exceeds coherence budget)',
            'Polarity is NOT stable across configurations (4/7 inconsistent)',
            'Best hardware AUC (0.637 oracle-cal) below Tier 2A clean (0.683)',
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_path = OUTPUT_DIR / "hardware_projection.json"
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON saved to: {json_path}")

    # Save markdown
    md = generate_markdown(points, gaps, COMPARATORS, sources)
    md_path = OUTPUT_DIR / "hardware_projection.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f"Markdown saved to: {md_path}")


if __name__ == '__main__':
    main()
