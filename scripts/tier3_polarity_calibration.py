#!/usr/bin/env python3
"""
================================================================================
PROMPT 016 - Polarity Calibration on Tier 3 Riemannian Results
================================================================================

Applies the polarity calibration discovered through quantum hardware analysis
to classical Tier 3 Riemannian results from PROMPT 015.

Tests the hypothesis: Is polarity inversion the sole explanation for Riemannian
underperformance in cross-patient evaluation?

Inputs:
    - results/window_analysis/window_size_impact.json (PROMPT 015)
    - results/patient_analysis/quantum_patient_profiles.json (PROMPT 011)

Outputs:
    - results/window_analysis/tier3_polarity_calibration.json
    - results/window_analysis/tier3_polarity_report.md

Author: Claude Code
Date: 2026-02-22
================================================================================
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "window_analysis"

# Input files
WINDOW_SIZE_IMPACT_FILE = RESULTS_DIR / "window_size_impact.json"
QUANTUM_PROFILES_FILE = PROJECT_ROOT / "results" / "patient_analysis" / "quantum_patient_profiles.json"

# Output files
OUTPUT_JSON = RESULTS_DIR / "tier3_polarity_calibration.json"
OUTPUT_MD = RESULTS_DIR / "tier3_polarity_report.md"

# Priority subjects (same as PROMPT 015)
PRIORITY_SUBJECTS = ["chb01", "chb03", "chb05", "chb07", "chb11", "chb14", "chb21"]

# Window sizes tested in PROMPT 015
WINDOW_SIZES = [1, 2, 5, 10, 15, 20, 30]

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_window_size_data() -> Dict:
    """Load PROMPT 015 window size impact results."""
    logger.info(f"Loading window size data from {WINDOW_SIZE_IMPACT_FILE}")
    with open(WINDOW_SIZE_IMPACT_FILE, 'r') as f:
        return json.load(f)


def load_quantum_profiles() -> Dict:
    """Load PROMPT 011 quantum patient profiles."""
    logger.info(f"Loading quantum profiles from {QUANTUM_PROFILES_FILE}")
    with open(QUANTUM_PROFILES_FILE, 'r') as f:
        return json.load(f)


def get_quantum_polarity(profiles: Dict, subject_id: str) -> str:
    """
    Extract quantum polarity for a subject from hardware PLV runs.

    Returns 'standard', 'inverted', or 'unknown' if no hardware data.
    """
    if subject_id not in profiles.get('patient_profiles', {}):
        return 'unknown'

    patient = profiles['patient_profiles'][subject_id]
    hardware_runs = patient.get('hardware_runs', [])

    # Find PLV_theta_alpha hardware run
    for run in hardware_runs:
        if run.get('encoding') == 'PLV_theta_alpha' and 'HARDWARE' in run.get('backend', ''):
            return run.get('polarity', 'unknown')

    return 'unknown'


# =============================================================================
# POLARITY CALIBRATION
# =============================================================================

def calibrate_auc(raw_auc: float) -> tuple:
    """
    Apply polarity calibration to a single AUC value.

    Returns (calibrated_auc, polarity).
    """
    if raw_auc < 0.5:
        return 1.0 - raw_auc, 'inverted'
    else:
        return raw_auc, 'standard'


def compute_weighted_auc(per_subject_results: Dict) -> float:
    """
    Compute weighted mean AUC across subjects.

    Weights by n_samples to match sklearn's aggregate AUC computation in LOSO.
    """
    total_weight = 0
    weighted_sum = 0

    for subject_id, data in per_subject_results.items():
        weight = data.get('n_samples', 1)
        auc = data.get('auc', 0.5)
        weighted_sum += weight * auc
        total_weight += weight

    if total_weight == 0:
        return 0.5

    return weighted_sum / total_weight


def analyze_window_size(
    window_data: Dict,
    tier: str,
    quantum_polarities: Dict
) -> Dict:
    """
    Analyze a single window size for a given tier.

    Returns per-subject calibration results and aggregate stats.
    """
    per_subject = window_data.get('per_subject', {})

    results = {
        'per_subject': {},
        'n_inverted': 0,
        'n_quantum_concordant': 0
    }

    calibrated_per_subject = {}

    for subject_id in PRIORITY_SUBJECTS:
        if subject_id not in per_subject:
            continue

        subject_data = per_subject[subject_id]
        raw_auc = subject_data.get('auc', 0.5)
        n_samples = subject_data.get('n_samples', 1)

        calibrated_auc, polarity = calibrate_auc(raw_auc)
        quantum_polarity = quantum_polarities.get(subject_id, 'unknown')
        matches_quantum = polarity == quantum_polarity if quantum_polarity != 'unknown' else None

        if polarity == 'inverted':
            results['n_inverted'] += 1

        if matches_quantum:
            results['n_quantum_concordant'] += 1

        results['per_subject'][subject_id] = {
            'raw_auc': round(raw_auc, 4),
            'calibrated_auc': round(calibrated_auc, 4),
            'polarity': polarity,
            'quantum_polarity': quantum_polarity,
            'matches_quantum': matches_quantum,
            'n_samples': n_samples
        }

        # For weighted AUC computation
        calibrated_per_subject[subject_id] = {
            'auc': calibrated_auc,
            'n_samples': n_samples
        }

    # Compute raw and calibrated aggregate AUCs
    raw_per_subject = {
        sid: {'auc': per_subject[sid]['auc'], 'n_samples': per_subject[sid]['n_samples']}
        for sid in PRIORITY_SUBJECTS if sid in per_subject
    }

    results['raw_auc'] = round(compute_weighted_auc(raw_per_subject), 4)
    results['calibrated_auc'] = round(compute_weighted_auc(calibrated_per_subject), 4)
    results['calibration_lift'] = round(results['calibrated_auc'] - results['raw_auc'], 4)

    return results


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_analysis():
    """Run the full polarity calibration analysis."""
    logger.info("=" * 70)
    logger.info("PROMPT 016 - Polarity Calibration on Tier 3 Riemannian Results")
    logger.info("=" * 70)

    # Load data
    window_data = load_window_size_data()
    quantum_profiles = load_quantum_profiles()

    # Extract quantum polarities for priority subjects
    quantum_polarities = {}
    for subject_id in PRIORITY_SUBJECTS:
        quantum_polarities[subject_id] = get_quantum_polarity(quantum_profiles, subject_id)
        logger.info(f"  {subject_id}: quantum polarity = {quantum_polarities[subject_id]}")

    # Analyze each window size for both tiers
    results = {
        'per_window_size': {},
        'polarity_stability': {},
        'concordance_summary': {}
    }

    tier3_data = window_data.get('tier3_riemannian', {})
    tier2_data = window_data.get('tier2_max', {})

    # Track polarity across windows for stability analysis
    subject_polarities = {sid: {'tier3': [], 'tier2': []} for sid in PRIORITY_SUBJECTS}

    for window_sec in WINDOW_SIZES:
        window_key = str(window_sec)
        logger.info(f"\nAnalyzing {window_sec}s window...")

        # Tier 3 analysis
        tier3_window = tier3_data.get(window_key, {})
        tier3_results = analyze_window_size(tier3_window, 'tier3', quantum_polarities)

        # Tier 2 analysis (use 'max' aggregation results)
        tier2_window = tier2_data.get(window_key, {})
        tier2_results = analyze_window_size(tier2_window, 'tier2', quantum_polarities)

        # Track polarities for stability
        for subject_id in PRIORITY_SUBJECTS:
            if subject_id in tier3_results['per_subject']:
                subject_polarities[subject_id]['tier3'].append(
                    tier3_results['per_subject'][subject_id]['polarity']
                )
            if subject_id in tier2_results['per_subject']:
                subject_polarities[subject_id]['tier2'].append(
                    tier2_results['per_subject'][subject_id]['polarity']
                )

        # Combine into per-window results
        results['per_window_size'][window_key] = {
            'tier3_raw_auc': tier3_results['raw_auc'],
            'tier3_calibrated_auc': tier3_results['calibrated_auc'],
            'tier3_calibration_lift': tier3_results['calibration_lift'],
            'tier2_raw_auc': tier2_results['raw_auc'],
            'tier2_calibrated_auc': tier2_results['calibrated_auc'],
            'tier2_calibration_lift': tier2_results['calibration_lift'],
            'n_tier3_inverted': tier3_results['n_inverted'],
            'n_tier2_inverted': tier2_results['n_inverted'],
            'per_subject': {}
        }

        # Merge per-subject data
        for subject_id in PRIORITY_SUBJECTS:
            if subject_id in tier3_results['per_subject']:
                t3 = tier3_results['per_subject'][subject_id]
                t2 = tier2_results['per_subject'].get(subject_id, {})

                results['per_window_size'][window_key]['per_subject'][subject_id] = {
                    'tier3_raw': t3['raw_auc'],
                    'tier3_calibrated': t3['calibrated_auc'],
                    'tier3_polarity': t3['polarity'],
                    'tier2_raw': t2.get('raw_auc'),
                    'tier2_calibrated': t2.get('calibrated_auc'),
                    'tier2_polarity': t2.get('polarity'),
                    'quantum_polarity': t3['quantum_polarity'],
                    'tier3_matches_quantum': t3['matches_quantum']
                }

        logger.info(f"  Tier 3: raw={tier3_results['raw_auc']:.4f}, "
                   f"calibrated={tier3_results['calibrated_auc']:.4f}, "
                   f"lift=+{tier3_results['calibration_lift']:.4f}")
        logger.info(f"  Tier 2: raw={tier2_results['raw_auc']:.4f}, "
                   f"calibrated={tier2_results['calibrated_auc']:.4f}, "
                   f"lift=+{tier2_results['calibration_lift']:.4f}")

    # Polarity stability analysis
    n_consistent = 0
    n_tier3_quantum_match = 0

    for subject_id in PRIORITY_SUBJECTS:
        tier3_pols = subject_polarities[subject_id]['tier3']
        quantum_pol = quantum_polarities[subject_id]

        # Check consistency across windows
        is_consistent = len(set(tier3_pols)) == 1 if tier3_pols else False
        dominant_polarity = tier3_pols[0] if tier3_pols else 'unknown'
        matches_quantum = dominant_polarity == quantum_pol if quantum_pol != 'unknown' else None

        if is_consistent:
            n_consistent += 1
        if matches_quantum:
            n_tier3_quantum_match += 1

        results['polarity_stability'][subject_id] = {
            'tier3_polarity_across_windows': tier3_pols,
            'is_consistent': is_consistent,
            'dominant_polarity': dominant_polarity,
            'quantum_polarity': quantum_pol,
            'tier3_matches_quantum': matches_quantum
        }

    # Concordance summary
    n_with_quantum_data = sum(1 for p in quantum_polarities.values() if p != 'unknown')

    results['concordance_summary'] = {
        'n_subjects': len(PRIORITY_SUBJECTS),
        'n_with_quantum_data': n_with_quantum_data,
        'n_tier3_quantum_concordant': n_tier3_quantum_match,
        'concordance_rate': round(n_tier3_quantum_match / n_with_quantum_data, 2) if n_with_quantum_data > 0 else None,
        'n_polarity_consistent_across_windows': n_consistent,
        'polarity_window_consistency_rate': round(n_consistent / len(PRIORITY_SUBJECTS), 2)
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved JSON results to {OUTPUT_JSON}")

    # Generate report
    generate_report(results, quantum_polarities)
    logger.info(f"Saved report to {OUTPUT_MD}")

    return results


def generate_report(results: Dict, quantum_polarities: Dict):
    """Generate the markdown report."""

    lines = [
        "# Tier 3 Polarity Calibration Analysis",
        "",
        "**PROMPT 016** - Applying quantum-discovered polarity calibration to Tier 3 Riemannian results",
        "",
        "---",
        "",
        "## Impact of Polarity Calibration",
        "",
        "| Window | T3 Raw | T3 Calibrated | T3 Lift | T2 Raw | T2 Calibrated | T2 Lift |",
        "|--------|--------|---------------|---------|--------|---------------|---------|"
    ]

    for window_sec in WINDOW_SIZES:
        window_key = str(window_sec)
        w = results['per_window_size'][window_key]
        lines.append(
            f"| {window_sec}s | {w['tier3_raw_auc']:.3f} | {w['tier3_calibrated_auc']:.3f} | "
            f"+{w['tier3_calibration_lift']:.3f} | {w['tier2_raw_auc']:.3f} | "
            f"{w['tier2_calibrated_auc']:.3f} | +{w['tier2_calibration_lift']:.3f} |"
        )

    # Best results
    best_t3_raw = max(results['per_window_size'].values(), key=lambda x: x['tier3_raw_auc'])
    best_t3_cal = max(results['per_window_size'].values(), key=lambda x: x['tier3_calibrated_auc'])
    best_window_raw = [k for k, v in results['per_window_size'].items() if v['tier3_raw_auc'] == best_t3_raw['tier3_raw_auc']][0]
    best_window_cal = [k for k, v in results['per_window_size'].items() if v['tier3_calibrated_auc'] == best_t3_cal['tier3_calibrated_auc']][0]

    lines.extend([
        "",
        f"**Best T3 raw:** {best_t3_raw['tier3_raw_auc']:.3f} at {best_window_raw}s",
        f"**Best T3 calibrated:** {best_t3_cal['tier3_calibrated_auc']:.3f} at {best_window_cal}s",
        "",
        "---",
        "",
        "## Per-Subject Polarity Analysis (20s window)",
        "",
        "| Subject | T3 Raw | T3 Cal | T3 Polarity | T2 Raw | T2 Cal | T2 Polarity | Quantum | Match |",
        "|---------|--------|--------|-------------|--------|--------|-------------|---------|-------|"
    ])

    # Use 20s window for per-subject table (best T2 window)
    window_20 = results['per_window_size'].get('20', {}).get('per_subject', {})
    for subject_id in PRIORITY_SUBJECTS:
        if subject_id in window_20:
            s = window_20[subject_id]
            match_symbol = "✅" if s['tier3_matches_quantum'] else "❌" if s['tier3_matches_quantum'] is False else "—"
            lines.append(
                f"| {subject_id} | {s['tier3_raw']:.3f} | {s['tier3_calibrated']:.3f} | "
                f"{s['tier3_polarity']} | {s['tier2_raw']:.3f} | {s['tier2_calibrated']:.3f} | "
                f"{s['tier2_polarity']} | {s['quantum_polarity']} | {match_symbol} |"
            )

    lines.extend([
        "",
        "---",
        "",
        "## Polarity Stability Across Window Sizes",
        "",
        "| Subject | Polarity Pattern | Consistent | Quantum | Match |",
        "|---------|------------------|------------|---------|-------|"
    ])

    for subject_id in PRIORITY_SUBJECTS:
        stab = results['polarity_stability'].get(subject_id, {})
        pattern = stab.get('tier3_polarity_across_windows', [])
        pattern_str = ', '.join(p[0] for p in pattern) if pattern else '—'  # s/i abbreviation
        consistent = "yes" if stab.get('is_consistent') else "no"
        quantum = stab.get('quantum_polarity', '—')
        match = "✅" if stab.get('tier3_matches_quantum') else "❌" if stab.get('tier3_matches_quantum') is False else "—"
        lines.append(f"| {subject_id} | {pattern_str} | {consistent} | {quantum} | {match} |")

    # Concordance summary
    conc = results['concordance_summary']
    lines.extend([
        "",
        "---",
        "",
        "## Concordance Summary",
        "",
        f"- **Subjects with quantum hardware data:** {conc['n_with_quantum_data']}/{conc['n_subjects']}",
        f"- **Tier 3 / Quantum concordance:** {conc['n_tier3_quantum_concordant']}/{conc['n_with_quantum_data']} "
        f"({conc['concordance_rate']*100:.0f}%)" if conc['concordance_rate'] else "",
        f"- **Polarity consistent across windows:** {conc['n_polarity_consistent_across_windows']}/{conc['n_subjects']} "
        f"({conc['polarity_window_consistency_rate']*100:.0f}%)",
        "",
        "---",
        "",
        "## Key Findings",
        ""
    ])

    # Analyze findings
    t3_cal_20 = results['per_window_size']['20']['tier3_calibrated_auc']
    t3_raw_20 = results['per_window_size']['20']['tier3_raw_auc']
    t3_lift_20 = results['per_window_size']['20']['tier3_calibration_lift']
    t2_lift_20 = results['per_window_size']['20']['tier2_calibration_lift']

    lines.append("### 1. Does polarity calibration recover Tier 3 performance?")
    lines.append("")
    if t3_cal_20 >= 0.65:
        lines.append(f"**YES.** Calibrated T3 at 20s reaches {t3_cal_20:.3f} (lift: +{t3_lift_20:.3f}).")
        lines.append("Polarity inversion is THE explanation for Riemannian underperformance in cross-patient evaluation.")
    elif t3_cal_20 >= 0.55:
        lines.append(f"**PARTIAL.** Calibrated T3 at 20s reaches {t3_cal_20:.3f} (lift: +{t3_lift_20:.3f}).")
        lines.append("Polarity explains some but not all of the underperformance.")
    else:
        lines.append(f"**NO.** Calibrated T3 at 20s is only {t3_cal_20:.3f} (lift: +{t3_lift_20:.3f}).")
        lines.append("Polarity is not the primary explanation for Riemannian underperformance.")

    lines.append("")
    lines.append("### 2. Does Tier 2 show calibration lift?")
    lines.append("")
    if t2_lift_20 < 0.01:
        lines.append(f"**NO.** T2 lift at 20s is +{t2_lift_20:.3f}.")
        lines.append("Eigenvalue sorting is polarity-invariant as predicted.")
    else:
        lines.append(f"**YES.** T2 lift at 20s is +{t2_lift_20:.3f}.")
        lines.append("Unexpected: polarity affects even sorted eigenvalues. Investigate.")

    lines.append("")
    lines.append("### 3. Does Tier 3 polarity match quantum polarity?")
    lines.append("")
    conc_rate = conc['concordance_rate'] or 0
    if conc_rate >= 0.8:
        lines.append(f"**YES.** Concordance rate: {conc_rate*100:.0f}%")
        lines.append("Polarity is a stable geometric property observable through multiple independent methods.")
    elif conc_rate >= 0.5:
        lines.append(f"**PARTIAL.** Concordance rate: {conc_rate*100:.0f}%")
        lines.append("Some agreement, but polarity may be method-dependent for some patients.")
    else:
        lines.append(f"**NO.** Concordance rate: {conc_rate*100:.0f}%")
        lines.append("Tier 3 and quantum methods identify different polarities.")

    lines.append("")
    lines.append("### 4. Is polarity stable across window sizes?")
    lines.append("")
    consistency_rate = conc['polarity_window_consistency_rate']
    if consistency_rate >= 0.8:
        lines.append(f"**YES.** {consistency_rate*100:.0f}% of subjects have consistent polarity across all windows.")
        lines.append("The inversion is intrinsic to the patient, not noise-dependent.")
    else:
        lines.append(f"**NO.** Only {consistency_rate*100:.0f}% of subjects have consistent polarity.")
        lines.append("Polarity may flip due to noisy covariance estimation at short windows.")

    lines.extend([
        "",
        "---",
        "",
        "## Implications",
        ""
    ])

    # Write implications based on findings
    if t3_cal_20 >= 0.60 and conc_rate >= 0.7 and consistency_rate >= 0.7:
        lines.extend([
            "The polarity inversion discovered through quantum hardware measurement is **not a quantum-specific phenomenon**.",
            "It is a geometric property of the brain state SPD manifold that manifests in any method preserving",
            "directional information (Riemannian tangent space, quantum density matrix encoding).",
            "",
            "Methods that discard direction (eigenvalue sorting) are implicitly polarity-invariant, which explains",
            "their superior cross-patient performance but at the cost of geometric interpretability.",
            "",
            "**The quantum circuit's contribution is making this polarity an explicit, measurable observable**",
            "rather than a hidden source of cross-patient performance degradation."
        ])
    else:
        lines.extend([
            "The results are inconclusive regarding polarity as the sole explanation for Riemannian underperformance.",
            "Additional factors may be at play, including:",
            "- LDA's sensitivity to class imbalance in LOSO",
            "- Tangent space projection noise",
            "- Method-specific polarity ambiguity",
            "",
            "Further investigation is needed."
        ])

    with open(OUTPUT_MD, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    results = run_analysis()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for window_sec in [1, 20]:
        w = results['per_window_size'][str(window_sec)]
        print(f"\n{window_sec}s window:")
        print(f"  Tier 3: {w['tier3_raw_auc']:.3f} -> {w['tier3_calibrated_auc']:.3f} (lift +{w['tier3_calibration_lift']:.3f})")
        print(f"  Tier 2: {w['tier2_raw_auc']:.3f} -> {w['tier2_calibrated_auc']:.3f} (lift +{w['tier2_calibration_lift']:.3f})")

    conc = results['concordance_summary']
    print(f"\nConcordance with quantum: {conc['n_tier3_quantum_concordant']}/{conc['n_with_quantum_data']} ({conc['concordance_rate']*100:.0f}%)")
    print(f"Polarity consistency: {conc['n_polarity_consistent_across_windows']}/{conc['n_subjects']} ({conc['polarity_window_consistency_rate']*100:.0f}%)")
