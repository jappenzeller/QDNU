#!/usr/bin/env python3
"""
PROMPT 012 — Polarity-Calibrated Gap Model Input for Projection

Produces calibrated inputs for PROMPT 010 (hardware readiness projection).
Reprocesses existing per-patient results from PROMPT 011 to compute
polarity-corrected aggregate AUC values.

This is a data reprocessing task only — no circuits are run.
"""

import json
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "calibration"

# Classical ceiling from Tier 2 MAX results
CLASSICAL_CEILING = 0.7234


def load_json(path):
    """Load JSON file, return None if not found."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None


def main():
    print("=" * 70)
    print("PROMPT 012 — Polarity-Calibrated Gap Model Input")
    print("=" * 70)
    print()

    # Load patient profiles from PROMPT 011
    profiles_path = RESULTS_DIR / "patient_analysis" / "quantum_patient_profiles.json"
    profiles_data = load_json(profiles_path)
    if not profiles_data:
        print("ERROR: Could not load patient profiles. Run PROMPT 011 first.")
        return

    # Load simulation scaling data if available
    sim_scaling_path = RESULTS_DIR / "projection" / "simulation_scaling.json"
    sim_scaling = load_json(sim_scaling_path)

    # Load classical channel scaling
    classical_path = RESULTS_DIR / "scaling" / "classical_channel_scaling.json"
    classical_data = load_json(classical_path)

    # Extract per-subject data from patient profiles
    patient_profiles = profiles_data.get("patient_profiles", {})

    # Collect hardware subjects
    hw_subjects = {}
    inverted_subjects = []

    for subj_id, profile in patient_profiles.items():
        if not profile.get("hardware_runs"):
            continue

        # Get best hardware run
        best_hw_run = max(
            profile["hardware_runs"],
            key=lambda x: x.get("raw_auc") or 0
        )

        raw_auc = best_hw_run.get("raw_auc")
        if raw_auc is None:
            continue

        # Compute calibrated AUC
        calibrated_auc = max(raw_auc, 1.0 - raw_auc)
        polarity = "standard" if raw_auc >= 0.5 else "inverted"

        hw_subjects[subj_id] = {
            "raw": raw_auc,
            "calibrated": calibrated_auc,
            "polarity": polarity
        }

        if polarity == "inverted":
            inverted_subjects.append(subj_id)

    n_subjects = len(hw_subjects)
    n_inverted = len(inverted_subjects)

    print(f"Hardware subjects: {n_subjects}")
    print(f"Inverted subjects: {n_inverted}")
    print()

    # Compute aggregate AUCs
    if hw_subjects:
        raw_aucs = [s["raw"] for s in hw_subjects.values()]
        calibrated_aucs = [s["calibrated"] for s in hw_subjects.values()]

        raw_overall = sum(raw_aucs) / len(raw_aucs)
        calibrated_overall = sum(calibrated_aucs) / len(calibrated_aucs)
        calibration_lift = calibrated_overall - raw_overall
    else:
        raw_overall = calibrated_overall = calibration_lift = None

    # Compute gaps to classical ceiling
    raw_gap = CLASSICAL_CEILING - raw_overall if raw_overall else None
    calibrated_gap = CLASSICAL_CEILING - calibrated_overall if calibrated_overall else None
    gap_reduction = raw_gap - calibrated_gap if raw_gap and calibrated_gap else None

    # Build output
    output = {
        "timestamp": datetime.now().isoformat(),
        "classical_ceiling": CLASSICAL_CEILING,
        "ch8": {
            "raw_overall_auc": raw_overall,
            "calibrated_overall_auc": calibrated_overall,
            "calibration_lift": calibration_lift,
            "n_subjects": n_subjects,
            "n_inverted": n_inverted,
            "inverted_subjects": sorted(inverted_subjects),
            "per_subject": hw_subjects
        },
        "raw_gap_ch8": raw_gap,
        "calibrated_gap_ch8": calibrated_gap,
        "gap_reduction_from_calibration": gap_reduction,
        "future_run_requirements": {
            "store_sub_window_scores": True,
            "reason_sub_window": (
                "STD aggregation requires all 30 sub-window predictions per segment, "
                "not just MAX. Barachant showed STD > MAX (0.805 vs 0.776). "
                "STD is also naturally polarity-robust — it detects presence of "
                "transient regardless of direction."
            ),
            "store_per_patient_raw_scores": True,
            "reason_raw_scores": (
                "Enables exact calibrated overall AUC recomputation and ROC "
                "curve analysis with polarity correction applied."
            )
        },
        "projection_note": (
            "PROMPT 010 should produce two scenarios: "
            "Scenario A (Raw) uses raw hardware AUC for conservative projection. "
            "Scenario B (Calibrated) uses polarity-calibrated AUC for realistic "
            "projection assuming per-patient sign bit calibration."
        )
    }

    # Add simulation data if available
    if sim_scaling and "ch8" in sim_scaling:
        output["simulation_ch8_auc"] = sim_scaling["ch8"].get("overall_auc")
        output["simulation_gap_ch8"] = (
            CLASSICAL_CEILING - sim_scaling["ch8"]["overall_auc"]
            if sim_scaling["ch8"].get("overall_auc") else None
        )

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "calibrated_gap_inputs.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {output_path}")
    print()

    # Print summary
    print("=" * 70)
    print("Hardware CH8 AUC:")
    print("=" * 70)
    print(f"  Raw:        {raw_overall:.4f}  (gap to classical: {raw_gap:.4f})" if raw_overall else "  Raw:        N/A")
    print(f"  Calibrated: {calibrated_overall:.4f}  (gap to classical: {calibrated_gap:.4f})" if calibrated_overall else "  Calibrated: N/A")
    print(f"  Lift from polarity calibration: +{calibration_lift:.4f}" if calibration_lift else "  Lift: N/A")
    print(f"  Inverted patients: {n_inverted} / {n_subjects}")
    if inverted_subjects:
        print(f"  Inverted subject IDs: {', '.join(sorted(inverted_subjects))}")
    print()

    print("=" * 70)
    print("Impact on Projection:")
    print("=" * 70)
    print(f"  If PROMPT 010 uses raw AUC:        gap = {raw_gap:.4f}" if raw_gap else "  Raw gap: N/A")
    print(f"  If PROMPT 010 uses calibrated AUC: gap = {calibrated_gap:.4f}" if calibrated_gap else "  Calibrated gap: N/A")
    print(f"  Gap reduction from calibration:    {gap_reduction:.4f}" if gap_reduction else "  Gap reduction: N/A")
    print()

    if gap_reduction and gap_reduction > 0:
        pct_reduction = (gap_reduction / raw_gap) * 100 if raw_gap else 0
        print(f"  Polarity calibration reduces the gap by {pct_reduction:.1f}%")
        print("  Projected hardware readiness shifts EARLIER with calibration")
    print()

    print("=" * 70)
    print("Per-Subject Details:")
    print("=" * 70)
    print(f"{'Subject':<10} {'Raw AUC':<10} {'Cal AUC':<10} {'Polarity':<10}")
    print("-" * 40)
    for subj in sorted(hw_subjects.keys()):
        data = hw_subjects[subj]
        print(f"{subj:<10} {data['raw']:.4f}     {data['calibrated']:.4f}     {data['polarity']}")
    print()

    return output


if __name__ == "__main__":
    main()
