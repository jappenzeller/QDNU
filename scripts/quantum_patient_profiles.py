#!/usr/bin/env python3
"""
PROMPT 011 — Per-Patient Quantum Performance Analysis with Polarity Detection

Compiles all per-subject quantum results across every hardware and simulation run
into a patient-level analysis. Identifies which patients produce coherent quantum
signal on real hardware, whether polarity inversion explains low-AUC patients,
and whether the simulation-hardware gap is uniform or patient-specific.

This is a file analysis and reporting task only. No quantum circuits are run.
"""

import json
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS_DIR = BASE_DIR / "analysis_results"
OUTPUT_DIR = RESULTS_DIR / "patient_analysis"


def load_json(path):
    """Load JSON file, return None if not found."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None


def compute_calibrated_auc(raw_auc):
    """Compute calibrated AUC and polarity."""
    if raw_auc is None:
        return None, None
    calibrated = max(raw_auc, 1.0 - raw_auc)
    polarity = "standard" if raw_auc >= 0.5 else "inverted"
    return calibrated, polarity


def extract_mandelbrot_direction(patient_data):
    """
    Extract Mandelbrot direction from patient statistics.

    Returns "ictal_inside" if ictal boundary_distance < interictal,
    "ictal_outside" otherwise.
    """
    if not patient_data or "metrics" not in patient_data:
        return None

    bd = patient_data["metrics"].get("Boundary Distance", {})
    ictal_mean = bd.get("ictal_mean")
    interictal_mean = bd.get("interictal_mean")

    if ictal_mean is None or interictal_mean is None:
        return None

    # More negative = deeper inside Mandelbrot set
    # Less negative/more positive = closer to or outside boundary
    return "ictal_inside" if ictal_mean < interictal_mean else "ictal_outside"


def classify_patient(calibrated_hw_auc, calibrated_sim_auc, polarity):
    """
    Classify patient based on calibrated AUC values.

    STRONG          — calibrated hardware AUC > 0.65, standard polarity
    STRONG-INVERTED — calibrated hardware AUC > 0.65, inverted polarity
    MODERATE        — calibrated hardware AUC 0.55-0.65
    WEAK            — calibrated hardware AUC < 0.55, simulation calibrated > 0.65
    NOISE           — calibrated hardware AUC < 0.55 AND simulation calibrated < 0.55
    """
    if calibrated_hw_auc is None:
        # No hardware data - classify based on simulation only
        if calibrated_sim_auc and calibrated_sim_auc > 0.65:
            return "UNTESTED-PROMISING"
        return "UNTESTED"

    if calibrated_hw_auc > 0.65:
        if polarity == "inverted":
            return "STRONG-INVERTED"
        return "STRONG"
    elif calibrated_hw_auc >= 0.55:
        return "MODERATE"
    else:
        # calibrated_hw_auc < 0.55
        if calibrated_sim_auc and calibrated_sim_auc > 0.65:
            return "WEAK"
        return "NOISE"


def main():
    print("=" * 70)
    print("PROMPT 011 — Per-Patient Quantum Performance Analysis")
    print("=" * 70)
    print()

    # Load encoding audit results
    encoding_audit_path = RESULTS_DIR / "hardware_validation" / "encoding_audit.json"
    encoding_audit = load_json(encoding_audit_path)
    if not encoding_audit:
        print("ERROR: Could not load encoding audit. Run PROMPT 007 first.")
        return

    # Load classical baseline
    classical_path = RESULTS_DIR / "scaling" / "classical_channel_scaling.json"
    classical_data = load_json(classical_path)
    classical_ch8 = {}
    if classical_data and "ch8" in classical_data:
        for subj, data in classical_data["ch8"]["per_subject"].items():
            classical_ch8[subj] = data["auc"] if isinstance(data, dict) else data

    # Load Mandelbrot/statistics data
    stats_path = ANALYSIS_DIR / "statistics.json"
    stats_data = load_json(stats_path)

    # Map Patient_N to chbNN format
    mandelbrot_by_patient = {}
    if stats_data and "per_patient" in stats_data:
        # The statistics.json uses Patient_1, Patient_2, etc.
        # We need to map these to chb01, chb02, etc.
        # Based on typical CHB-MIT ordering: Patient_1 = chb01, Patient_2 = chb02, etc.
        patient_mapping = {
            "Patient_1": "chb01",
            "Patient_2": "chb02",
            "Patient_3": "chb03",
            "Patient_4": "chb04",
            "Patient_5": "chb05",
            "Patient_6": "chb06",
            "Patient_7": "chb07",
            "Patient_8": "chb08",
        }
        for patient_key, patient_data in stats_data["per_patient"].items():
            chb_id = patient_mapping.get(patient_key)
            if chb_id:
                mandelbrot_by_patient[chb_id] = extract_mandelbrot_direction(patient_data)

    # Collect all subjects across all quantum results
    all_subjects = set()
    for result in encoding_audit["all_quantum_result_files"]:
        all_subjects.update(result.get("per_subject", {}).keys())

    print(f"Found {len(all_subjects)} unique subjects across quantum results")
    print()

    # Build patient profiles
    profiles = {}

    for subject in sorted(all_subjects):
        profile = {
            "subject": subject,
            "classical_ch8_auc": classical_ch8.get(subject),
            "hardware_runs": [],
            "simulation_runs": [],
            "best_hardware_raw_auc": None,
            "best_hardware_calibrated_auc": None,
            "best_simulation_calibrated_auc": None,
            "hw_sim_gap": None,
            "polarity_consistent_across_runs": True,
            "mandelbrot_direction": mandelbrot_by_patient.get(subject, "not_available"),
            "polarity_matches_mandelbrot": None,
            "classification": None
        }

        polarities_seen = set()

        for result in encoding_audit["all_quantum_result_files"]:
            per_subject = result.get("per_subject", {})
            if subject not in per_subject:
                continue

            raw_auc = per_subject[subject]
            calibrated_auc, polarity = compute_calibrated_auc(raw_auc)

            if polarity:
                polarities_seen.add(polarity)

            backend = result.get("backend", "")
            is_hardware = "hardware" in backend.lower() or "ibm_torino" in backend.lower()

            run_info = {
                "encoding": result.get("encoding", "not_logged"),
                "backend": result.get("backend", "not_logged"),
                "raw_auc": raw_auc,
                "calibrated_auc": calibrated_auc,
                "polarity": polarity
            }

            if is_hardware:
                run_info["beats_classical"] = (
                    raw_auc > classical_ch8.get(subject, 0)
                    if raw_auc and classical_ch8.get(subject) else None
                )
                profile["hardware_runs"].append(run_info)
            else:
                profile["simulation_runs"].append(run_info)

        # Compute best values
        if profile["hardware_runs"]:
            best_hw = max(profile["hardware_runs"], key=lambda x: x.get("raw_auc") or 0)
            profile["best_hardware_raw_auc"] = best_hw["raw_auc"]
            profile["best_hardware_calibrated_auc"] = best_hw["calibrated_auc"]
            hw_polarity = best_hw["polarity"]
        else:
            hw_polarity = None

        if profile["simulation_runs"]:
            best_sim = max(profile["simulation_runs"], key=lambda x: x.get("calibrated_auc") or 0)
            profile["best_simulation_calibrated_auc"] = best_sim["calibrated_auc"]

        # Compute gap
        if profile["best_hardware_calibrated_auc"] and profile["best_simulation_calibrated_auc"]:
            profile["hw_sim_gap"] = (
                profile["best_simulation_calibrated_auc"] -
                profile["best_hardware_calibrated_auc"]
            )

        # Check polarity consistency
        profile["polarity_consistent_across_runs"] = len(polarities_seen) <= 1

        # Check polarity-Mandelbrot concordance
        if hw_polarity and profile["mandelbrot_direction"] != "not_available":
            # Hypothesis: inverted polarity correlates with ictal_inside
            # (or vice versa - we'll check empirically)
            profile["polarity_matches_mandelbrot"] = (
                (hw_polarity == "inverted" and profile["mandelbrot_direction"] == "ictal_inside") or
                (hw_polarity == "standard" and profile["mandelbrot_direction"] == "ictal_outside")
            )

        # Classify patient
        profile["classification"] = classify_patient(
            profile["best_hardware_calibrated_auc"],
            profile["best_simulation_calibrated_auc"],
            hw_polarity
        )

        profiles[subject] = profile

    # Compute aggregate statistics
    hw_subjects = [p for p in profiles.values() if p["hardware_runs"]]

    if hw_subjects:
        raw_aucs = [p["best_hardware_raw_auc"] for p in hw_subjects if p["best_hardware_raw_auc"]]
        calibrated_aucs = [p["best_hardware_calibrated_auc"] for p in hw_subjects if p["best_hardware_calibrated_auc"]]

        raw_overall = sum(raw_aucs) / len(raw_aucs) if raw_aucs else None
        calibrated_overall = sum(calibrated_aucs) / len(calibrated_aucs) if calibrated_aucs else None

        n_inverted = sum(1 for p in hw_subjects
                        if p["hardware_runs"] and
                        max(p["hardware_runs"], key=lambda x: x.get("raw_auc") or 0).get("polarity") == "inverted")

        calibration_lift = (calibrated_overall - raw_overall) if raw_overall and calibrated_overall else None
    else:
        raw_overall = calibrated_overall = n_inverted = calibration_lift = None

    # Mandelbrot concordance
    mandelbrot_subjects = [p for p in profiles.values()
                          if p["polarity_matches_mandelbrot"] is not None]
    concordant = sum(1 for p in mandelbrot_subjects if p["polarity_matches_mandelbrot"])
    total_mandelbrot = len(mandelbrot_subjects)
    concordance_rate = concordant / total_mandelbrot if total_mandelbrot > 0 else None

    # Classification summary
    classification_counts = {}
    classification_aucs = {}
    for p in profiles.values():
        cat = p["classification"]
        if cat not in classification_counts:
            classification_counts[cat] = 0
            classification_aucs[cat] = []
        classification_counts[cat] += 1
        if p["best_hardware_calibrated_auc"]:
            classification_aucs[cat].append(p["best_hardware_calibrated_auc"])

    # Build output
    output = {
        "audit_timestamp": datetime.now().isoformat(),
        "aggregate": {
            "raw_overall_hw_auc": raw_overall,
            "calibrated_overall_auc": calibrated_overall,
            "n_hw_subjects": len(hw_subjects),
            "n_inverted": n_inverted,
            "calibration_lift": calibration_lift,
        },
        "mandelbrot_concordance": {
            "patients_in_both_analyses": total_mandelbrot,
            "polarity_matches_direction": concordant,
            "concordance_rate": concordance_rate,
            "interpretation": (
                "concordant" if concordance_rate and concordance_rate > 0.7 else
                "discordant" if concordance_rate and concordance_rate < 0.3 else
                "inconclusive"
            ) if concordance_rate else "insufficient_data"
        },
        "classification_summary": {
            cat: {
                "n": classification_counts[cat],
                "avg_calibrated_auc": (sum(classification_aucs[cat]) / len(classification_aucs[cat])
                                       if classification_aucs[cat] else None)
            }
            for cat in classification_counts
        },
        "patient_profiles": profiles,
        "data_sparsity_notes": [
            f"Hardware runs per patient: min=0, max={max(len(p['hardware_runs']) for p in profiles.values())}",
            f"Subjects with hardware data: {len(hw_subjects)} / {len(profiles)}",
            f"Single-run hardware patients: {sum(1 for p in hw_subjects if len(p['hardware_runs']) == 1)}"
        ]
    }

    # Save JSON output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "quantum_patient_profiles.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {json_path}")

    # Generate markdown report
    # Format values for markdown
    raw_overall_str = f"{raw_overall:.4f}" if raw_overall else "N/A"
    calibrated_overall_str = f"{calibrated_overall:.4f}" if calibrated_overall else "N/A"
    inverted_str = f"{n_inverted} / {len(hw_subjects)}" if hw_subjects else "N/A"
    lift_str = f"+{calibration_lift:.4f}" if calibration_lift and calibration_lift > 0 else (f"{calibration_lift:.4f}" if calibration_lift else "N/A")

    md_lines = [
        "## Per-Patient Quantum Performance — Polarity-Aware Analysis",
        "",
        "### Aggregate Impact of Polarity Calibration",
        "| Metric                 | Value  |",
        "|------------------------|--------|",
        f"| Raw overall HW AUC     | {raw_overall_str} |",
        f"| Calibrated overall AUC | {calibrated_overall_str} |",
        f"| N patients inverted    | {inverted_str} |",
        f"| Calibration lift       | {lift_str} |",
        "",
        "### Mandelbrot Concordance",
        f"Patients in both analyses: {total_mandelbrot}",
        f"Polarity matches Mandelbrot direction: {concordant} / {total_mandelbrot} ({concordance_rate*100:.0f}%)" if concordance_rate else "Polarity matches Mandelbrot direction: insufficient data",
        f"Interpretation: {output['mandelbrot_concordance']['interpretation']}",
        "",
        "### Patient Classification Summary",
        "| Category         | N | Avg Calibrated AUC | Notes |",
        "|------------------|---|--------------------|-------|",
    ]

    category_order = ["STRONG", "STRONG-INVERTED", "MODERATE", "WEAK", "NOISE", "UNTESTED-PROMISING", "UNTESTED"]
    for cat in category_order:
        if cat in classification_counts:
            avg = output["classification_summary"][cat]["avg_calibrated_auc"]
            avg_str = f"{avg:.3f}" if avg else "N/A"
            note = "polarity flip needed" if "INVERTED" in cat else ("hardware-limited" if cat == "WEAK" else "")
            md_lines.append(f"| {cat:<16} | {classification_counts[cat]} | {avg_str:<18} | {note} |")

    md_lines.extend([
        "",
        "### Per-Patient Table (sorted by calibrated HW AUC descending)",
        "| Subject | Classical | Raw HW | Cal HW | Polarity | Mandelbrot | Category |",
        "|---------|-----------|--------|--------|----------|------------|----------|",
    ])

    # Sort by calibrated HW AUC descending
    sorted_profiles = sorted(
        profiles.values(),
        key=lambda p: p["best_hardware_calibrated_auc"] or 0,
        reverse=True
    )

    for p in sorted_profiles:
        classical = f"{p['classical_ch8_auc']:.3f}" if p['classical_ch8_auc'] else "N/A"
        raw_hw = f"{p['best_hardware_raw_auc']:.3f}" if p['best_hardware_raw_auc'] else "N/A"
        cal_hw = f"{p['best_hardware_calibrated_auc']:.3f}" if p['best_hardware_calibrated_auc'] else "N/A"

        # Get polarity from best hardware run
        if p["hardware_runs"]:
            best_hw_run = max(p["hardware_runs"], key=lambda x: x.get("raw_auc") or 0)
            pol = "inv" if best_hw_run.get("polarity") == "inverted" else "std"
        else:
            pol = "N/A"

        mand = p["mandelbrot_direction"][:10] if p["mandelbrot_direction"] != "not_available" else "N/A"
        cat = p["classification"][:8] if p["classification"] else "N/A"

        md_lines.append(f"| {p['subject']:<7} | {classical:<9} | {raw_hw:<6} | {cal_hw:<6} | {pol:<8} | {mand:<10} | {cat:<8} |")

    md_lines.extend([
        "",
        "### Data Sparsity Notes",
    ])
    for note in output["data_sparsity_notes"]:
        md_lines.append(f"- {note}")

    md_path = OUTPUT_DIR / "quantum_patient_profiles.md"
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))
    print(f"Saved: {md_path}")
    print()

    # Print summary to stdout
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("Aggregate Impact of Polarity Calibration:")
    print(f"  Raw overall HW AUC:     {raw_overall:.4f}" if raw_overall else "  Raw overall HW AUC:     N/A")
    print(f"  Calibrated overall AUC: {calibrated_overall:.4f}" if calibrated_overall else "  Calibrated overall AUC: N/A")
    print(f"  N patients inverted:    {n_inverted} / {len(hw_subjects)}" if hw_subjects else "  N patients inverted:    N/A")
    print(f"  Calibration lift:       +{calibration_lift:.4f}" if calibration_lift and calibration_lift > 0 else f"  Calibration lift:       {calibration_lift:.4f}" if calibration_lift else "  Calibration lift:       N/A")
    print()
    print("Mandelbrot Concordance:")
    print(f"  Patients in both analyses: {total_mandelbrot}")
    if concordance_rate:
        print(f"  Polarity matches direction: {concordant} / {total_mandelbrot} ({concordance_rate*100:.0f}%)")
    print(f"  Interpretation: {output['mandelbrot_concordance']['interpretation']}")
    print()
    print("Classification Summary:")
    for cat in category_order:
        if cat in classification_counts:
            avg = output["classification_summary"][cat]["avg_calibrated_auc"]
            print(f"  {cat:<18}: {classification_counts[cat]} patients, avg calibrated AUC = {avg:.3f}" if avg else f"  {cat:<18}: {classification_counts[cat]} patients")
    print()

    return output


if __name__ == "__main__":
    main()
