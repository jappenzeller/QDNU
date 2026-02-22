#!/usr/bin/env python3
"""
PROMPT 007 — Encoding Audit: Identify Prior Best Hardware Encoding

Audits all prior result files to identify which encoding produced the best
hardware AUC, and checks whether chb11 and chb21 (lowest performers in
PROMPT 006) were also low performers in prior runs.

This is a file audit only — no quantum circuits are run.
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent
ANALYSIS_RESULTS_DIR = BASE_DIR / "analysis_results"
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = RESULTS_DIR / "hardware_validation"


def load_json(path):
    """Load JSON file, return None if not found or invalid."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"  Warning: Could not load {path}: {e}")
        return None


def extract_quantum_result_info(path, data):
    """Extract standardized info from a quantum result file."""
    info = {
        "path": str(path.relative_to(BASE_DIR)),
        "backend": "not_logged",
        "encoding": "not_logged",
        "mitigation": "not_logged",
        "overall_auc": None,
        "per_subject": {},
        "timestamp": "not_logged",
        "n_qubits": None,
        "n_channels": None
    }

    # Extract overall AUC
    if "overall" in data and "auc" in data["overall"]:
        info["overall_auc"] = data["overall"]["auc"]
    elif "overall_auc" in data:
        info["overall_auc"] = data["overall_auc"]
    elif "ch8" in data and isinstance(data["ch8"], dict):
        info["overall_auc"] = data["ch8"].get("overall_auc")

    # Extract per-subject AUC
    if "per_subject" in data:
        for subj, subj_data in data["per_subject"].items():
            if isinstance(subj_data, dict) and "auc" in subj_data:
                info["per_subject"][subj] = subj_data["auc"]
            elif isinstance(subj_data, (int, float)):
                info["per_subject"][subj] = subj_data
    elif "ch8" in data and isinstance(data["ch8"], dict) and "per_subject" in data["ch8"]:
        for subj, subj_data in data["ch8"]["per_subject"].items():
            if isinstance(subj_data, dict) and "auc" in subj_data:
                info["per_subject"][subj] = subj_data["auc"]
            elif isinstance(subj_data, (int, float)):
                info["per_subject"][subj] = subj_data

    # Extract model params
    if "model_params" in data:
        params = data["model_params"]
        info["n_qubits"] = params.get("n_qubits")
        info["n_channels"] = params.get("n_channels")

        # Determine encoding from classifier name or method
        classifier = params.get("classifier", "")
        if "DualTemplate" in classifier:
            info["encoding"] = f"V2_band_power ({params.get('pn_method', '')})"
        elif "PLV-V3" in classifier:
            band = params.get("band_hz", data.get("band", ""))
            info["encoding"] = f"V3_PLV_{band}"
        elif "QPNN" in classifier:
            info["encoding"] = "V1_PN_dynamics"

    # Check for hardware-specific fields
    if "backend" in data:
        info["backend"] = data["backend"]
        if "ibm" in data["backend"].lower() or "torino" in data["backend"].lower():
            info["backend"] = f"HARDWARE: {data['backend']}"
        elif "statevector" in data["backend"].lower() or "simulation" in data["backend"].lower():
            info["backend"] = f"simulation: {data['backend']}"

    if "encoding_variant" in data:
        info["encoding"] = data["encoding_variant"]

    if "error_mitigation" in data:
        info["mitigation"] = data["error_mitigation"]

    if "shots" in data:
        info["shots"] = data["shots"]

    if "timestamp" in data:
        info["timestamp"] = data["timestamp"]
    elif "completed" in data:
        info["timestamp"] = data["completed"]

    return info


def scan_quantum_results():
    """Scan all directories for quantum result files."""
    all_results = []

    # Define quantum result files to check
    quantum_files = [
        # quantum_loso
        ANALYSIS_RESULTS_DIR / "quantum_loso" / "quantum_results.json",
        # quantum_loso_v2
        ANALYSIS_RESULTS_DIR / "quantum_loso_v2" / "quantum_results.json",
        # quantum_loso_v3
        ANALYSIS_RESULTS_DIR / "quantum_loso_v3" / "quantum_theta_results.json",
        ANALYSIS_RESULTS_DIR / "quantum_loso_v3" / "quantum_theta_alpha_results.json",
        ANALYSIS_RESULTS_DIR / "quantum_loso_v3" / "quantum_alpha_beta_results.json",
        # scaling
        RESULTS_DIR / "scaling" / "quantum_channel_scaling.json",
        # hardware validation
        RESULTS_DIR / "hardware_validation" / "ch8_heron_loso_results.json",
        # projection
        RESULTS_DIR / "projection" / "simulation_scaling.json",
    ]

    for qfile in quantum_files:
        if qfile.exists():
            print(f"Reading: {qfile}")
            data = load_json(qfile)
            if data:
                info = extract_quantum_result_info(qfile, data)
                all_results.append(info)

    return all_results


def classify_backend(result):
    """Classify whether a result is hardware or simulation."""
    backend = result.get("backend", "").lower()
    if "hardware" in backend or "ibm_torino" in backend or "heron" in backend:
        return "hardware"
    return "simulation"


def compile_patient_history(all_results, patient_id):
    """Compile AUC history for a specific patient across all runs."""
    history = []
    for result in all_results:
        if patient_id in result.get("per_subject", {}):
            history.append({
                "path": result["path"],
                "encoding": result["encoding"],
                "backend_type": classify_backend(result),
                "auc": result["per_subject"][patient_id]
            })
    return sorted(history, key=lambda x: x["auc"], reverse=True)


def find_best_hardware_result(all_results):
    """Find the best performing hardware result."""
    hardware_results = [r for r in all_results if classify_backend(r) == "hardware"]
    if not hardware_results:
        return None
    return max(hardware_results, key=lambda x: x.get("overall_auc") or 0)


def main():
    print("=" * 70)
    print("PROMPT 007 — Encoding Audit")
    print("=" * 70)
    print()

    # Scan all result files
    print("Scanning quantum result files...")
    all_results = scan_quantum_results()
    print(f"Found {len(all_results)} quantum result files")
    print()

    # Classify results
    hardware_results = [r for r in all_results if classify_backend(r) == "hardware"]
    simulation_results = [r for r in all_results if classify_backend(r) == "simulation"]

    print(f"Hardware runs: {len(hardware_results)}")
    print(f"Simulation runs: {len(simulation_results)}")
    print()

    # Compile chb11 and chb21 history
    print("Compiling patient histories...")
    chb11_history = compile_patient_history(all_results, "chb11")
    chb21_history = compile_patient_history(all_results, "chb21")

    # Find best hardware result
    best_hardware = find_best_hardware_result(all_results)

    # Find best simulation result
    best_simulation = max(
        simulation_results,
        key=lambda x: x.get("overall_auc") or 0
    ) if simulation_results else None

    # Collect all encoding variants found
    encoding_variants = list(set(r["encoding"] for r in all_results if r["encoding"] != "not_logged"))

    # Build output structure per PROMPT 007 spec
    output = {
        "audit_timestamp": datetime.now().isoformat(),
        "all_quantum_result_files": all_results,
        "best_hardware_result": {
            "path": best_hardware["path"] if best_hardware else None,
            "encoding": best_hardware["encoding"] if best_hardware else None,
            "mitigation": best_hardware.get("mitigation") if best_hardware else None,
            "overall_auc": best_hardware["overall_auc"] if best_hardware else None
        } if best_hardware else None,
        "best_simulation_result": {
            "path": best_simulation["path"] if best_simulation else None,
            "encoding": best_simulation["encoding"] if best_simulation else None,
            "overall_auc": best_simulation["overall_auc"] if best_simulation else None
        } if best_simulation else None,
        "chb11_history": chb11_history,
        "chb21_history": chb21_history,
        "encoding_variants_found": encoding_variants,
        "notes": []
    }

    # Add notes about findings
    if best_hardware:
        output["notes"].append(
            f"Only one hardware LOSO run exists: {best_hardware['path']} with AUC {best_hardware['overall_auc']:.4f}"
        )

    if chb11_history:
        hw_chb11 = [h for h in chb11_history if h["backend_type"] == "hardware"]
        sim_chb11 = [h for h in chb11_history if h["backend_type"] == "simulation"]
        if hw_chb11 and sim_chb11:
            best_hw = max(hw_chb11, key=lambda x: x["auc"])
            best_sim = max(sim_chb11, key=lambda x: x["auc"])
            output["notes"].append(
                f"chb11: Best simulation AUC {best_sim['auc']:.3f} ({best_sim['encoding']}), "
                f"hardware AUC {best_hw['auc']:.3f} ({best_hw['encoding']})"
            )

    if chb21_history:
        hw_chb21 = [h for h in chb21_history if h["backend_type"] == "hardware"]
        sim_chb21 = [h for h in chb21_history if h["backend_type"] == "simulation"]
        if hw_chb21 and sim_chb21:
            best_hw = max(hw_chb21, key=lambda x: x["auc"])
            best_sim = max(sim_chb21, key=lambda x: x["auc"])
            output["notes"].append(
                f"chb21: Best simulation AUC {best_sim['auc']:.3f} ({best_sim['encoding']}), "
                f"hardware AUC {best_hw['auc']:.3f} ({best_hw['encoding']})"
            )

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "encoding_audit.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Saved: {output_path}")
    print()

    # Print summary table
    print("=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print()
    print(f"{'File':<50} {'Backend':<15} {'Encoding':<25} {'Mitigation':<10} {'Overall AUC':<12}")
    print("-" * 100)

    for r in sorted(all_results, key=lambda x: x.get("overall_auc") or 0, reverse=True):
        path_short = r["path"][-45:] if len(r["path"]) > 45 else r["path"]
        backend_short = "HARDWARE" if classify_backend(r) == "hardware" else "simulation"
        encoding_short = r["encoding"][:22] if len(r["encoding"]) > 22 else r["encoding"]
        mitigation = r.get("mitigation", "n/a")[:8]
        auc = f"{r['overall_auc']:.4f}" if r["overall_auc"] else "N/A"
        print(f"{path_short:<50} {backend_short:<15} {encoding_short:<25} {mitigation:<10} {auc:<12}")

    print()
    print("=" * 100)
    print("CHB11 HISTORY (sorted by AUC descending)")
    print("=" * 100)
    for h in chb11_history:
        print(f"  {h['backend_type']:<10} {h['encoding']:<30} AUC: {h['auc']:.3f}")

    print()
    print("=" * 100)
    print("CHB21 HISTORY (sorted by AUC descending)")
    print("=" * 100)
    for h in chb21_history:
        print(f"  {h['backend_type']:<10} {h['encoding']:<30} AUC: {h['auc']:.3f}")

    print()
    print("=" * 100)
    print("KEY FINDINGS")
    print("=" * 100)
    for note in output["notes"]:
        print(f"  - {note}")
    print()

    return output


if __name__ == "__main__":
    main()
