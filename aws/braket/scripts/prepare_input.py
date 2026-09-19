"""
Build the Hybrid Job input bundle.

Two modes:

  --mode synthetic
      Generate a small fixed parameter set per patient. Used for validating
      the end-to-end Braket plumbing without needing the CHB-MIT EEG dataset
      or the PLV pipeline. Parameters are deterministic per patient_id.

  --mode plv  (TODO: not yet wired)
      Read raw CHB-MIT EEG segments and derive (a, b, c) using
      extract_plv_params() from scripts/quantum_20s_hardware.py. This is the
      scientifically meaningful path; deferred until the smoke test passes.

Writes:
    aws/braket/job/input_data/config.json
    aws/braket/job/input_data/params.json
    aws/braket/job/input_data/templates.json   (per-patient ictal/interictal
                                                count templates for Hellinger
                                                fidelity scoring)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = REPO_ROOT / "aws" / "braket" / "job" / "input_data"

PRIORITY_PATIENTS = ("chb01", "chb03", "chb05", "chb07", "chb11", "chb14", "chb21")
WINDOW_STATES = ("interictal", "preictal", "ictal")
DEFAULT_NUM_CHANNELS = 8


def _synthetic_window(
    patient_id: str, window_id: str, state: str, num_channels: int, seed: int
) -> dict[str, Any]:
    """Deterministic per-patient synthetic (a, b, c). NOT scientifically meaningful;
    intended for end-to-end plumbing validation only.

    b is set to pi/4 here (not the canonical pi) so the synthetic E-channel
    observable carries information. With b = pi the E sandwich collapses to
    a diagonal operator on |0> and <Z_E> = +1.0 exactly. The real-data mode
    will derive b from the PLV pipeline; smoke-test mode picks pi/4 to keep
    both qubits' observables non-trivial for dashboard validation.
    """
    import random

    rng = random.Random(seed)
    channels = []
    for k in range(num_channels):
        a = max(0.05, min(0.95, 0.2 + 0.1 * k + rng.uniform(-0.05, 0.05)))
        b = math.pi / 4.0
        c = max(0.05, min(0.95, 0.6 - 0.05 * k + rng.uniform(-0.05, 0.05)))
        channels.append({"a": a, "b": b, "c": c})
    return {
        "patient_id": patient_id,
        "window_id": window_id,
        "state": state,
        "channels": channels,
    }


def _build_synthetic(
    patients: tuple[str, ...], num_channels: int
) -> dict[str, Any]:
    windows: list[dict[str, Any]] = []
    for patient_id in patients:
        # A stable seed per patient makes runs comparable across backends.
        base_seed = abs(hash(patient_id)) % (2**31)
        for i, state in enumerate(WINDOW_STATES):
            window_id = f"{patient_id}_w{i:02d}_{state}"
            windows.append(
                _synthetic_window(
                    patient_id, window_id, state, num_channels, base_seed + i
                )
            )
    return {"windows": windows}


def _synthetic_template(
    patient_id: str, state: str, num_qubits: int, seed: int
) -> dict[str, float]:
    """Generate a sparse per-patient count template (probability per bitstring).

    Real templates come from aggregating measurement counts across training
    windows of the same state. For the smoke test we sample a small number
    of bitstrings with Dirichlet-allocated probabilities; the shape is
    representative of a sparse 17-qubit measurement distribution but the
    values are not scientifically meaningful.
    """
    import random

    rng = random.Random(seed)
    # Sample ~64 random bitstrings to populate; the rest of the 2^17 space
    # is implicit zeros. State-dependent variation gives the two templates
    # different fingerprints.
    n_support = 64
    # Bias the bitstring sampling so ictal and interictal templates
    # diverge for the same patient.
    bias = 0.5 if state == "interictal" else 0.7
    bitstrings = []
    for _ in range(n_support):
        bs = "".join("1" if rng.random() < bias else "0" for _ in range(num_qubits))
        bitstrings.append(bs)
    # Dirichlet-like weights via random gamma
    weights = [rng.gammavariate(1.0, 1.0) for _ in bitstrings]
    total = sum(weights)
    template = {}
    for bs, w in zip(bitstrings, weights):
        # Aggregate duplicates by adding their probability mass.
        template[bs] = template.get(bs, 0.0) + w / total
    return template


def _build_synthetic_templates(
    patients: tuple[str, ...], num_qubits: int
) -> dict[str, dict[str, dict[str, float]]]:
    """Returns {patient_id: {ictal: {bitstr: prob}, interictal: {bitstr: prob}}}."""
    templates: dict[str, dict[str, dict[str, float]]] = {}
    for patient_id in patients:
        base_seed = abs(hash(patient_id + "_template")) % (2**31)
        templates[patient_id] = {
            "ictal": _synthetic_template(patient_id, "ictal", num_qubits, base_seed),
            "interictal": _synthetic_template(
                patient_id, "interictal", num_qubits, base_seed + 1
            ),
        }
    return templates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("synthetic", "plv"),
        default="synthetic",
        help="synthetic = smoke test; plv = real data (not yet wired)",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--patients",
        nargs="+",
        default=list(PRIORITY_PATIENTS),
        help="patient IDs to include",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        default=DEFAULT_NUM_CHANNELS,
        help="channels per window (synthetic mode only)",
    )
    parser.add_argument(
        "--shots", type=int, default=1024, help="recorded in config; not used here"
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "plv":
        raise SystemExit(
            "plv mode is not yet wired. Use --mode synthetic for the smoke test, "
            "or follow scripts/quantum_20s_hardware.py:extract_plv_params() to "
            "build the real bundle once the EEG path is set up."
        )

    params = _build_synthetic(tuple(args.patients), args.num_channels)
    num_qubits = 2 * args.num_channels + 1  # ancilla + 2 per channel
    templates = _build_synthetic_templates(tuple(args.patients), num_qubits)
    config = {
        "schema_version": "v2",
        "mode": args.mode,
        "source_artifact": "synthetic",
        "patients": list(args.patients),
        "num_channels": args.num_channels,
        "num_qubits": num_qubits,
        "shots_default": args.shots,
        "encoding": {
            "a": "synthetic, sweeps 0.2..0.95 across channels",
            "b": "pi/4 (smoke-test choice; canonical real-data value is pi but that collapses <Z_E>)",
            "c": "synthetic, sweeps 0.6..0.05 across channels",
        },
        "templates": "synthetic Dirichlet-sampled sparse 17-qubit distributions",
        "warning": "Synthetic parameters and templates; not derived from EEG. For plumbing validation only.",
    }

    (args.output / "config.json").write_text(json.dumps(config, indent=2))
    (args.output / "params.json").write_text(json.dumps(params, indent=2))
    (args.output / "templates.json").write_text(json.dumps(templates, indent=2))

    n_windows = len(params["windows"])
    n_channels_total = sum(len(w["channels"]) for w in params["windows"])
    n_templates = sum(len(t) for t in templates.values())
    log.info(
        "wrote %d files (%d windows, %d total channels, %d templates, mode=%s)",
        3,
        n_windows,
        n_channels_total,
        n_templates,
        args.mode,
    )


if __name__ == "__main__":
    main()
