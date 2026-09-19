"""
PROMPT AWS-001 - Braket local-simulator equivalence check.

Loads chb01's first ictal segment, extracts (a, b, c) per channel using the
existing PLV pipeline, builds the canonical 17-qubit A-Gate via the existing
multichannel circuit factory, and runs the SAME Qiskit QuantumCircuit through
two simulators:
    - Qiskit Aer (existing local reference)
    - BraketLocalBackend via qiskit-braket-provider

Compares <Z> on ancilla + each E + each I qubit, plus total variation distance
between the count distributions. Writes a stable-schema JSON to
results/braket/local_equivalence_chb01.json.

No AWS spend. No network calls. No source modifications outside scripts/braket/
and requirements-hardware.txt.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# braket-default-simulator's numba-jit'd kernel cannot handle reshape on
# >=17-qubit state tensors. Disable numba for the duration of this script.
# Must be set before any 'braket' import resolves the JIT-decorated kernels.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np

# Preload the Braket SDK and qiskit-braket-provider BEFORE adding REPO_ROOT/scripts
# to sys.path. Otherwise the local 'scripts/braket/' package shadows the
# installed 'braket' SDK when qiskit_braket_provider imports it.
import braket  # noqa: F401  (force resolution to installed package)
import qiskit_braket_provider  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "QA1"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sagemaker.train_chbmit import (  # noqa: E402
    extract_segments_for_subject,
)
from QA1.multichannel_circuit import (  # noqa: E402
    create_multichannel_circuit,
    get_qubit_indices,
    add_measurements,
)
from run_quantum_loso_v3 import (  # noqa: E402
    DATA_DIR,
    QUANTUM_CHANNELS,
    extract_plv_params,
    load_eeg_segment,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


DEFAULT_SHOTS = 4096
SUBJECT = "chb01"
WINDOW_S = 1.95
THRESHOLD_Z = 0.02
THRESHOLD_TVD = 0.05

# Output path is selected at runtime: v1 (no adapter) vs v2 (adapter).
OUTPUT_PATH_V1 = REPO_ROOT / "results" / "braket" / "local_equivalence_chb01.json"
OUTPUT_PATH_V2 = REPO_ROOT / "results" / "braket" / "local_equivalence_chb01_v2.json"


def _load_first_ictal(subject: str) -> tuple[np.ndarray, str]:
    """Return (eeg_array, segment_id) for the subject's first ictal segment.

    Uses the canonical 8-channel montage. Raises RuntimeError if no ictal
    segment can be loaded.
    """
    subject_dir = DATA_DIR / subject
    segments = extract_segments_for_subject(subject_dir)
    ictal = [s for s in segments if s.label == "ictal"]
    if not ictal:
        raise RuntimeError(f"no ictal segments for {subject}")

    for seg in ictal:
        eeg = load_eeg_segment(seg, subject_dir, QUANTUM_CHANNELS)
        if eeg is not None:
            seg_id = f"{seg.source_file}#start={seg.start_sec:.1f}s"
            return eeg, seg_id

    raise RuntimeError(f"failed to load any ictal segment for {subject}")


def _expectation_z(counts: dict[str, int], qubit_index: int, n_qubits: int) -> float:
    """Compute <Z> on a single qubit from a Qiskit-style counts dict.

    Qiskit bitstring convention: leftmost char = highest qubit index. So
    bitstring[0] corresponds to qubit (n_qubits - 1), bitstring[-1] to qubit 0.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    weighted = 0.0
    for bitstring, n in counts.items():
        # Strip whitespace (Qiskit can join multiple registers with spaces)
        bs = bitstring.replace(" ", "")
        if len(bs) != n_qubits:
            raise ValueError(
                f"bitstring length {len(bs)} != n_qubits {n_qubits}: {bitstring!r}"
            )
        bit = bs[n_qubits - 1 - qubit_index]
        weighted += n * (1.0 if bit == "0" else -1.0)
    return weighted / total


def _total_variation_distance(
    counts_a: dict[str, int], counts_b: dict[str, int]
) -> float:
    """TVD between two count distributions. Returns value in [0, 1]."""
    total_a = sum(counts_a.values()) or 1
    total_b = sum(counts_b.values()) or 1
    keys = set(counts_a) | set(counts_b)
    s = 0.0
    for k in keys:
        pa = counts_a.get(k, 0) / total_a
        pb = counts_b.get(k, 0) / total_b
        s += abs(pa - pb)
    return 0.5 * s


def _run_aer(circuit, shots: int, seed: int) -> dict[str, int]:
    """Run on Qiskit AerSimulator with seeding."""
    from qiskit_aer import AerSimulator

    sim = AerSimulator(seed_simulator=seed)
    transpiled = sim.run(circuit, shots=shots, seed_simulator=seed)
    return transpiled.result().get_counts()


def _run_braket_local(circuit, shots: int) -> dict[str, int]:
    """Run on BraketLocalBackend via qiskit-braket-provider."""
    from qiskit_braket_provider import BraketLocalBackend

    backend = BraketLocalBackend()
    job = backend.run(circuit, shots=shots)
    return job.result().get_counts()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--use-adapter",
        action="store_true",
        help=(
            "Apply scripts/braket/braket_adapter.py before submitting to "
            "BraketLocalBackend. Output goes to ..._v2.json instead of v1."
        ),
    )
    ap.add_argument(
        "--shots",
        type=int,
        default=DEFAULT_SHOTS,
        help=f"shot count per backend (default {DEFAULT_SHOTS})",
    )
    args = ap.parse_args()
    shots = args.shots

    log.info("loading %s first ictal segment via canonical pipeline", SUBJECT)
    eeg, seg_id = _load_first_ictal(SUBJECT)
    log.info("loaded segment %s shape=%s", seg_id, eeg.shape)

    log.info("extracting V3 PLV params (theta-alpha 4-13 Hz)")
    params = extract_plv_params(eeg, fs=256.0, band=(4, 13))
    log.info("got %d channel params", len(params))

    num_channels = len(QUANTUM_CHANNELS)
    n_qubits = 2 * num_channels + 1
    qmap = get_qubit_indices(num_channels)

    log.info("building canonical multichannel circuit (n_qubits=%d)", n_qubits)
    base_circuit = create_multichannel_circuit(params)
    # Single circuit, measured. Both backends consume the same QuantumCircuit.
    measured_aer = add_measurements(
        copy.deepcopy(base_circuit), num_channels, measure_all=True
    )

    unitary_equivalence: dict[str, Any] | None = None
    if args.use_adapter:
        from braket_adapter import prepare_circuit_for_braket
        from qiskit.quantum_info import Statevector

        log.info("applying braket_adapter.prepare_circuit_for_braket")
        measured_braket = prepare_circuit_for_braket(copy.deepcopy(measured_aer))

        # Prove unitary equivalence: original Qiskit circuit (no measurements)
        # vs decomposed Qiskit circuit (no measurements). Bypasses sampling
        # entirely so the result is independent of shot noise.
        log.info("computing statevector equivalence (no sampling)")
        sv_orig = Statevector.from_instruction(base_circuit)
        sv_decomp = Statevector.from_instruction(
            prepare_circuit_for_braket(copy.deepcopy(base_circuit))
        )
        fid = float(abs(sv_orig.inner(sv_decomp)) ** 2)
        ideal_tvd = float(
            0.5 * np.sum(np.abs(np.abs(sv_orig.data) ** 2 - np.abs(sv_decomp.data) ** 2))
        )
        unitary_equivalence = {
            "fidelity": fid,
            "ideal_tvd_probabilities": ideal_tvd,
            "passes": fid > 0.9999 and ideal_tvd < 1e-10,
        }
        log.info(
            "unitary equivalence: fidelity=%.10f ideal_tvd=%.2e",
            fid,
            ideal_tvd,
        )
    else:
        measured_braket = measured_aer

    log.info("running Aer (shots=%d)", shots)
    aer_counts = _run_aer(measured_aer, shots, seed=42)

    log.info("running BraketLocalBackend (shots=%d)", shots)
    braket_counts = _run_braket_local(measured_braket, shots)

    # Per-qubit <Z>
    z_anc_aer = _expectation_z(aer_counts, qmap["ancilla"], n_qubits)
    z_anc_braket = _expectation_z(braket_counts, qmap["ancilla"], n_qubits)
    z_e_aer = [_expectation_z(aer_counts, q, n_qubits) for q in qmap["E_qubits"]]
    z_e_braket = [_expectation_z(braket_counts, q, n_qubits) for q in qmap["E_qubits"]]
    z_i_aer = [_expectation_z(aer_counts, q, n_qubits) for q in qmap["I_qubits"]]
    z_i_braket = [_expectation_z(braket_counts, q, n_qubits) for q in qmap["I_qubits"]]

    z_anc_diff = abs(z_anc_aer - z_anc_braket)
    z_e_diff_max = max(abs(a - b) for a, b in zip(z_e_aer, z_e_braket))
    z_i_diff_max = max(abs(a - b) for a, b in zip(z_i_aer, z_i_braket))
    tvd = _total_variation_distance(aer_counts, braket_counts)

    pass_z = (
        z_anc_diff < THRESHOLD_Z
        and z_e_diff_max < THRESHOLD_Z
        and z_i_diff_max < THRESHOLD_Z
    )
    pass_tvd = tvd < THRESHOLD_TVD
    overall_pass = bool(pass_z and pass_tvd)

    output: dict[str, Any] = {
        "prompt_id": "AWS-001",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "subject": SUBJECT,
        "segment_id": seg_id,
        "channels": num_channels,
        "window_s": WINDOW_S,
        "shots": shots,
        "encoding": "V3_PLV_theta_alpha",
        "params": [[float(a), float(b), float(c)] for (a, b, c) in params],
        "aer": {
            "counts": {k: int(v) for k, v in aer_counts.items()},
            "z_ancilla": float(z_anc_aer),
            "z_e": [float(x) for x in z_e_aer],
            "z_i": [float(x) for x in z_i_aer],
        },
        "braket_local": {
            "counts": {k: int(v) for k, v in braket_counts.items()},
            "z_ancilla": float(z_anc_braket),
            "z_e": [float(x) for x in z_e_braket],
            "z_i": [float(x) for x in z_i_braket],
        },
        "diff": {
            "z_ancilla_abs": float(z_anc_diff),
            "z_e_max_abs": float(z_e_diff_max),
            "z_i_max_abs": float(z_i_diff_max),
            "tvd": float(tvd),
        },
        "pass": overall_pass,
        "device": "BraketLocalBackend",
        "notes": (
            "Equivalence test: same Qiskit QuantumCircuit object run on Aer "
            "and on BraketLocalBackend via qiskit-braket-provider. Threshold "
            f"z_abs<{THRESHOLD_Z}, tvd<{THRESHOLD_TVD}."
        ),
    }

    output_path = OUTPUT_PATH_V2 if args.use_adapter else OUTPUT_PATH_V1
    output["adapter"] = "braket_adapter.prepare_circuit_for_braket" if args.use_adapter else None
    if unitary_equivalence is not None:
        output["unitary_equivalence"] = unitary_equivalence
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    log.info("wrote %s", output_path)

    # One-screen summary
    print()
    print("=" * 76)
    print(f"AWS-001 Braket Local Equivalence Check  --  {SUBJECT}/{seg_id}")
    print("=" * 76)
    print(
        f"shots={shots:>5}   tvd={tvd:.4f}   "
        f"|z_anc|={z_anc_diff:.4f}   "
        f"|z_e|max={z_e_diff_max:.4f}   "
        f"|z_i|max={z_i_diff_max:.4f}"
    )
    print()
    print(f"  qubit          aer        braket       |diff|")
    print(f"  -----     --------     --------     --------")
    print(
        f"  ancilla   {z_anc_aer:>+8.4f}   {z_anc_braket:>+8.4f}   {z_anc_diff:>8.4f}"
    )
    for i, (a, b) in enumerate(zip(z_e_aer, z_e_braket)):
        print(f"  E_{i}       {a:>+8.4f}   {b:>+8.4f}   {abs(a - b):>8.4f}")
    for i, (a, b) in enumerate(zip(z_i_aer, z_i_braket)):
        print(f"  I_{i}       {a:>+8.4f}   {b:>+8.4f}   {abs(a - b):>8.4f}")
    print()
    print(f"  result: {'PASS' if overall_pass else 'FAIL'}")
    print("=" * 76)

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
