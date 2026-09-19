"""
PROMPT AWS-003 - Cloud SV1 equivalence + cost telemetry.

Mirrors scripts/braket/braket_local_validation.py. Same chb01 segment, same
encoding, same shots. Differences:

  - Backend is Braket SV1 (cloud), not LocalSimulator.
  - Two SV1 paths run in sequence:
      A) adapter applied (prepare_circuit_for_braket)
      B) bypass (raw qc submitted to SV1)
  - Each submission is cost-estimated. The script aborts before submitting
    if the estimate exceeds the hard cap, and refuses to submit at all
    without the --confirm-spend flag.
  - Outputs follow the AWS-003 schema with task_arn, S3 location,
    billable_seconds, estimated_cost_usd, and actual_cost_usd populated.
  - One Aer reference run is shared between Path A and Path B.

Bounded to ~$2 total spend.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# braket-default-simulator's numba kernel can't reshape >=17-qubit state
# tensors. We use Aer for the local reference; the cloud SV1 path doesn't
# need numba. Setting this anyway is harmless and matches the local script.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np  # noqa: E402

# Preload Braket SDK + provider before adding scripts/ to sys.path so the
# local scripts/braket/ package doesn't shadow the installed braket SDK.
import braket  # noqa: F401, E402
import qiskit_braket_provider  # noqa: F401, E402

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "QA1"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "braket"))

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_braket_provider import BraketProvider  # noqa: E402

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

from braket_adapter import prepare_circuit_for_braket  # noqa: E402
from cost_tracker import (  # noqa: E402
    HARD_CAP_USD,
    estimate_sv1_cost,
    log_actual_cost,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


SHOTS = 4096
SUBJECT = "chb01"
WINDOW_S = 1.95
THRESHOLD_Z = 0.02
THRESHOLD_TVD = 0.05

DEVICE_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
DEFAULT_REGION = "us-east-1"
DEFAULT_S3_PREFIX = "validation/aws-003"

OUTPUT_PATH_A = REPO_ROOT / "results" / "braket" / "sv1_equivalence_chb01.json"
OUTPUT_PATH_B = REPO_ROOT / "results" / "braket" / "sv1_no_adapter_chb01.json"


# ============================================================================
# Data + circuit
# ============================================================================


def _load_first_ictal(subject: str) -> tuple[np.ndarray, str]:
    subject_dir = DATA_DIR / subject
    segments = extract_segments_for_subject(subject_dir)
    ictal = [s for s in segments if s.label == "ictal"]
    if not ictal:
        raise RuntimeError(f"no ictal segments for {subject}")
    for seg in ictal:
        eeg = load_eeg_segment(seg, subject_dir, QUANTUM_CHANNELS)
        if eeg is not None:
            return eeg, f"{seg.source_file}#start={seg.start_sec:.1f}s"
    raise RuntimeError(f"failed to load any ictal segment for {subject}")


def _expectation_z(counts: dict[str, int], qubit_index: int, n_qubits: int) -> float:
    """Compute <Z> on qubit `qubit_index` from a Qiskit-style counts dict.

    Qiskit bitstring convention: leftmost char = highest qubit. So
    bitstring[n_qubits - 1 - qubit_index] is the bit for `qubit_index`.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    weighted = 0.0
    for bitstring, n in counts.items():
        bs = bitstring.replace(" ", "")
        bit = bs[n_qubits - 1 - qubit_index]
        weighted += n * (1.0 if bit == "0" else -1.0)
    return weighted / total


def _total_variation_distance(
    counts_a: dict[str, int], counts_b: dict[str, int]
) -> float:
    total_a = sum(counts_a.values()) or 1
    total_b = sum(counts_b.values()) or 1
    keys = set(counts_a) | set(counts_b)
    s = 0.0
    for k in keys:
        pa = counts_a.get(k, 0) / total_a
        pb = counts_b.get(k, 0) / total_b
        s += abs(pa - pb)
    return 0.5 * s


# ============================================================================
# Backends
# ============================================================================


def _run_aer(circuit: QuantumCircuit, shots: int, seed: int = 42) -> dict[str, int]:
    sim = AerSimulator(seed_simulator=seed)
    return sim.run(circuit, shots=shots, seed_simulator=seed).result().get_counts()


SV1_SHOTS_PER_TASK_MAX = 100_000


def _run_sv1(
    circuit: QuantumCircuit,
    shots: int,
    s3_destination: tuple[str, str] | None,
) -> tuple[dict[str, int], list[str]]:
    """Submit to SV1 and block until result. Returns (counts, [task_arns]).

    SV1 caps a single task at 100k shots. If more are requested, submit
    multiple tasks of up to 100k each and aggregate the counts. The list of
    ARNs is returned so each task can be cost-logged separately.
    """
    backend = BraketProvider().get_backend("SV1")

    # Chunk into <= SV1_SHOTS_PER_TASK_MAX per task
    remaining = shots
    chunks: list[int] = []
    while remaining > 0:
        c = min(SV1_SHOTS_PER_TASK_MAX, remaining)
        chunks.append(c)
        remaining -= c
    log.info(
        "SV1 dispatch: %d shots in %d task(s) (chunk sizes %s)",
        shots,
        len(chunks),
        chunks,
    )

    aggregate: dict[str, int] = {}
    task_arns: list[str] = []
    for i, chunk_shots in enumerate(chunks, 1):
        if s3_destination is not None:
            job = backend.run(
                circuit, shots=chunk_shots, s3_destination_folder=s3_destination
            )
        else:
            job = backend.run(circuit, shots=chunk_shots)
        task_arn_attr = getattr(job, "task_id", None)
        task_arn = task_arn_attr() if callable(task_arn_attr) else task_arn_attr
        if not task_arn:
            wrapped = getattr(job, "_task", None) or getattr(job, "task", None)
            task_arn = (
                getattr(wrapped, "id", None) or getattr(wrapped, "arn", None) or ""
            )
        log.info("SV1 task %d/%d submitted: %s", i, len(chunks), task_arn)
        task_arns.append(task_arn)
        result = job.result()
        for k, v in result.get_counts().items():
            aggregate[k] = aggregate.get(k, 0) + v
    return aggregate, task_arns


# ============================================================================
# Single-path runner
# ============================================================================


def _run_path(
    *,
    label: str,
    measured_aer: QuantumCircuit,
    base_circuit: QuantumCircuit,
    use_adapter: bool,
    aer_counts: dict[str, int],
    shots: int,
    seg_id: str,
    params: list[tuple[float, float, float]],
    num_channels: int,
    n_qubits: int,
    qmap: dict[str, Any],
    s3_destination: tuple[str, str] | None,
    region: str,
    output_path: Path,
) -> dict[str, Any]:
    """Run one SV1 submission (Path A or B), compute diffs, write JSON,
    log cost. Returns the output dict.
    """
    log.info("=== %s (use_adapter=%s) ===", label, use_adapter)

    if use_adapter:
        circuit_for_sv1 = prepare_circuit_for_braket(copy.deepcopy(measured_aer))
        log.info("adapter applied")
    else:
        circuit_for_sv1 = measured_aer

    estimated = estimate_sv1_cost(n_qubits, shots)
    log.info("%s estimated SV1 cost: $%.4f", label, estimated)

    submitted_at = datetime.now(timezone.utc)
    sv1_counts, task_arns = _run_sv1(circuit_for_sv1, shots, s3_destination)
    primary_task_arn = task_arns[0] if task_arns else ""

    # Optional unitary equivalence proof (Path A only). For Path B the
    # circuit IS identical to the Aer one; equivalence is trivial.
    unitary_equivalence: dict[str, Any] | None = None
    if use_adapter:
        sv_orig = Statevector.from_instruction(base_circuit)
        sv_decomp = Statevector.from_instruction(
            prepare_circuit_for_braket(copy.deepcopy(base_circuit))
        )
        fid = float(abs(sv_orig.inner(sv_decomp)) ** 2)
        ideal_tvd = float(
            0.5 * np.sum(
                np.abs(np.abs(sv_orig.data) ** 2 - np.abs(sv_decomp.data) ** 2)
            )
        )
        unitary_equivalence = {
            "fidelity": fid,
            "ideal_tvd_probabilities": ideal_tvd,
            "passes": fid > 0.9999 and ideal_tvd < 1e-10,
        }

    # Per-qubit Z + TVD vs Aer
    z_anc_aer = _expectation_z(aer_counts, qmap["ancilla"], n_qubits)
    z_anc_sv1 = _expectation_z(sv1_counts, qmap["ancilla"], n_qubits)
    z_e_aer = [_expectation_z(aer_counts, q, n_qubits) for q in qmap["E_qubits"]]
    z_e_sv1 = [_expectation_z(sv1_counts, q, n_qubits) for q in qmap["E_qubits"]]
    z_i_aer = [_expectation_z(aer_counts, q, n_qubits) for q in qmap["I_qubits"]]
    z_i_sv1 = [_expectation_z(sv1_counts, q, n_qubits) for q in qmap["I_qubits"]]

    z_anc_diff = abs(z_anc_aer - z_anc_sv1)
    z_e_diff_max = max(abs(a - b) for a, b in zip(z_e_aer, z_e_sv1))
    z_i_diff_max = max(abs(a - b) for a, b in zip(z_i_aer, z_i_sv1))
    tvd = _total_variation_distance(aer_counts, sv1_counts)

    pass_z = (
        z_anc_diff < THRESHOLD_Z
        and z_e_diff_max < THRESHOLD_Z
        and z_i_diff_max < THRESHOLD_Z
    )
    pass_tvd = tvd < THRESHOLD_TVD
    overall_pass = bool(pass_z and pass_tvd)

    # Log each task in the cost log; sum total cost.
    actual_cost = 0.0
    billable_s = 0.0
    for arn in task_arns:
        cost_info = log_actual_cost(
            task_arn=arn,
            region=region,
            output_path=str(output_path.relative_to(REPO_ROOT)),
            notes=f"AWS-003 {label}",
        )
        actual_cost += cost_info["cost_usd"]
        billable_s += cost_info["billable_seconds"]

    s3_bucket, s3_key = "", ""
    if s3_destination is not None and primary_task_arn:
        s3_bucket = s3_destination[0]
        s3_key = f"{s3_destination[1]}/{primary_task_arn.rsplit('/', 1)[-1]}"

    output: dict[str, Any] = {
        "prompt_id": "AWS-003",
        "timestamp": submitted_at.isoformat(),
        "subject": SUBJECT,
        "segment_id": seg_id,
        "channels": num_channels,
        "window_s": WINDOW_S,
        "shots": shots,
        "encoding": "V3_PLV_theta_alpha",
        "params": [[float(a), float(b), float(c)] for (a, b, c) in params],
        "device": "SV1",
        "device_arn": DEVICE_ARN,
        "adapter_applied": use_adapter,
        "aer": {
            "counts": {k: int(v) for k, v in aer_counts.items()},
            "z_ancilla": float(z_anc_aer),
            "z_e": [float(x) for x in z_e_aer],
            "z_i": [float(x) for x in z_i_aer],
        },
        "braket": {
            "counts": {k: int(v) for k, v in sv1_counts.items()},
            "z_ancilla": float(z_anc_sv1),
            "z_e": [float(x) for x in z_e_sv1],
            "z_i": [float(x) for x in z_i_sv1],
        },
        "diff": {
            "z_ancilla_abs": float(z_anc_diff),
            "z_e_max_abs": float(z_e_diff_max),
            "z_i_max_abs": float(z_i_diff_max),
            "tvd": float(tvd),
        },
        "pass": overall_pass,
        "task_arn": primary_task_arn,
        "task_arns": task_arns,
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
        "region": region,
        "billable_seconds": float(billable_s),
        "estimated_cost_usd": float(estimated),
        "actual_cost_usd": float(actual_cost),
        "notes": (
            f"AWS-003 path {label}; thresholds unchanged from AWS-001. "
            f"Aer reference is the same Qiskit local sim run; SV1 path "
            f"{'has' if use_adapter else 'does not have'} the AWS-002 "
            f"manual-decomposition adapter applied."
        ),
    }
    if unitary_equivalence is not None:
        output["unitary_equivalence"] = unitary_equivalence

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    log.info("wrote %s", output_path)
    return output


# ============================================================================
# Driver
# ============================================================================


def _interpret(path_a_pass: bool, path_b_pass: bool) -> str:
    if path_a_pass and path_b_pass:
        return (
            "BOTH PASS: SV1 reproduces neither AWS-001 anomaly. The local-sim "
            "divergence was specific to braket-default-simulator. The adapter "
            "is local-only insurance; do NOT carry it forward to QPU "
            "submissions where extra depth costs gate count."
        )
    if path_a_pass and not path_b_pass:
        return (
            "ADAPTER PASS, BYPASS FAIL: SV1 reproduces the AWS-002 anomaly. "
            "The adapter is genuinely needed everywhere. Carry it to all "
            "Braket-bridge submissions including QPU; accept the gate-count "
            "overhead until the upstream provider is fixed."
        )
    if not path_a_pass and path_b_pass:
        return (
            "ADAPTER FAIL, BYPASS PASS: unexpected. The adapter is breaking "
            "what would otherwise work on SV1. Inspect both circuits, "
            "regenerate AWS-002 v2 evidence with care, do not advance to AWS-004."
        )
    return (
        "BOTH FAIL: SV1 disagrees with Aer regardless of adapter. Bridge or "
        "Aer reference may be inconsistent. STOP. Investigate before any "
        "further spend."
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=SHOTS)
    ap.add_argument(
        "--confirm-spend",
        action="store_true",
        help="Required to actually submit to SV1. Without this, the script "
             "prints estimated cost and exits.",
    )
    ap.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", DEFAULT_REGION),
    )
    ap.add_argument(
        "--s3-bucket",
        default=os.environ.get("BRAKET_OUTPUT_S3_BUCKET"),
        help="S3 bucket for SV1 task outputs. Defaults to "
             "BRAKET_OUTPUT_S3_BUCKET env var. If unset, Braket uses its "
             "own default (created on first task).",
    )
    ap.add_argument(
        "--s3-prefix",
        default=os.environ.get("BRAKET_OUTPUT_S3_PREFIX", DEFAULT_S3_PREFIX),
    )
    args = ap.parse_args()
    shots = args.shots

    # Pin region for the SDK
    os.environ["AWS_DEFAULT_REGION"] = args.region

    log.info("loading %s first ictal segment via canonical pipeline", SUBJECT)
    eeg, seg_id = _load_first_ictal(SUBJECT)
    log.info("loaded segment %s shape=%s", seg_id, eeg.shape)

    log.info("extracting V3 PLV params")
    params = extract_plv_params(eeg, fs=256.0, band=(4, 13))

    num_channels = len(QUANTUM_CHANNELS)
    n_qubits = 2 * num_channels + 1
    qmap = get_qubit_indices(num_channels)

    base_circuit = create_multichannel_circuit(params)
    measured_aer = add_measurements(
        copy.deepcopy(base_circuit), num_channels, measure_all=True
    )

    # Cost guard: estimate both paths up front. SV1 chunks at 100k shots/task
    # so multi-chunk requests sum estimates per chunk. Abort if any single
    # chunk exceeds the per-task cap (estimate_sv1_cost enforces).
    n_chunks = max(1, (shots + SV1_SHOTS_PER_TASK_MAX - 1) // SV1_SHOTS_PER_TASK_MAX)
    last_chunk_shots = shots - (n_chunks - 1) * SV1_SHOTS_PER_TASK_MAX
    chunk_shots = min(SV1_SHOTS_PER_TASK_MAX, shots)
    est_a = estimate_sv1_cost(n_qubits, chunk_shots) * (n_chunks - 1) + estimate_sv1_cost(
        n_qubits, last_chunk_shots
    )
    est_b = est_a
    total_estimated = est_a + est_b
    log.info(
        "AWS-003 cost estimate: Path A $%.4f + Path B $%.4f = $%.4f total "
        "(hard cap per task $%.2f)",
        est_a,
        est_b,
        total_estimated,
        HARD_CAP_USD,
    )

    if not args.confirm_spend:
        print()
        print("=" * 72)
        print("AWS-003 SV1 Equivalence Check (DRY RUN)")
        print("=" * 72)
        print(f"  region:           {args.region}")
        print(f"  device:           SV1 ({DEVICE_ARN})")
        print(f"  segment:          {SUBJECT}/{seg_id}")
        print(f"  shots per path:   {shots}  (split into {n_chunks} task(s))")
        print(f"  Path A estimate:  ${est_a:.4f}  (adapter applied)")
        print(f"  Path B estimate:  ${est_b:.4f}  (no adapter)")
        print(f"  Total estimate:   ${total_estimated:.4f}")
        print(f"  Hard cap (each):  ${HARD_CAP_USD:.2f}")
        print()
        print("To submit, re-run with --confirm-spend.")
        print("=" * 72)
        return 0

    # Aer reference run (free, local).
    log.info("running Aer reference (shots=%d)", shots)
    aer_counts = _run_aer(measured_aer, shots, seed=42)

    # S3 destination tuple expected by qiskit-braket-provider
    s3_destination = None
    if args.s3_bucket:
        s3_destination = (args.s3_bucket, args.s3_prefix)
        log.info("SV1 outputs -> s3://%s/%s/", args.s3_bucket, args.s3_prefix)

    # Path A: with adapter
    out_a = _run_path(
        label="A_adapter",
        measured_aer=measured_aer,
        base_circuit=base_circuit,
        use_adapter=True,
        aer_counts=aer_counts,
        shots=shots,
        seg_id=seg_id,
        params=params,
        num_channels=num_channels,
        n_qubits=n_qubits,
        qmap=qmap,
        s3_destination=s3_destination,
        region=args.region,
        output_path=OUTPUT_PATH_A,
    )

    # Path B: bypass
    out_b = _run_path(
        label="B_bypass",
        measured_aer=measured_aer,
        base_circuit=base_circuit,
        use_adapter=False,
        aer_counts=aer_counts,
        shots=shots,
        seg_id=seg_id,
        params=params,
        num_channels=num_channels,
        n_qubits=n_qubits,
        qmap=qmap,
        s3_destination=s3_destination,
        region=args.region,
        output_path=OUTPUT_PATH_B,
    )

    total_actual = out_a["actual_cost_usd"] + out_b["actual_cost_usd"]
    interpretation = _interpret(out_a["pass"], out_b["pass"])

    # One-screen comparative summary
    print()
    print("=" * 72)
    print("AWS-003 SV1 Equivalence Comparative Summary")
    print("=" * 72)
    print(f"  segment:                 {SUBJECT}/{seg_id}")
    print(f"  shots:                   {shots}")
    print()
    print(f"  Path A (adapter applied)")
    print(f"    pass:       {out_a['pass']}")
    d = out_a["diff"]
    print(
        f"    diffs:      tvd={d['tvd']:.4f}  "
        f"|z_anc|={d['z_ancilla_abs']:.4f}  "
        f"|z_e|max={d['z_e_max_abs']:.4f}  "
        f"|z_i|max={d['z_i_max_abs']:.4f}"
    )
    print(f"    estimated:  ${out_a['estimated_cost_usd']:.4f}")
    print(f"    actual:     ${out_a['actual_cost_usd']:.4f} "
          f"({out_a['billable_seconds']:.2f}s, {len(out_a['task_arns'])} task(s))")
    print(f"    task arn:   {out_a['task_arn']}")
    print()
    print(f"  Path B (bypass)")
    print(f"    pass:       {out_b['pass']}")
    d = out_b["diff"]
    print(
        f"    diffs:      tvd={d['tvd']:.4f}  "
        f"|z_anc|={d['z_ancilla_abs']:.4f}  "
        f"|z_e|max={d['z_e_max_abs']:.4f}  "
        f"|z_i|max={d['z_i_max_abs']:.4f}"
    )
    print(f"    estimated:  ${out_b['estimated_cost_usd']:.4f}")
    print(f"    actual:     ${out_b['actual_cost_usd']:.4f} "
          f"({out_b['billable_seconds']:.2f}s, {len(out_b['task_arns'])} task(s))")
    print(f"    task arn:   {out_b['task_arn']}")
    print()
    print(f"  Total actual: ${total_actual:.4f}")
    print()
    print("INTERPRETATION:")
    print(f"  {interpretation}")
    print("=" * 72)

    # Cost sanity guard required by AWS-003 acceptance criteria
    if (
        out_a["actual_cost_usd"]
        > out_a["estimated_cost_usd"] * 1.5
    ):
        log.warning(
            "Path A actual cost ($%.4f) exceeds 1.5x estimate ($%.4f); "
            "estimator is too aggressive.",
            out_a["actual_cost_usd"],
            out_a["estimated_cost_usd"],
        )

    return 0 if out_a["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
