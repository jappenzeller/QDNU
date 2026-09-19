"""
Braket Hybrid Job entry point.

Reads pre-computed (a, b, c) parameters from input_data/, runs the multi-channel
A-Gate on the configured backend, computes <Z> per channel, writes one row per
(patient, window, channel) to a Parquet file in the job's output directory.

Braket Hybrid Jobs invoke this script in a managed container; it picks up
configuration from the standard hyperparameters JSON and writes results to
the path Braket's runtime expects (AMZN_BRAKET_JOB_RESULTS_DIR).
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from braket.aws import AwsDevice
from braket.devices import LocalSimulator
from braket.jobs import save_job_result
from braket.jobs.environment_variables import (
    get_hyperparameters,
    get_input_data_dir,
    get_job_device_arn,
    get_results_dir,
)

try:
    # When loaded as a package (Braket validation step on local machine)
    from .agate_circuit import (
        ChannelParams,
        expectation_z,
        multichannel_agate,
        polarity,
    )
except ImportError:
    # When loaded inside the Hybrid Jobs container (cwd-based imports)
    from agate_circuit import (
        ChannelParams,
        expectation_z,
        multichannel_agate,
        polarity,
    )

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _load_inputs(input_dir: Path) -> dict[str, Any]:
    """Load the precomputed parameter set, run config, and per-patient templates."""
    config_path = input_dir / "config.json"
    params_path = input_dir / "params.json"
    if not config_path.exists() or not params_path.exists():
        raise FileNotFoundError(
            f"expected config.json and params.json under {input_dir}"
        )
    with config_path.open() as f:
        config = json.load(f)
    with params_path.open() as f:
        params = json.load(f)

    templates_path = input_dir / "templates.json"
    templates: dict[str, dict[str, dict[str, float]]] = {}
    if templates_path.exists():
        with templates_path.open() as f:
            templates = json.load(f)
    return {"config": config, "params": params, "templates": templates}


def _hellinger_fidelity(
    counts: dict[str, int], template: dict[str, float]
) -> float:
    """Hellinger fidelity between an empirical count distribution and a template
    probability distribution. F = (sum sqrt(p * q))^2 in [0, 1].

    Both inputs are sparse: counts is the measurement output, template maps
    bitstrings to probabilities. Bitstrings missing from either side
    contribute zero.
    """
    total = sum(counts.values())
    if total == 0 or not template:
        return 0.0
    # Iterate over the smaller side for efficiency.
    keys = counts.keys() if len(counts) <= len(template) else template.keys()
    s = 0.0
    for k in keys:
        p = counts.get(k, 0) / total
        q = template.get(k, 0.0)
        if p > 0 and q > 0:
            s += math.sqrt(p * q)
    return s * s


def _select_device(device_arn: str):
    """Return a Braket device handle.

    Special-case 'local:braket_sv' to use the local simulator for cheap dry runs.
    """
    if device_arn.startswith("local:"):
        return LocalSimulator(device_arn.split(":", 1)[1])
    return AwsDevice(device_arn)


def _run_one_window(
    device,
    channel_params: list[ChannelParams],
    shots: int,
    s3_dest: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Build the multi-channel A-Gate, run it, return per-channel <Z>, polarity,
    and the raw counts dict (for downstream Hellinger fidelity scoring).
    """
    circuit, qubit_map = multichannel_agate(channel_params)

    t0 = time.time()
    if s3_dest is not None:
        task = device.run(circuit, shots=shots, s3_destination_folder=s3_dest)
    else:
        task = device.run(circuit, shots=shots)
    counts = task.result().measurement_counts
    elapsed = time.time() - t0

    total_qubits = qubit_map["total_qubits"]
    e_qubits = qubit_map["e_qubits"]
    i_qubits = qubit_map["i_qubits"]

    per_channel = []
    for k, (eq, iq) in enumerate(zip(e_qubits, i_qubits)):
        z_e = expectation_z(counts, eq, total_qubits)
        z_i = expectation_z(counts, iq, total_qubits)
        per_channel.append(
            {
                "channel_index": k,
                "z_e": z_e,
                "z_i": z_i,
                "polarity_e": polarity(z_e),
                "polarity_i": polarity(z_i),
            }
        )

    return {
        "per_channel": per_channel,
        "total_qubits": total_qubits,
        "shots": shots,
        "wall_time_seconds": elapsed,
        "counts": dict(counts),
    }


def main() -> None:
    hp = get_hyperparameters()
    input_dir = Path(get_input_data_dir("input"))
    results_dir = Path(get_results_dir())
    device_arn = get_job_device_arn() or hp.get("device_arn", "local:braket_sv")
    shots = int(hp.get("shots", 1024))
    run_id = hp.get("run_id", "unspecified")
    task_bucket = hp.get("task_bucket")  # amazon-braket-{account}-{region}
    s3_dest = (task_bucket, f"tasks/{run_id}") if task_bucket else None

    log.info("device_arn=%s shots=%d run_id=%s", device_arn, shots, run_id)

    payload = _load_inputs(input_dir)
    config = payload["config"]
    params = payload["params"]
    templates = payload["templates"]  # may be empty
    device = _select_device(device_arn)

    rows: list[dict[str, Any]] = []

    backend_name = device_arn.rsplit("/", 1)[-1] if "/" in device_arn else device_arn

    for entry in params["windows"]:
        patient_id = entry["patient_id"]
        window_id = entry["window_id"]
        window_state = entry["state"]  # "interictal" / "preictal" / "ictal"
        channel_params = [
            ChannelParams(a=ch["a"], b=ch["b"], c=ch["c"])
            for ch in entry["channels"]
        ]

        log.info(
            "running patient=%s window=%s state=%s channels=%d",
            patient_id,
            window_id,
            window_state,
            len(channel_params),
        )

        try:
            result = _run_one_window(device, channel_params, shots, s3_dest=s3_dest)
        except Exception as exc:
            log.exception("run failed: %s", exc)
            rows.append(
                {
                    "run_id": run_id,
                    "backend": backend_name,
                    "patient_id": patient_id,
                    "window_id": window_id,
                    "window_state": window_state,
                    "channel_index": -1,
                    "z_e": None,
                    "z_i": None,
                    "polarity_e": None,
                    "polarity_i": None,
                    "fid_ictal": None,
                    "fid_inter": None,
                    "fid_score": None,
                    "shots": shots,
                    "wall_time_seconds": None,
                    "error": str(exc),
                    "schema_version": config.get("schema_version", "v1"),
                }
            )
            continue

        # Compute Hellinger fidelities once per window (template lookup is per
        # patient; the score is the same across all rows of the same window).
        patient_templates = templates.get(patient_id) if templates else None
        if patient_templates:
            fid_ictal = _hellinger_fidelity(
                result["counts"], patient_templates.get("ictal", {})
            )
            fid_inter = _hellinger_fidelity(
                result["counts"], patient_templates.get("interictal", {})
            )
            fid_score = fid_ictal - fid_inter
        else:
            fid_ictal = None
            fid_inter = None
            fid_score = None

        for ch in result["per_channel"]:
            rows.append(
                {
                    "run_id": run_id,
                    "backend": backend_name,
                    "patient_id": patient_id,
                    "window_id": window_id,
                    "window_state": window_state,
                    "channel_index": ch["channel_index"],
                    "z_e": ch["z_e"],
                    "z_i": ch["z_i"],
                    "polarity_e": ch["polarity_e"],
                    "polarity_i": ch["polarity_i"],
                    "fid_ictal": fid_ictal,
                    "fid_inter": fid_inter,
                    "fid_score": fid_score,
                    "shots": shots,
                    "wall_time_seconds": result["wall_time_seconds"],
                    "error": None,
                    "schema_version": config.get("schema_version", "v1"),
                }
            )

    table = pa.Table.from_pylist(rows)
    out_path = results_dir / "results.parquet"
    pq.write_table(table, out_path)
    log.info("wrote %d rows to %s", len(rows), out_path)

    # Also upload directly to the Athena-partitioned layout. The Hybrid Job's
    # standard output mechanism produces a tarball at .../output/model.tar.gz
    # which is opaque to Athena's partition projection. Writing a second copy
    # to runs/{run_id}/backend={backend}/results.parquet makes the data
    # queryable as soon as the job completes.
    results_bucket = hp.get("results_bucket")
    if results_bucket:
        import boto3

        # Flat layout, no Athena partition projection. The dataset is small
        # enough that scan-and-filter on the run_id/backend columns is fine,
        # and it avoids the WHERE-clause constraints partition projection
        # imposes.
        key = f"parquet/{run_id}__{backend_name}.parquet"
        log.info("uploading queryable copy to s3://%s/%s", results_bucket, key)
        boto3.client("s3").upload_file(str(out_path), results_bucket, key)

    # Also surface a small summary so the Hybrid Jobs UI shows something useful.
    save_job_result(
        {
            "run_id": run_id,
            "backend": backend_name,
            "row_count": len(rows),
            "patients": sorted({r["patient_id"] for r in rows}),
            "windows": sorted({r["window_id"] for r in rows}),
        }
    )


if __name__ == "__main__":
    main()
