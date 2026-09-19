"""
Submit a Braket Hybrid Job for a chosen backend.

Usage:
    python submit_job.py --backend sv1 --shots 1024
    python submit_job.py --backend rigetti-ankaa --shots 1024
    python submit_job.py --backend ionq-aria-1 --shots 256 --dry-run

Backends are name-aliased to ARNs in BACKEND_ARNS. The job uploads
job/algorithm_script.py + agate_circuit.py and the input_data/ bundle, then
streams Parquet results to:
    s3://qdnu-braket-results/runs/{run_id}/backend={backend}/results.parquet

Run prepare_input.py first to populate input_data/.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[3]
JOB_DIR = REPO_ROOT / "aws" / "braket" / "job"
INPUT_DIR = JOB_DIR / "input_data"

BACKEND_ARNS = {
    # Managed simulators (cheapest, run first)
    "sv1": "arn:aws:braket:::device/quantum-simulator/amazon/sv1",
    "dm1": "arn:aws:braket:::device/quantum-simulator/amazon/dm1",
    "tn1": "arn:aws:braket:::device/quantum-simulator/amazon/tn1",
    # Local simulator for dry runs (handled specially in algorithm_script)
    "local": "local:braket_sv",
    # Real hardware (verify availability before paid runs)
    "ionq-aria-1": "arn:aws:braket:us-east-1::device/qpu/ionq/Aria-1",
    "ionq-forte-1": "arn:aws:braket:us-east-1::device/qpu/ionq/Forte-1",
    "rigetti-ankaa-3": "arn:aws:braket:us-west-1::device/qpu/rigetti/Ankaa-3",
    "iqm-garnet": "arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet",
}

DEFAULT_S3_BUCKET = "qdnu-braket-results"
DEFAULT_ROLE_ARN_ENV = "BRAKET_JOBS_ROLE_ARN"
DEFAULT_TASK_BUCKET_ENV = "BRAKET_TASK_BUCKET"


def _make_run_id(backend: str) -> str:
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:6]
    return f"{stamp}_{backend}_{short}"


def _validate_inputs() -> None:
    if not (INPUT_DIR / "config.json").exists() or not (INPUT_DIR / "params.json").exists():
        raise SystemExit(
            "missing job/input_data/{config,params}.json. "
            "Run scripts/prepare_input.py first."
        )


def _summarize_inputs() -> dict:
    with (INPUT_DIR / "params.json").open() as f:
        params = json.load(f)
    n_windows = len(params["windows"])
    n_rows = sum(len(w["channels"]) for w in params["windows"])
    patients = sorted({w["patient_id"] for w in params["windows"]})
    return {"windows": n_windows, "rows": n_rows, "patients": patients}


def _submit(args: argparse.Namespace, run_id: str, device_arn: str) -> str | None:
    if args.dry_run:
        log.info("dry-run: skipping AwsQuantumJob.create")
        return None

    # Imported lazily so dry-runs don't require boto3.
    from braket.aws import AwsQuantumJob
    from braket.jobs.config import OutputDataConfig

    image_uri = args.image_uri  # if not set, Braket uses the default Hybrid Jobs image
    s3_destination = f"s3://{args.s3_bucket}/runs/{run_id}"
    output_cfg = OutputDataConfig(s3Path=s3_destination)

    role_arn = args.role_arn or os.environ.get(DEFAULT_ROLE_ARN_ENV)
    if not role_arn:
        raise SystemExit(
            f"missing role ARN. Pass --role-arn or set {DEFAULT_ROLE_ARN_ENV}. "
            "Get it from the IAM stack output: "
            "qdnu-dev-braket-jobs-role-arn"
        )

    task_bucket = args.task_bucket or os.environ.get(DEFAULT_TASK_BUCKET_ENV)
    if not task_bucket:
        raise SystemExit(
            f"missing task bucket. Pass --task-bucket or set {DEFAULT_TASK_BUCKET_ENV}. "
            "Should be of the form amazon-braket-{account}-{region}."
        )

    log.info("creating Hybrid Job run_id=%s device=%s role=%s", run_id, device_arn, role_arn)
    job = AwsQuantumJob.create(
        device=device_arn,
        source_module=str(JOB_DIR),  # Braket packages the directory
        entry_point=f"{JOB_DIR.name}.algorithm_script:main",
        job_name=run_id.replace("_", "-")[:50],
        role_arn=role_arn,
        hyperparameters={
            "shots": args.shots,
            "run_id": run_id,
            "device_arn": device_arn,
            "task_bucket": task_bucket,
            "results_bucket": args.s3_bucket,
        },
        input_data={"input": str(INPUT_DIR)},
        output_data_config=output_cfg,
        image_uri=image_uri,
        wait_until_complete=False,
    )
    log.info("submitted: arn=%s", job.arn)
    return job.arn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=sorted(BACKEND_ARNS.keys()),
        required=True,
    )
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--s3-bucket", default=DEFAULT_S3_BUCKET)
    parser.add_argument(
        "--image-uri",
        default=None,
        help="Override Hybrid Jobs container image URI",
    )
    parser.add_argument(
        "--role-arn",
        default=None,
        help=f"Hybrid Jobs execution role ARN (or set {DEFAULT_ROLE_ARN_ENV})",
    )
    parser.add_argument(
        "--task-bucket",
        default=None,
        help=f"amazon-braket-prefixed bucket for task outputs (or set {DEFAULT_TASK_BUCKET_ENV})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without calling AWS",
    )
    args = parser.parse_args()

    _validate_inputs()
    summary = _summarize_inputs()
    log.info(
        "input bundle: %d windows, %d rows, patients=%s",
        summary["windows"],
        summary["rows"],
        summary["patients"],
    )

    device_arn = BACKEND_ARNS[args.backend]
    run_id = _make_run_id(args.backend)
    log.info("backend=%s -> arn=%s", args.backend, device_arn)
    log.info("run_id=%s", run_id)
    log.info("output s3://%s/runs/%s/", args.s3_bucket, run_id)

    arn = _submit(args, run_id, device_arn)
    if arn:
        print(arn)
    else:
        print(f"DRY RUN run_id={run_id}")


if __name__ == "__main__":
    main()
