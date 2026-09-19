"""
PROMPT AWS-004 - Upload locked-schema JSONs to the partitioned S3 layout.

Reads each JSON under --glob, extracts (prompt_id, device, subject) for the
Hive-style partition path, and PUTs to:

    s3://qdnu-braket-{account}-us-east-1/processed/
        prompt_id={pid}/device={dev}/subject={subj}/{ts}.json

Pre-existing keys are skipped, never silently overwritten. After the session,
appends one cost-log row to docs/aws/cost_log.md.

Usage:
    python scripts/braket/upload_results_to_s3.py
    python scripts/braket/upload_results_to_s3.py --glob 'results/braket/sv1_*.json'
    python scripts/braket/upload_results_to_s3.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


REPO_ROOT = Path(__file__).resolve().parents[2]
COST_LOG_PATH = REPO_ROOT / "docs" / "aws" / "cost_log.md"
DEFAULT_REGION = "us-east-1"
S3_PUT_PRICE_USD = 0.000005  # $5 / million PUTs (us-east-1 standard)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _bucket_name(account_id: str) -> str:
    return f"qdnu-braket-{account_id}-{DEFAULT_REGION}"


def _safe_for_s3(s: str) -> str:
    """Convert ISO8601 colons to dashes for S3-key safety."""
    return s.replace(":", "-").replace("+", "-")


def _required_fields(record: dict[str, Any], path: Path) -> tuple[str, str, str, str]:
    """Pull the partition + timestamp fields. Fail loudly if missing."""
    missing = [k for k in ("prompt_id", "device", "subject", "timestamp") if k not in record]
    if missing:
        raise ValueError(f"{path}: missing required fields {missing}")
    return (
        record["prompt_id"],
        record["device"],
        record["subject"],
        record["timestamp"],
    )


def _key_for(record: dict[str, Any], path: Path) -> str:
    pid, dev, subj, ts = _required_fields(record, path)
    return (
        "processed/"
        f"prompt_id={pid}/"
        f"device={dev}/"
        f"subject={subj}/"
        f"{_safe_for_s3(ts)}.json"
    )


def _key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _ensure_cost_log() -> None:
    if COST_LOG_PATH.exists():
        return
    COST_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    COST_LOG_PATH.write_text(
        "# AWS-004 Data Layer Cost Log\n\n"
        "Append-only. One row per cost-incurring action (S3 PUT, Athena scan, etc.).\n\n"
        "| timestamp | action | resource | quantity | unit | cost_usd |\n"
        "|---|---|---|---|---|---|\n",
        encoding="utf-8",
    )


def _append_cost_row(
    *,
    action: str,
    resource: str,
    quantity: float,
    unit: str,
    cost_usd: float,
) -> None:
    _ensure_cost_log()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row = (
        f"| {ts} | {action} | {resource} | {quantity:g} | {unit} | "
        f"{cost_usd:.6f} |\n"
    )
    with COST_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--glob",
        default="results/braket/*.json",
        help="local glob pattern for JSON files to upload",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print what would be uploaded; no S3 calls",
    )
    ap.add_argument("--region", default=DEFAULT_REGION)
    args = ap.parse_args()

    files = sorted((REPO_ROOT).glob(args.glob))
    if not files:
        log.warning("no files match glob: %s", args.glob)
        return 0

    sts = boto3.client("sts", region_name=args.region)
    account_id = sts.get_caller_identity()["Account"]
    bucket = _bucket_name(account_id)
    s3 = boto3.client("s3", region_name=args.region)

    n_uploaded = 0
    n_skipped = 0
    total_bytes = 0

    for path in files:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log.warning("not valid JSON, skipping: %s (%s)", path, e)
            continue
        try:
            key = _key_for(record, path)
        except ValueError as e:
            log.warning("missing fields, skipping: %s", e)
            continue

        s3_uri = f"s3://{bucket}/{key}"
        if args.dry_run:
            log.info("DRY RUN  %s -> %s", path.relative_to(REPO_ROOT), s3_uri)
            n_uploaded += 1  # for dry-run summary
            total_bytes += path.stat().st_size
            continue

        if _key_exists(s3, bucket, key):
            log.warning("EXISTS   %s -> %s (skipping; no overwrite)",
                        path.relative_to(REPO_ROOT), s3_uri)
            n_skipped += 1
            continue

        # OpenX JsonSerDe (used by the Athena table) requires one JSON
        # object per line. Compact the record to a single line.
        body = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/x-ndjson",
        )
        log.info("UPLOAD   %s -> %s (%d bytes compact)",
                 path.relative_to(REPO_ROOT), s3_uri, len(body))
        n_uploaded += 1
        total_bytes += len(body)

    cost_estimate = n_uploaded * S3_PUT_PRICE_USD if not args.dry_run else 0.0
    if not args.dry_run and n_uploaded > 0:
        _append_cost_row(
            action="s3_put",
            resource=f"s3://{bucket}/processed/",
            quantity=n_uploaded,
            unit="objects",
            cost_usd=cost_estimate,
        )

    print()
    print("=" * 72)
    print("AWS-004 upload summary")
    print("=" * 72)
    print(f"  bucket:    s3://{bucket}/")
    print(f"  uploaded:  {n_uploaded}")
    print(f"  skipped:   {n_skipped}")
    print(f"  bytes:     {total_bytes}")
    print(f"  cost est:  ${cost_estimate:.6f}")
    if args.dry_run:
        print("  (dry run; no S3 calls made)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
