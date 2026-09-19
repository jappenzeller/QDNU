"""
PROMPT AWS-004 - Verification Athena query against aws004.braket_results.

Default behavior runs the locked verification SQL and prints results as a
table. The verification SQL is hardcoded; expected output is 2 rows from
AWS-003 (Path A and Path B).

Usage:
    python scripts/braket/athena_query.py
    python scripts/braket/athena_query.py --output-format json
    python scripts/braket/athena_query.py --sql-file path/to/custom.sql
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import boto3


REPO_ROOT = Path(__file__).resolve().parents[2]
COST_LOG_PATH = REPO_ROOT / "docs" / "aws" / "cost_log.md"
DEFAULT_REGION = "us-east-1"
WORKGROUP_NAME = "qdnu-aws004"
DATABASE_NAME = "aws004"
ATHENA_PRICE_PER_TB = 5.0  # $5 / TB scanned
MIN_BYTES_BILLED = 10 * 1024 * 1024  # 10 MB minimum scan billed by Athena

# Locked verification query for AWS-004 acceptance.
DEFAULT_SQL = """
SELECT
  prompt_id,
  subject,
  device,
  adapter_applied,
  diff.tvd               AS tvd,
  diff.z_e_max_abs       AS z_e_max,
  diff.z_i_max_abs       AS z_i_max,
  "pass",
  billable_seconds,
  actual_cost_usd
FROM aws004.braket_results
WHERE prompt_id = 'AWS-003'
ORDER BY device, adapter_applied DESC
""".strip()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


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


def _start_query(athena, sql: str, workgroup: str, database: str) -> str:
    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        WorkGroup=workgroup,
    )
    return resp["QueryExecutionId"]


def _wait(athena, qid: str, max_wait_s: float = 60.0) -> dict:
    """Poll with exponential backoff. Returns the final QueryExecution dict."""
    delay = 0.25
    elapsed = 0.0
    while elapsed < max_wait_s:
        info = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = info["Status"]["State"]
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            return info
        time.sleep(delay)
        elapsed += delay
        delay = min(delay * 1.7, 4.0)
    raise TimeoutError(f"Athena query {qid} did not finish within {max_wait_s}s")


def _fetch_csv(workgroup_output: str, qid: str, region: str) -> str:
    """Fetch the {qid}.csv result blob from the workgroup output bucket."""
    if not workgroup_output.startswith("s3://"):
        raise ValueError(f"unexpected output location: {workgroup_output!r}")
    rest = workgroup_output[len("s3://"):].rstrip("/")
    bucket, _, prefix = rest.partition("/")
    key = f"{prefix}/{qid}.csv" if prefix else f"{qid}.csv"
    s3 = boto3.client("s3", region_name=region)
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Tiny self-contained table printer (no tabulate dep)."""
    widths = [
        max(len(str(c)) for c in [headers[i]] + [r[i] for r in rows])
        for i in range(len(headers))
    ]
    sep = "  ".join("-" * w for w in widths)
    print("  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)))
    print(sep)
    for r in rows:
        print("  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sql-file",
        type=Path,
        default=None,
        help="path to a SQL file; default = locked verification query",
    )
    ap.add_argument(
        "--output-format",
        choices=("table", "json", "csv"),
        default="table",
    )
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--workgroup", default=WORKGROUP_NAME)
    ap.add_argument("--database", default=DATABASE_NAME)
    args = ap.parse_args()

    if args.sql_file is not None:
        sql = args.sql_file.read_text(encoding="utf-8").strip()
    else:
        sql = DEFAULT_SQL

    athena = boto3.client("athena", region_name=args.region)

    log.info("workgroup=%s database=%s", args.workgroup, args.database)
    qid = _start_query(athena, sql, args.workgroup, args.database)
    log.info("query started: %s", qid)
    info = _wait(athena, qid)
    state = info["Status"]["State"]

    bytes_scanned = info.get("Statistics", {}).get("DataScannedInBytes", 0)
    bytes_billed = max(bytes_scanned, MIN_BYTES_BILLED)
    cost = (bytes_billed / (1024**4)) * ATHENA_PRICE_PER_TB

    if state != "SUCCEEDED":
        reason = info["Status"].get("StateChangeReason", "")
        log.error("query %s: %s", state, reason)
        # Still log the partial cost (Athena may bill failed queries that scanned data)
        _append_cost_row(
            action="athena_scan",
            resource=f"{args.workgroup}/{qid}",
            quantity=bytes_scanned,
            unit="bytes",
            cost_usd=cost,
        )
        return 1

    # Resolve the workgroup's output location to fetch the CSV
    wg = athena.get_work_group(WorkGroup=args.workgroup)
    output_loc = wg["WorkGroup"]["Configuration"]["ResultConfiguration"]["OutputLocation"]
    csv_text = _fetch_csv(output_loc, qid, args.region)

    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        print("(no rows)")
    else:
        headers, *body = rows
        if args.output_format == "csv":
            print(csv_text, end="")
        elif args.output_format == "json":
            objs = [dict(zip(headers, r)) for r in body]
            print(json.dumps(objs, indent=2))
        else:
            _print_table(headers, body)

    _append_cost_row(
        action="athena_scan",
        resource=f"{args.workgroup}/{qid}",
        quantity=bytes_scanned,
        unit="bytes",
        cost_usd=cost,
    )

    print()
    log.info(
        "scanned=%d bytes (billed=%d); est cost=$%.6f",
        bytes_scanned,
        bytes_billed,
        cost,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
