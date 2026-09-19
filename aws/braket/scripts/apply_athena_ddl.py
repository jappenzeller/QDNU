"""
Render and apply the Athena DDL for a given environment.

Substitutes the placeholders in infra/athena_ddl.sql with values discovered
from the deployed CloudFormation stack outputs, then submits each statement
to Athena via the chosen workgroup.

Usage:
    python apply_athena_ddl.py                  # dev
    python apply_athena_ddl.py --env prod
    python apply_athena_ddl.py --dry-run        # print rendered SQL only
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[3]
DDL_PATH = REPO_ROOT / "aws" / "braket" / "infra" / "athena_ddl.sql"


def _stack_outputs(cf, stack_name: str) -> dict[str, str]:
    resp = cf.describe_stacks(StackName=stack_name)
    outputs = resp["Stacks"][0].get("Outputs", []) or []
    return {o["OutputKey"]: o["OutputValue"] for o in outputs}


def _render(template: str, db: str, bucket: str) -> str:
    """Replace placeholder values with deployed names."""
    return (
        template
        .replace("qdnu_quantum.", f"{db}.")
        .replace("CREATE DATABASE IF NOT EXISTS qdnu_quantum", f"CREATE DATABASE IF NOT EXISTS {db}")
        .replace("s3://qdnu-braket-results/", f"s3://{bucket}/")
    )


def _split_statements(sql: str) -> list[str]:
    """Split a multi-statement SQL file into individual statements.

    Strategy: strip all line comments first, then split on `;`. None of our
    DDL contains string literals with semicolons, so plain split is safe.
    """
    no_comments_lines = []
    for line in sql.splitlines():
        # Drop everything from `--` to end-of-line, but only outside quotes.
        # Our DDL has no quoted `--` so a simple split is fine.
        idx = line.find("--")
        if idx >= 0:
            line = line[:idx]
        no_comments_lines.append(line)
    cleaned = "\n".join(no_comments_lines)

    parts = [s.strip() for s in cleaned.split(";")]
    return [p for p in parts if p]


def _run_statement(athena, workgroup: str, sql: str, output_loc: str) -> str:
    resp = athena.start_query_execution(
        QueryString=sql,
        WorkGroup=workgroup,
        ResultConfiguration={"OutputLocation": output_loc},
    )
    qid = resp["QueryExecutionId"]
    while True:
        info = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = info["Status"]["State"]
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            if state != "SUCCEEDED":
                reason = info["Status"].get("StateChangeReason", "(no reason)")
                raise RuntimeError(f"Athena query {qid} {state}: {reason}")
            return qid
        time.sleep(1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dev")
    parser.add_argument("--project", default="qdnu")
    parser.add_argument("--module", default="braket")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    storage_stack = f"{args.project}-{args.env}-{args.module}-storage"
    cf = boto3.client("cloudformation")
    outs = _stack_outputs(cf, storage_stack)
    db = outs.get("GlueDatabaseName")
    bucket = outs.get("ResultsBucketName")
    athena_bucket = outs.get("AthenaResultsBucketName")
    workgroup = outs.get("AthenaWorkgroupName")
    if not all([db, bucket, athena_bucket, workgroup]):
        log.error("missing outputs from stack %s: %s", storage_stack, outs)
        sys.exit(1)

    log.info("db=%s bucket=%s workgroup=%s", db, bucket, workgroup)

    template = DDL_PATH.read_text(encoding="utf-8")
    rendered = _render(template, db, bucket)

    if args.dry_run:
        print(rendered)
        return

    statements = _split_statements(rendered)
    output_loc = f"s3://{athena_bucket}/queries/"
    log.info("submitting %d statements", len(statements))

    athena = boto3.client("athena")
    for i, stmt in enumerate(statements, 1):
        head = stmt.split("\n", 1)[0][:80]
        log.info("[%d/%d] %s", i, len(statements), head)
        qid = _run_statement(athena, workgroup, stmt, output_loc)
        log.info("        -> %s OK", qid)

    log.info("done")


if __name__ == "__main__":
    main()
