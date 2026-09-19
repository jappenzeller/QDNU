"""
Run a sample Athena query against the deployed agate_results table.

Usage:
    python query_athena.py                          # default summary
    python query_athena.py --sql "SELECT ..."       # custom query
"""

from __future__ import annotations

import argparse
import logging
import time
from io import StringIO

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


DEFAULT_SQL = """
SELECT
    run_id,
    backend,
    patient_id,
    window_state,
    AVG(z_e) AS z_e_mean,
    AVG(z_i) AS z_i_mean,
    COUNT(*) AS n_channels
FROM qdnu_quantum_dev.agate_results
WHERE channel_index >= 0
GROUP BY run_id, backend, patient_id, window_state
ORDER BY patient_id, window_state
""".strip()


def _stack_outputs(stack: str) -> dict[str, str]:
    cf = boto3.client("cloudformation")
    outs = cf.describe_stacks(StackName=stack)["Stacks"][0].get("Outputs", []) or []
    return {o["OutputKey"]: o["OutputValue"] for o in outs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="dev")
    parser.add_argument("--project", default="qdnu")
    parser.add_argument("--module", default="braket")
    parser.add_argument("--sql", default=DEFAULT_SQL)
    args = parser.parse_args()

    storage_stack = f"{args.project}-{args.env}-{args.module}-storage"
    outs = _stack_outputs(storage_stack)
    workgroup = outs["AthenaWorkgroupName"]
    out_loc = f"s3://{outs['AthenaResultsBucketName']}/queries/"

    athena = boto3.client("athena")
    log.info("submitting query to workgroup=%s", workgroup)
    qid = athena.start_query_execution(
        QueryString=args.sql,
        WorkGroup=workgroup,
        ResultConfiguration={"OutputLocation": out_loc},
    )["QueryExecutionId"]

    while True:
        info = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = info["Status"]["State"]
        if state in {"SUCCEEDED", "FAILED", "CANCELLED"}:
            if state != "SUCCEEDED":
                reason = info["Status"].get("StateChangeReason", "(no reason)")
                raise SystemExit(f"query {qid} {state}: {reason}")
            break
        time.sleep(0.5)

    # Stream the result CSV from S3 and print it as a tidy table
    bucket = outs["AthenaResultsBucketName"]
    key = f"queries/{qid}.csv"
    body = boto3.client("s3").get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    print(body)


if __name__ == "__main__":
    main()
