"""
PROMPT AWS-004 - Tear down the AWS data layer.

Reverses bootstrap_data_layer.py. Idempotent in reverse: missing resources are
fine. Requires --i-mean-it; without it, prints what would be destroyed and
exits.

Order of operations (reverse of bootstrap):
  1. Delete Athena workgroup (with --recursive-delete-option)
  2. Delete Glue table
  3. Delete Glue database
  4. (optional) Empty + delete S3 bucket  [requires --include-bucket]

The S3 bucket is opt-in for deletion: it holds the only copy of historical
processed records. By default we keep it.

Usage:
    python infra/aws/teardown_data_layer.py                  # dry run
    python infra/aws/teardown_data_layer.py --i-mean-it      # tear down everything except S3 bucket
    python infra/aws/teardown_data_layer.py --i-mean-it --include-bucket  # also delete S3 bucket
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "infra" / "aws" / "glue_table_schema.json"
DEFAULT_REGION = "us-east-1"
WORKGROUP_NAME = "qdnu-aws004"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _bucket_name(account_id: str) -> str:
    return f"qdnu-braket-{account_id}-{DEFAULT_REGION}"


def _delete_athena_workgroup(athena, name: str) -> None:
    try:
        athena.delete_work_group(WorkGroup=name, RecursiveDeleteOption=True)
        log.info("deleted Athena workgroup %s", name)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in {"InvalidRequestException"}:
            log.info("Athena workgroup %s not found (ok)", name)
        else:
            raise


def _delete_glue_table(glue, database: str, table: str) -> None:
    try:
        glue.delete_table(DatabaseName=database, Name=table)
        log.info("deleted Glue table %s.%s", database, table)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "EntityNotFoundException":
            log.info("Glue table %s.%s not found (ok)", database, table)
        else:
            raise


def _delete_glue_database(glue, database: str) -> None:
    try:
        glue.delete_database(Name=database)
        log.info("deleted Glue database %s", database)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "EntityNotFoundException":
            log.info("Glue database %s not found (ok)", database)
        else:
            raise


def _empty_and_delete_bucket(s3, bucket: str) -> None:
    """Delete all objects + versions, then the bucket itself."""
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in {"404", "NoSuchBucket", "NotFound"}:
            log.info("S3 bucket %s not found (ok)", bucket)
            return
        raise

    log.info("emptying S3 bucket %s", bucket)
    paginator = s3.get_paginator("list_object_versions")
    for page in paginator.paginate(Bucket=bucket):
        objs = []
        for v in page.get("Versions", []) or []:
            objs.append({"Key": v["Key"], "VersionId": v["VersionId"]})
        for m in page.get("DeleteMarkers", []) or []:
            objs.append({"Key": m["Key"], "VersionId": m["VersionId"]})
        if objs:
            for chunk_start in range(0, len(objs), 1000):
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": objs[chunk_start:chunk_start + 1000]},
                )
    # Also fall back to plain object listing in case versioning was off
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", []) or []]
        if objs:
            for chunk_start in range(0, len(objs), 1000):
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": objs[chunk_start:chunk_start + 1000]},
                )

    s3.delete_bucket(Bucket=bucket)
    log.info("deleted S3 bucket %s", bucket)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--i-mean-it",
        action="store_true",
        help="actually destroy resources; without this, only print",
    )
    ap.add_argument(
        "--include-bucket",
        action="store_true",
        help="also empty + delete the S3 bucket (HISTORY LOSS)",
    )
    ap.add_argument("--region", default=DEFAULT_REGION)
    args = ap.parse_args()

    schema = json.loads(SCHEMA_PATH.read_text())
    database = schema["database"]
    table = schema["table_name"]

    sts = boto3.client("sts", region_name=args.region)
    account_id = sts.get_caller_identity()["Account"]
    bucket = _bucket_name(account_id)

    plan = [
        f"  Athena workgroup:  {WORKGROUP_NAME}",
        f"  Glue table:        {database}.{table}",
        f"  Glue database:     {database}",
    ]
    if args.include_bucket:
        plan.append(f"  S3 bucket:         {bucket}  (ALL OBJECTS DELETED)")
    else:
        plan.append(f"  S3 bucket:         {bucket}  (KEPT; pass --include-bucket to delete)")

    print()
    print("=" * 72)
    print(f"AWS-004 teardown plan (region={args.region}, account={account_id})")
    print("=" * 72)
    for line in plan:
        print(line)
    print()

    if not args.i_mean_it:
        print("(dry run; pass --i-mean-it to execute)")
        print("=" * 72)
        return 0

    print("Executing teardown...")
    print("=" * 72)

    athena = boto3.client("athena", region_name=args.region)
    glue = boto3.client("glue", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)

    _delete_athena_workgroup(athena, WORKGROUP_NAME)
    _delete_glue_table(glue, database, table)
    _delete_glue_database(glue, database)
    if args.include_bucket:
        _empty_and_delete_bucket(s3, bucket)

    print()
    print("teardown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
