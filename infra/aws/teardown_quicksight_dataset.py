"""
PROMPT AWS-005 + AWS-007 - Tear down QuickSight datasets + data source.

Reverses bootstrap_quicksight_dataset.py. Idempotent: missing resources are
fine. Requires --i-mean-it; without it, prints what would be destroyed.

Order of operations (reverse of bootstrap):
  1. Delete each dataset in DATASET_IDS
  2. Delete the shared data source qdnu-athena
  3. (optional, --full-iam-cleanup) detach the QuickSight inline policy from
     the QuickSight service role

Does NOT touch the AWS-004 layer (Athena workgroup, Glue, S3) -- those have
their own teardown script. Does NOT delete or modify the QuickSight service
role itself by default.

Usage:
    python infra/aws/teardown_quicksight_dataset.py                       # dry run
    python infra/aws/teardown_quicksight_dataset.py --i-mean-it           # destroy datasets + data source
    python infra/aws/teardown_quicksight_dataset.py --i-mean-it --full-iam-cleanup
"""
from __future__ import annotations

import argparse
import logging
import sys

import boto3
from botocore.exceptions import ClientError


DEFAULT_REGION = "us-east-1"
DATA_SOURCE_ID = "qdnu-athena"
DATASET_IDS = ["qdnu-braket-results", "qdnu-polarity-results"]

QUICKSIGHT_SERVICE_ROLE = "aws-quicksight-service-role-v0"
ATHENA_WRITE_INLINE_POLICY = "QuickSightAthenaWriteToQdnuBucket"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _delete_dataset(qs, account_id: str, dataset_id: str) -> None:
    try:
        qs.delete_data_set(AwsAccountId=account_id, DataSetId=dataset_id)
        log.info("deleted QuickSight dataset %s", dataset_id)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            log.info("QuickSight dataset %s not found (ok)", dataset_id)
        else:
            raise


def _delete_data_source(qs, account_id: str, ds_id: str) -> None:
    try:
        qs.delete_data_source(AwsAccountId=account_id, DataSourceId=ds_id)
        log.info("deleted QuickSight data source %s", ds_id)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
            log.info("QuickSight data source %s not found (ok)", ds_id)
        else:
            raise


def _detach_iam_inline_policy() -> None:
    iam = boto3.client("iam")
    try:
        iam.delete_role_policy(
            RoleName=QUICKSIGHT_SERVICE_ROLE,
            PolicyName=ATHENA_WRITE_INLINE_POLICY,
        )
        log.info(
            "detached inline policy %s from %s",
            ATHENA_WRITE_INLINE_POLICY,
            QUICKSIGHT_SERVICE_ROLE,
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in {"NoSuchEntity"}:
            log.info(
                "inline policy %s on %s not found (ok)",
                ATHENA_WRITE_INLINE_POLICY,
                QUICKSIGHT_SERVICE_ROLE,
            )
        else:
            raise


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--i-mean-it",
        action="store_true",
        help="actually destroy resources",
    )
    ap.add_argument(
        "--full-iam-cleanup",
        action="store_true",
        help="also detach the QuickSight inline IAM policy "
        "(QuickSightAthenaWriteToQdnuBucket on aws-quicksight-service-role-v0)",
    )
    ap.add_argument("--region", default=DEFAULT_REGION)
    args = ap.parse_args()

    sts = boto3.client("sts", region_name=args.region)
    account_id = sts.get_caller_identity()["Account"]

    print()
    print("=" * 72)
    print(f"AWS-005 + AWS-007 teardown plan (region={args.region}, account={account_id})")
    print("=" * 72)
    for ds_id in DATASET_IDS:
        print(f"  QuickSight dataset:     {ds_id}")
    print(f"  QuickSight data source: {DATA_SOURCE_ID}")
    if args.full_iam_cleanup:
        print(f"  IAM inline policy:      {ATHENA_WRITE_INLINE_POLICY} on {QUICKSIGHT_SERVICE_ROLE}")
    print()

    if not args.i_mean_it:
        print("(dry run; pass --i-mean-it to execute)")
        print("=" * 72)
        return 0

    print("Executing teardown...")
    print("=" * 72)

    qs = boto3.client("quicksight", region_name=args.region)
    for ds_id in DATASET_IDS:
        _delete_dataset(qs, account_id, ds_id)
    _delete_data_source(qs, account_id, DATA_SOURCE_ID)
    if args.full_iam_cleanup:
        _detach_iam_inline_policy()

    print()
    print("teardown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
