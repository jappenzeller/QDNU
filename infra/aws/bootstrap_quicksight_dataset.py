"""
PROMPT AWS-005 + AWS-007 - Bootstrap QuickSight Athena data source + datasets.

Idempotent. Pre-flight checks abort with clear messages instead of producing
partial state. Stops at "dataset is visible and previewable in QuickSight".
No analysis, no dashboard.

Resources created (us-east-1):
  - QuickSight Athena data source: qdnu-athena
  - QuickSight dataset:            qdnu-braket-results   (DIRECT_QUERY)
  - QuickSight dataset:            qdnu-polarity-results (DIRECT_QUERY, AWS-007)

All datasets grant full owner permissions to the resolved QuickSight user.
The shared data source provisions once; each dataset is created or updated
in turn from its DATASETS spec.

Usage:
    python infra/aws/bootstrap_quicksight_dataset.py [--region us-east-1]
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
SQL_DIR = REPO_ROOT / "infra" / "aws"
COST_LOG_PATH = REPO_ROOT / "docs" / "aws" / "cost_log.md"

DEFAULT_REGION = "us-east-1"
NAMESPACE = "default"
DATA_SOURCE_ID = "qdnu-athena"
DATA_SOURCE_NAME = "qdnu-athena"
ATHENA_WORKGROUP = "qdnu-aws004"
GLUE_DB = "aws004"
EXPECTED_TABLES = ["braket_results", "polarity_results"]

# QuickSight's default service role. Athena queries from QuickSight assume
# this role, so it needs PutObject on the workgroup output bucket. The
# AWSQuickSightS3Policy managed policy grants Read on selected buckets but
# does NOT grant Write -- that has to be added explicitly per-bucket.
QUICKSIGHT_SERVICE_ROLE = "aws-quicksight-service-role-v0"
ATHENA_WRITE_INLINE_POLICY = "QuickSightAthenaWriteToQdnuBucket"


# Column lists for each dataset's CustomSql. Order MUST mirror the SELECT
# in the corresponding .sql file exactly.

BRAKET_COLUMNS: list[tuple[str, str]] = [
    # Identity / partition
    ("prompt_id", "STRING"),
    ("subject", "STRING"),
    ("device", "STRING"),
    ("channels", "INTEGER"),
    ("window_s", "DECIMAL"),
    ("encoding", "STRING"),
    ("adapter_applied", "BOOLEAN"),
    ("adapter_label", "STRING"),
    ("shots", "INTEGER"),
    # Pass status
    ("passed", "BOOLEAN"),
    ("pass_label", "STRING"),
    # Per-task scalars (repeated across 8 qubit rows in UNNEST'd dataset)
    ("aer_z_anc", "DECIMAL"),
    ("bkt_z_anc", "DECIMAL"),
    ("tvd", "DECIMAL"),
    ("z_anc_diff", "DECIMAL"),
    ("z_e_max_diff", "DECIMAL"),
    ("z_i_max_diff", "DECIMAL"),
    ("billable_seconds", "DECIMAL"),
    ("estimated_cost_usd", "DECIMAL"),
    ("actual_cost_usd", "DECIMAL"),
    # Per-qubit values from UNNEST
    ("qubit_index", "INTEGER"),
    ("aer_z_e_val", "DECIMAL"),
    ("aer_z_i_val", "DECIMAL"),
    ("bkt_z_e_val", "DECIMAL"),
    ("bkt_z_i_val", "DECIMAL"),
    # Derived per-qubit diffs
    ("z_e_diff", "DECIMAL"),
    ("z_i_diff", "DECIMAL"),
    # Provenance
    ("task_arn", "STRING"),
    ("region", "STRING"),
    ("segment_id", "STRING"),
    ("run_timestamp", "DATETIME"),
]

POLARITY_COLUMNS: list[tuple[str, str]] = [
    # Identity / partition
    ("prompt_id", "STRING"),
    ("subject", "STRING"),
    ("device", "STRING"),
    ("channels", "INTEGER"),
    ("window_s", "DECIMAL"),
    ("encoding", "STRING"),
    ("shots", "INTEGER"),
    # Display label
    ("encoding_config", "STRING"),
    # Polarity classification
    ("polarity_sign", "STRING"),
    ("polarity_strength", "STRING"),
    ("polarity_score", "INTEGER"),
    # AUC pair + calibration delta
    ("raw_auc", "DECIMAL"),
    ("oracle_cal_auc", "DECIMAL"),
    ("calibration_gain", "DECIMAL"),
    # Noise-floor flag
    ("at_noise_floor", "BOOLEAN"),
    # Provenance
    ("task_arn", "STRING"),
    ("region", "STRING"),
    ("run_timestamp", "DATETIME"),
]


# All datasets to provision. Each entry: an id (stable handle, used in URL +
# describe/update calls), a SQL filename, a column-types list, and a
# description. Adding a new dataset is one new entry here.
DATASETS: list[dict[str, Any]] = [
    {
        "id": "qdnu-braket-results",
        "name": "qdnu-braket-results",
        "sql_file": "quicksight_dataset_sql.sql",
        "columns": BRAKET_COLUMNS,
        "description": "Braket bridge regression validation (AWS-003 schema, UNNEST'd per-qubit)",
    },
    {
        "id": "qdnu-polarity-results",
        "name": "qdnu-polarity-results",
        "sql_file": "polarity_dataset_sql.sql",
        "columns": POLARITY_COLUMNS,
        "description": "IBM Heron polarity results (AWS-006 schema, per-task)",
    },
]


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _resolve_account_id() -> str:
    return boto3.client("sts").get_caller_identity()["Account"]


def _bucket_name(account_id: str) -> str:
    return f"qdnu-braket-{account_id}-{DEFAULT_REGION}"


def _ensure_quicksight_athena_write_policy(account_id: str) -> None:
    """Attach an inline policy on aws-quicksight-service-role-v0 granting
    PutObject + GetBucketLocation on the AWS-004 bucket so Athena queries
    from QuickSight can write results. The default AWSQuickSightS3Policy
    grants Read on selected buckets but not Write -- that has to be added
    per-bucket. Idempotent (put_role_policy overwrites by name).
    """
    bucket = _bucket_name(account_id)
    iam = boto3.client("iam")
    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "s3:ListBucketMultipartUploads",
                    "s3:GetBucketLocation",
                ],
                "Resource": f"arn:aws:s3:::{bucket}",
            },
            {
                "Effect": "Allow",
                "Action": [
                    "s3:PutObject",
                    "s3:AbortMultipartUpload",
                    "s3:ListMultipartUploadParts",
                ],
                "Resource": f"arn:aws:s3:::{bucket}/*",
            },
        ],
    }
    try:
        iam.put_role_policy(
            RoleName=QUICKSIGHT_SERVICE_ROLE,
            PolicyName=ATHENA_WRITE_INLINE_POLICY,
            PolicyDocument=json.dumps(policy),
        )
        log.info(
            "ensured inline policy %s on %s (Athena PutObject -> s3://%s/)",
            ATHENA_WRITE_INLINE_POLICY,
            QUICKSIGHT_SERVICE_ROLE,
            bucket,
        )
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "NoSuchEntity":
            log.warning(
                "QuickSight service role %s not found; skipping inline "
                "policy. Athena queries from QuickSight may fail with a "
                "permissions error until this role exists.",
                QUICKSIGHT_SERVICE_ROLE,
            )
        else:
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
    *, action: str, resource: str, quantity: float, unit: str, cost_usd: float
) -> None:
    _ensure_cost_log()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row = (
        f"| {ts} | {action} | {resource} | {quantity:g} | {unit} | "
        f"{cost_usd:.6f} |\n"
    )
    with COST_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(row)


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------


def _preflight_quicksight_enabled(qs, account_id: str) -> str:
    """Return the QuickSight identity region. Abort if QuickSight not enabled."""
    try:
        resp = qs.describe_account_settings(AwsAccountId=account_id)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in {"ResourceNotFoundException", "AccessDeniedException"}:
            log.error(
                "QuickSight is not enabled in this account. Enable it via the "
                "AWS Console at https://us-east-1.quicksight.aws.amazon.com/ "
                "(Standard or Enterprise edition; this is a one-time billed "
                "step). Then re-run this script."
            )
            raise SystemExit(2)
        raise
    settings = resp.get("AccountSettings", {})
    edition = settings.get("Edition", "UNKNOWN")
    log.info("QuickSight enabled (edition=%s)", edition)
    return settings.get("DefaultNamespace") or NAMESPACE


def _resolve_quicksight_user(qs, account_id: str, namespace: str, caller_arn: str) -> dict:
    """Return the dict for the QuickSight user matching the caller, or the
    only ADMIN if caller can't be matched.
    """
    try:
        resp = qs.list_users(AwsAccountId=account_id, Namespace=namespace)
    except ClientError as e:
        log.error(
            "Could not list QuickSight users in namespace '%s': %s. Make "
            "sure your IAM identity has the QuickSight admin permissions "
            "and that you've registered as a QuickSight user.",
            namespace,
            e,
        )
        raise SystemExit(2)

    users = resp.get("UserList", []) or []
    if not users:
        log.error(
            "No QuickSight users found in namespace '%s'. Create at least one "
            "Author seat via the QuickSight console first.",
            namespace,
        )
        raise SystemExit(2)

    # Try caller-IAM-arn match first (heuristic)
    iam_id = caller_arn.split("/")[-1].split(":")[-1].lower()
    for u in users:
        if iam_id and iam_id in u.get("UserName", "").lower():
            log.info("matched QuickSight user by caller IAM identity: %s",
                     u["UserName"])
            return u
    # Fall back to single ADMIN
    admins = [u for u in users if u.get("Role") == "ADMIN"]
    if len(admins) == 1:
        log.info("using sole QuickSight ADMIN: %s", admins[0]["UserName"])
        return admins[0]
    # Fall back to first AUTHOR
    authors = [u for u in users if u.get("Role") in ("AUTHOR", "ADMIN")]
    if authors:
        log.info("using first AUTHOR/ADMIN: %s", authors[0]["UserName"])
        return authors[0]

    log.error(
        "Could not resolve a QuickSight user for permissions. Found users: %s",
        [(u.get("UserName"), u.get("Role")) for u in users],
    )
    raise SystemExit(2)


def _preflight_athena_tables(glue, expected_tables: list[str]) -> None:
    for table in expected_tables:
        try:
            glue.get_table(DatabaseName=GLUE_DB, Name=table)
        except ClientError:
            log.error(
                "Athena table %s.%s not found. Run "
                "infra/aws/bootstrap_data_layer.py (AWS-004 and AWS-006) "
                "first.",
                GLUE_DB,
                table,
            )
            raise SystemExit(2)
        log.info("Athena table %s.%s confirmed", GLUE_DB, table)


# ---------------------------------------------------------------------------
# Data source + dataset
# ---------------------------------------------------------------------------


def _data_source_arn(account_id: str, region: str) -> str:
    return f"arn:aws:quicksight:{region}:{account_id}:datasource/{DATA_SOURCE_ID}"


def _dataset_arn(account_id: str, region: str, dataset_id: str) -> str:
    return f"arn:aws:quicksight:{region}:{account_id}:dataset/{dataset_id}"


def _data_source_permissions(user_arn: str) -> list[dict]:
    return [
        {
            "Principal": user_arn,
            "Actions": [
                "quicksight:DescribeDataSource",
                "quicksight:DescribeDataSourcePermissions",
                "quicksight:PassDataSource",
                "quicksight:UpdateDataSource",
                "quicksight:DeleteDataSource",
                "quicksight:UpdateDataSourcePermissions",
            ],
        }
    ]


def _dataset_permissions(user_arn: str) -> list[dict]:
    return [
        {
            "Principal": user_arn,
            "Actions": [
                "quicksight:DescribeDataSet",
                "quicksight:DescribeDataSetPermissions",
                "quicksight:PassDataSet",
                "quicksight:UpdateDataSet",
                "quicksight:DeleteDataSet",
                "quicksight:UpdateDataSetPermissions",
                "quicksight:CreateIngestion",
                "quicksight:CancelIngestion",
                "quicksight:DescribeIngestion",
                "quicksight:ListIngestions",
            ],
        }
    ]


def _ensure_data_source(qs, account_id: str, user_arn: str) -> str:
    """Create or update the qdnu-athena data source. Return its ARN."""
    params = {
        "AwsAccountId": account_id,
        "DataSourceId": DATA_SOURCE_ID,
        "Name": DATA_SOURCE_NAME,
        "Type": "ATHENA",
        "DataSourceParameters": {
            "AthenaParameters": {"WorkGroup": ATHENA_WORKGROUP}
        },
        "Permissions": _data_source_permissions(user_arn),
        "SslProperties": {"DisableSsl": False},
    }
    try:
        qs.describe_data_source(
            AwsAccountId=account_id, DataSourceId=DATA_SOURCE_ID
        )
        log.info("data source exists; updating: %s", DATA_SOURCE_ID)
        # update_data_source rejects keys other than the ones it allows.
        # Type and Permissions cannot be updated this way; Permissions has
        # its own API call below.
        update_params = {
            k: v
            for k, v in params.items()
            if k in {
                "AwsAccountId",
                "DataSourceId",
                "Name",
                "DataSourceParameters",
                "Credentials",
                "VpcConnectionProperties",
                "SslProperties",
            }
        }
        qs.update_data_source(**update_params)
        # Permissions are managed by a separate API. Skip on idempotent
        # re-runs to avoid racing UPDATE_IN_PROGRESS state and to keep
        # bootstrap a setup-only operation. To re-grant permissions, run
        # teardown_quicksight_dataset.py then bootstrap again.
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise
        log.info("creating data source: %s", DATA_SOURCE_ID)
        qs.create_data_source(**params)
        _append_cost_row(
            action="quicksight_data_source_create",
            resource=DATA_SOURCE_ID,
            quantity=1,
            unit="ds",
            cost_usd=0.0,
        )
    return _data_source_arn(account_id, qs.meta.region_name)


def _ensure_dataset(
    qs, account_id: str, user_arn: str, ds_arn: str, spec: dict[str, Any]
) -> tuple[str, str, str]:
    """Create or update one QuickSight dataset from a DATASETS spec entry.

    Returns (dataset_arn, dataset_id, action) where action is "created" or
    "updated".
    """
    dataset_id = spec["id"]
    dataset_name = spec["name"]
    sql_path = SQL_DIR / spec["sql_file"]
    sql = sql_path.read_text(encoding="utf-8")

    # QuickSight map keys must match [0-9a-zA-Z-]* -- no underscores allowed.
    pt_id = f"ptable-{dataset_id}"
    physical_map = {
        pt_id: {
            "CustomSql": {
                "DataSourceArn": ds_arn,
                "Name": f"{dataset_id}-flat",
                "SqlQuery": sql,
                "Columns": [
                    {"Name": col, "Type": typ} for (col, typ) in spec["columns"]
                ],
            }
        }
    }
    logical_map = {
        f"ltable-{dataset_id}": {
            "Alias": dataset_id,
            "Source": {"PhysicalTableId": pt_id},
            # No DataTransforms key: QuickSight requires this list to either
            # contain at least one transform or be absent entirely. Empty
            # list is a ParamValidationError. Transforms (when needed) live
            # in the analysis layer, not the dataset.
        }
    }
    params = {
        "AwsAccountId": account_id,
        "DataSetId": dataset_id,
        "Name": dataset_name,
        "PhysicalTableMap": physical_map,
        "LogicalTableMap": logical_map,
        "ImportMode": "DIRECT_QUERY",
        "Permissions": _dataset_permissions(user_arn),
    }
    action = "updated"
    try:
        qs.describe_data_set(AwsAccountId=account_id, DataSetId=dataset_id)
        log.info("dataset exists; updating: %s", dataset_id)
        update_params = {k: v for k, v in params.items() if k != "Permissions"}
        qs.update_data_set(**update_params)
        # Skip permissions on idempotent re-run; same reasoning as data source.
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
            raise
        log.info("creating dataset: %s", dataset_id)
        qs.create_data_set(**params)
        _append_cost_row(
            action="quicksight_dataset_create",
            resource=dataset_id,
            quantity=1,
            unit="ds",
            cost_usd=0.0,
        )
        action = "created"
    return _dataset_arn(account_id, qs.meta.region_name, dataset_id), dataset_id, action


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=DEFAULT_REGION)
    args = ap.parse_args()
    if args.region != DEFAULT_REGION:
        log.warning(
            "region '%s' is not %s; QuickSight identity region usually is "
            "us-east-1, cross-region setups are out of scope here.",
            args.region,
            DEFAULT_REGION,
        )

    account_id = _resolve_account_id()
    caller_arn = boto3.client("sts").get_caller_identity()["Arn"]
    log.info("account=%s caller=%s", account_id, caller_arn)

    qs = boto3.client("quicksight", region_name=args.region)
    glue = boto3.client("glue", region_name=args.region)

    namespace = _preflight_quicksight_enabled(qs, account_id)
    user = _resolve_quicksight_user(qs, account_id, namespace, caller_arn)
    user_arn = user["Arn"]
    log.info("user_arn=%s", user_arn)

    _preflight_athena_tables(glue, EXPECTED_TABLES)
    _ensure_quicksight_athena_write_policy(account_id)

    ds_arn = _ensure_data_source(qs, account_id, user_arn)
    log.info("data source ARN: %s", ds_arn)

    results: list[dict[str, str]] = []
    for spec in DATASETS:
        ds_dataset_arn, dataset_id, action = _ensure_dataset(
            qs, account_id, user_arn, ds_arn, spec
        )
        log.info("dataset %s (%s): %s", dataset_id, action, ds_dataset_arn)
        results.append({
            "id": dataset_id,
            "arn": ds_dataset_arn,
            "action": action,
            "description": spec["description"],
        })

    print()
    print("=" * 72)
    print("AWS-005 + AWS-007 QuickSight bootstrap complete")
    print("=" * 72)
    print(f"  region:              {args.region}")
    print(f"  account:             {account_id}")
    print(f"  data source ARN:     {ds_arn}")
    print(f"  QuickSight user ARN: {user_arn}")
    print()
    for r in results:
        print(f"  dataset:    {r['id']}  ({r['action']})")
        print(f"    arn:      {r['arn']}")
        print(f"    desc:     {r['description']}")
        print(f"    url:      https://{args.region}.quicksight.aws.amazon.com/sn/data-sets/{r['id']}")
        print()
    print("Next: return to your Claude chat for the analysis walkthrough.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
