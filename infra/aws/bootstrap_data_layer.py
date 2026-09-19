"""
PROMPT AWS-004 - Bootstrap the AWS data layer.

Idempotent. Re-runnable. Each step checks "exists?" before "create".

Resources created (all in us-east-1):
  - S3 bucket           qdnu-braket-{account-id}-us-east-1
  - Glue database       aws004
  - Glue table          aws004.braket_results
  - Athena workgroup    qdnu-aws004
  - S3 lifecycle rule   athena-results/ -> 7 day expiration
  - S3 schema upload    infra/glue_table_schema.json mirrored to bucket

Read infra/aws/glue_table_schema.json as the source of truth for the table.

Usage:
    python infra/aws/bootstrap_data_layer.py [--region us-east-1]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATHS = [
    REPO_ROOT / "infra" / "aws" / "glue_table_schema.json",
    REPO_ROOT / "infra" / "aws" / "polarity_glue_table_schema.json",
]

DEFAULT_REGION = "us-east-1"
WORKGROUP_NAME = "qdnu-aws004"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _bucket_name(account_id: str) -> str:
    return f"qdnu-braket-{account_id}-{DEFAULT_REGION}"


def _resolve_account_id() -> str:
    sts = boto3.client("sts")
    return sts.get_caller_identity()["Account"]


def _ensure_bucket(s3, bucket: str, region: str) -> bool:
    """Create bucket if missing. Returns True if newly created."""
    try:
        s3.head_bucket(Bucket=bucket)
        log.info("S3 bucket exists: %s", bucket)
        return False
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code not in {"404", "NoSuchBucket", "NotFound"}:
            # 403 means the bucket exists but is owned by another account or
            # we have no perms; we can't safely "fix" that here.
            raise
    log.info("creating S3 bucket: %s", bucket)
    if region == "us-east-1":
        # us-east-1 must NOT include LocationConstraint; AWS rejects it.
        s3.create_bucket(Bucket=bucket)
    else:
        s3.create_bucket(
            Bucket=bucket,
            CreateBucketConfiguration={"LocationConstraint": region},
        )
    return True


def _apply_bucket_security(s3, bucket: str) -> None:
    """Block public access + default SSE-S3 encryption + no versioning."""
    s3.put_public_access_block(
        Bucket=bucket,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True,
            "IgnorePublicAcls": True,
            "BlockPublicPolicy": True,
            "RestrictPublicBuckets": True,
        },
    )
    s3.put_bucket_encryption(
        Bucket=bucket,
        ServerSideEncryptionConfiguration={
            "Rules": [
                {
                    "ApplyServerSideEncryptionByDefault": {
                        "SSEAlgorithm": "AES256",
                    },
                    "BucketKeyEnabled": False,
                }
            ]
        },
    )
    # Versioning OFF (cost; we have local copies).
    s3.put_bucket_versioning(
        Bucket=bucket, VersioningConfiguration={"Status": "Suspended"}
    )
    log.info("applied bucket security: %s", bucket)


def _apply_lifecycle(s3, bucket: str) -> None:
    """athena-results/ expires after 7 days."""
    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "athena-results-expire-7d",
                    "Filter": {"Prefix": "athena-results/"},
                    "Status": "Enabled",
                    "Expiration": {"Days": 7},
                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1},
                },
            ]
        },
    )
    log.info("applied athena-results/ 7-day lifecycle: %s", bucket)


def _mirror_schemas_to_bucket(s3, bucket: str) -> None:
    """Copy each infra/aws/*_schema.json to s3://bucket/infra/ for provenance."""
    for schema_path in SCHEMA_PATHS:
        s3.put_object(
            Bucket=bucket,
            Key=f"infra/{schema_path.name}",
            Body=schema_path.read_bytes(),
            ContentType="application/json",
        )
        log.info("mirrored schema to s3://%s/infra/%s", bucket, schema_path.name)


def _ensure_glue_database(glue, database: str) -> None:
    try:
        glue.get_database(Name=database)
        log.info("Glue database exists: %s", database)
        return
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "EntityNotFoundException":
            raise
    glue.create_database(
        DatabaseInput={
            "Name": database,
            "Description": "AWS-004 processed records from Braket validation runs.",
        }
    )
    log.info("created Glue database: %s", database)


def _table_prefix(schema: dict[str, Any]) -> str:
    """S3 prefix this table's data lives under. One prefix per table.

    Convention:
      braket_results   -> processed/   (AWS-003, AWS-004)
      polarity_results -> polarity/    (AWS-006, future hardware ingests)
    """
    return {
        "braket_results": "processed",
        "polarity_results": "polarity",
    }[schema["table_name"]]


def _build_table_input(schema: dict[str, Any], bucket: str) -> dict[str, Any]:
    """Translate one *_schema.json into a Glue TableInput dict."""
    prefix = _table_prefix(schema)
    proj = schema["partition_projection"]
    table_properties: dict[str, str] = {
        "classification": "json",
        "projection.enabled": "true",
    }
    for col, cfg in proj.items():
        if cfg["type"] == "enum":
            table_properties[f"projection.{col}.type"] = "enum"
            table_properties[f"projection.{col}.values"] = ",".join(cfg["values"])
        else:
            raise ValueError(f"unsupported projection type: {cfg['type']}")
    table_properties["storage.location.template"] = (
        f"s3://{bucket}/{prefix}/"
        + "/".join(f"{k}=${{{k}}}" for k in proj.keys())
        + "/"
    )

    return {
        "Name": schema["table_name"],
        "Description": schema["description"],
        "TableType": "EXTERNAL_TABLE",
        "Parameters": table_properties,
        "PartitionKeys": [
            {"Name": k["Name"], "Type": k["Type"]}
            for k in schema["partition_keys"]
        ],
        "StorageDescriptor": {
            "Columns": [
                {"Name": c["Name"], "Type": c["Type"]}
                for c in schema["columns"]
            ],
            "Location": f"s3://{bucket}/{prefix}/",
            "InputFormat": schema["input_format"],
            "OutputFormat": schema["output_format"],
            "Compressed": False,
            "SerdeInfo": {
                "SerializationLibrary": schema["serde"],
                "Parameters": schema.get("serde_parameters", {}),
            },
            "StoredAsSubDirectories": False,
        },
    }


def _ensure_glue_table(glue, database: str, schema: dict[str, Any], bucket: str) -> None:
    table_input = _build_table_input(schema, bucket)
    table_name = schema["table_name"]
    try:
        glue.get_table(DatabaseName=database, Name=table_name)
        log.info(
            "Glue table %s.%s exists; updating to current schema", database, table_name
        )
        glue.update_table(DatabaseName=database, TableInput=table_input)
        return
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "EntityNotFoundException":
            raise
    glue.create_table(DatabaseName=database, TableInput=table_input)
    log.info("created Glue table %s.%s", database, table_name)


def _ensure_athena_workgroup(athena, workgroup: str, bucket: str) -> None:
    output_location = f"s3://{bucket}/athena-results/"
    config = {
        "ResultConfiguration": {
            "OutputLocation": output_location,
            "EncryptionConfiguration": {"EncryptionOption": "SSE_S3"},
        },
        "EnforceWorkGroupConfiguration": True,
        "PublishCloudWatchMetricsEnabled": True,
    }
    try:
        athena.get_work_group(WorkGroup=workgroup)
        log.info("Athena workgroup exists: %s; ensuring config", workgroup)
        athena.update_work_group(
            WorkGroup=workgroup,
            ConfigurationUpdates={
                "ResultConfigurationUpdates": {
                    "OutputLocation": output_location,
                    "EncryptionConfiguration": {"EncryptionOption": "SSE_S3"},
                },
                "EnforceWorkGroupConfiguration": True,
                "PublishCloudWatchMetricsEnabled": True,
            },
        )
        return
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") != "InvalidRequestException":
            # Older boto3 may surface as InvalidRequestException for "no such workgroup"
            if "not found" not in str(e).lower():
                raise
    athena.create_work_group(
        Name=workgroup,
        Description="AWS-004 workgroup for braket_results Athena queries.",
        Configuration=config,
    )
    log.info("created Athena workgroup: %s -> %s", workgroup, output_location)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default=DEFAULT_REGION)
    args = ap.parse_args()
    if args.region != DEFAULT_REGION:
        log.warning(
            "region '%s' is not %s; QuickSight cross-region setup will break later",
            args.region,
            DEFAULT_REGION,
        )

    log.info("resolving account id")
    account_id = _resolve_account_id()
    bucket = _bucket_name(account_id)
    log.info("account=%s bucket=%s", account_id, bucket)

    s3 = boto3.client("s3", region_name=args.region)
    glue = boto3.client("glue", region_name=args.region)
    athena = boto3.client("athena", region_name=args.region)

    schemas = [json.loads(p.read_text()) for p in SCHEMA_PATHS]
    databases = {s["database"] for s in schemas}
    if len(databases) != 1:
        raise ValueError(f"all schemas must share one database; got {databases}")
    database = databases.pop()

    _ensure_bucket(s3, bucket, args.region)
    _apply_bucket_security(s3, bucket)
    _apply_lifecycle(s3, bucket)
    _mirror_schemas_to_bucket(s3, bucket)

    _ensure_glue_database(glue, database)
    for schema in schemas:
        _ensure_glue_table(glue, database, schema, bucket)

    _ensure_athena_workgroup(athena, WORKGROUP_NAME, bucket)

    print()
    print("=" * 72)
    print("AWS-004 bootstrap complete")
    print("=" * 72)
    print(f"  region:           {args.region}")
    print(f"  account:          {account_id}")
    print(f"  bucket:           s3://{bucket}/")
    print(f"  glue db:          {database}")
    for schema in schemas:
        print(f"  glue table:       {database}.{schema['table_name']}")
    print(f"  athena workgroup: {WORKGROUP_NAME}")
    print()
    print("Verification command:")
    print("  python scripts/braket/athena_query.py")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
