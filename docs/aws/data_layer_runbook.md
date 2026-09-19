# AWS-004 Data Layer Runbook

## What this is

The AWS-native data layer for processed Braket validation records. One S3
bucket, one Glue catalog, one Athena workgroup — all in `us-east-1`. The
locked schema (set in AWS-003) lives in
[`infra/aws/glue_table_schema.json`](../../infra/aws/glue_table_schema.json).
Records flow: local JSON → S3 (Hive-partitioned) → Athena query.

No Lambda triggers, no DynamoDB. Manual upload is enough at our data volume;
revisit when there's a reason to automate. QuickSight is AWS-005.

## First-time setup

```bash
pip install boto3
aws configure                                                 # or aws sso login
python infra/aws/bootstrap_data_layer.py
python scripts/braket/upload_results_to_s3.py
python scripts/braket/athena_query.py
```

Expected output of the last command: 2 rows from AWS-003 (Path A and Path B),
both `pass=true`, TVD around 0.039.

## Adding a new device

Edit `infra/aws/glue_table_schema.json`, append the new value to
`partition_projection.device.values`. Re-run `python infra/aws/bootstrap_data_layer.py`
(idempotent; updates the table in place).

## Adding a new prompt_id

Same shape: append to `partition_projection.prompt_id.values` in the schema
JSON, re-run bootstrap.

## Adding a new subject

Same shape: append to `partition_projection.subject.values`, re-run bootstrap.

## Debugging

| Symptom | Where to look |
|---|---|
| Upload says EXISTS but file looks new | object's S3 key collides on `(prompt_id, device, subject, timestamp)`; bump the source JSON's timestamp or check for duplicates |
| Athena query returns 0 rows | check the partitioned S3 prefix: `aws s3 ls s3://qdnu-braket-{account}-us-east-1/processed/ --recursive`; if files are there, table state is `aws glue get-table --database-name aws004 --name braket_results` |
| Athena query fails with SerDe error | look at the Athena Query History (Athena console → Query history → click the failed query). The error often points at a single row whose JSON shape doesn't match the schema |
| Bucket creation fails with 409 | someone already owns that bucket name; check if `head_bucket` returns 403 (means you don't own it). Bucket names are global. Should not happen with the `qdnu-braket-{account-id}-us-east-1` convention because account IDs are unique. |
| `pass` column unparseable | `pass` is SQL-reserved; quote it: `"pass"` |

To inspect SerDe parsing errors specifically: run the query, then `aws athena get-query-execution --query-execution-id {qid}` and look at `Status.AthenaError`.

## Cost monitoring

[`docs/aws/cost_log.md`](cost_log.md) is the append-only log; every S3 PUT and
Athena scan adds a row. Total spend should stay under $0.10 for the lifetime
of AWS-003-scale data.

When to be concerned:

- Any single `athena_scan` row over **$0.01** indicates the table is
  effectively unpartitioned. Expected per-query scan: a few KB. If you see
  10 MB+ scanned, something fell back to a full-table scan — usually
  partition projection misconfigured or the WHERE clause doesn't filter on a
  partition column.
- `s3_put` rows summing over **$0.001** mean someone uploaded thousands of
  files. Sanity check the upload glob.
- Glue catalog has a 1M-object free tier; we're nowhere near it.

## Tearing down

```bash
python infra/aws/teardown_data_layer.py                                 # dry run
python infra/aws/teardown_data_layer.py --i-mean-it                     # remove Athena+Glue, keep bucket
python infra/aws/teardown_data_layer.py --i-mean-it --include-bucket    # also delete bucket (HISTORY LOSS)
```

The bucket is opt-in for deletion because it's the only AWS-side copy of
processed records. If you delete it and want it back, re-run
`bootstrap_data_layer.py` then `upload_results_to_s3.py` (local JSONs are the
canonical source).
