# AWS-005 QuickSight Dataset Runbook

## What got created

| Resource | ID | Notes |
|---|---|---|
| QuickSight Athena data source | `qdnu-athena` | Wraps the AWS-004 Athena workgroup `qdnu-aws004` |
| QuickSight dataset | `qdnu-braket-results` | DIRECT_QUERY mode; flattened custom-SQL view of `aws004.braket_results` |

The exact ARNs are printed by `infra/aws/bootstrap_quicksight_dataset.py` on
each run. Re-running the script prints them again.

The custom SQL the dataset wraps lives in
[`infra/aws/quicksight_dataset_sql.sql`](../../infra/aws/quicksight_dataset_sql.sql).
Edit there, not in the QuickSight UI.

## How to open the dataset in QuickSight

The bootstrap script prints a URL of this shape:

```
https://us-east-1.quicksight.aws.amazon.com/sn/data-sets/qdnu-braket-results
```

Open it in a browser. You should see:
- A dataset titled `qdnu-braket-results`
- A field list on the left including `prompt_id`, `subject`, `device`,
  `passed`, `tvd`, `aer_z_anc`, `bkt_z_anc`, `actual_cost_usd`, etc.
- A preview of ~2 rows (the AWS-003 Path A and Path B records)

If the preview is empty or shows an error, see "Debugging" below.

## How to refresh

DIRECT_QUERY datasets don't cache. Every visual interaction (filter, sort,
drill, opening an analysis) issues a fresh Athena query. New data appears
automatically: run `scripts/braket/upload_results_to_s3.py` to push more
JSONs into S3, then refresh any open visual in QuickSight.

There's no manual refresh step. There's no schedule to configure. If the
data appears in `python scripts/braket/athena_query.py` output, it's also
queryable from QuickSight.

## How to add columns

1. Edit `infra/aws/quicksight_dataset_sql.sql` — add the new column to the
   `SELECT` list.
2. Add a matching entry to `COLUMN_TYPES` in
   `infra/aws/bootstrap_quicksight_dataset.py` (column name + QuickSight
   type: `STRING`, `INTEGER`, `DECIMAL`, `BOOLEAN`, `DATETIME`).
3. Re-run `python infra/aws/bootstrap_quicksight_dataset.py`. The dataset's
   custom SQL is updated in place.
4. Existing analyses keep working as long as you didn't drop a column they
   reference.

## How to remove columns

Same flow but in reverse. Be aware: any analysis or dashboard that
references a removed column will show the column as missing/error after the
update. Dashboards survive the operation; you'll just need to remove the
broken visual elements manually.

## Debugging

| Symptom | Where to look |
|---|---|
| Bootstrap aborts with "QuickSight is not enabled" | The QuickSight account hasn't been created. Open `https://us-east-1.quicksight.aws.amazon.com/` and complete the one-time enable flow (Standard or Enterprise). Then re-run. |
| Bootstrap aborts with "no QuickSight users found" | You're not a QuickSight user yet. Add yourself as an Author in the QuickSight admin console. |
| Bootstrap aborts with "Athena table not found" | Run `python infra/aws/bootstrap_data_layer.py` (AWS-004) first. |
| Dataset preview is empty | Run `python scripts/braket/athena_query.py` directly. If that returns rows, the data is there but QuickSight permissions might be off — re-run the bootstrap to refresh permissions. If `athena_query.py` is also empty, run `python scripts/braket/upload_results_to_s3.py`. |
| Dataset preview shows an error about column types | The custom SQL columns don't match the `COLUMN_TYPES` declaration in the bootstrap script. Check the `SELECT` list against the type list line-by-line. |
| "Permission denied" when opening the URL | The QuickSight user the bootstrap resolved isn't you. Check the printed `QuickSight user ARN` against `aws sts get-caller-identity`. If they don't match, manually grant yourself permissions in the QuickSight UI under the dataset's "Share" menu. |

## Cost

QuickSight billing is per-user-per-month, charged separately and not
captured in `cost_log.md`. The data source and dataset have no per-resource
fees. Per-query Athena cost is logged under `athena_scan` rows by
`scripts/braket/athena_query.py`.

DIRECT_QUERY queries triggered by QuickSight visuals also incur Athena scan
cost. At AWS-003 data volume (kilobytes), this is functionally zero, but
each visual interaction is one Athena query — heavy dashboard use will show
up as `athena_scan` rows for queries you didn't run from the CLI.

## Tearing down

```bash
python infra/aws/teardown_quicksight_dataset.py                # dry run
python infra/aws/teardown_quicksight_dataset.py --i-mean-it    # destroy
```

This removes only the dataset and data source. The AWS-004 data layer
(Athena, Glue, S3) is untouched. Use `infra/aws/teardown_data_layer.py` for
that.

## Next step

The dataset is now ready. Return to your Claude chat and ask Claude to walk
you through building the analysis. The UI work is intentionally not in this
runbook because it's iterative and visual — Claude in chat is the right
tool for it, not Claude Code.
