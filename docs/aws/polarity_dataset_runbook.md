# AWS-007 Polarity Dataset Runbook

## What got created

| Resource | ID | Notes |
|---|---|---|
| QuickSight Athena data source | `qdnu-athena` | Shared with the regression dataset; created in AWS-005 |
| QuickSight dataset | `qdnu-polarity-results` | DIRECT_QUERY mode; per-task view of `aws004.polarity_results` |

Exact ARNs print on each `bootstrap_quicksight_dataset.py` run. Both
datasets (regression + polarity) print on the same run.

The custom SQL the dataset wraps lives in
[`infra/aws/polarity_dataset_sql.sql`](../../infra/aws/polarity_dataset_sql.sql).
Edit there, not in the QuickSight UI.

## What's in the dataset

**Per-task shape: 28 rows, one per (subject × encoding-config).** UNNEST
is intentionally absent — the AWS-006 IBM Heron ingest left `mean_z_e` and
`mean_z_i` arrays as `null` (source data doesn't carry per-channel ⟨Z⟩),
and `CROSS JOIN UNNEST(NULL)` would drop every row. When future device
ingests populate per-qubit arrays, a sibling per-qubit dataset goes here.

Columns (18 total):

- **Identity / config**: `prompt_id`, `subject`, `device`, `channels`,
  `window_s`, `encoding`, `shots`, `encoding_config`
- **Polarity classification**: `polarity_sign` (`positive`/`negative`/`null`),
  `polarity_strength` (`strong`/`weak`), `polarity_score` (-1, 0, +1)
- **AUC + calibration**: `raw_auc`, `oracle_cal_auc`, `calibration_gain`
  (= `oracle_cal_auc - raw_auc`)
- **Quality flag**: `at_noise_floor` (TRUE when |raw_auc - 0.5| < 0.05)
- **Provenance**: `task_arn`, `region`, `run_timestamp` (DATETIME)

`encoding_config` is the dashboard-friendly label
(`CH8 @ 1.95s`, `CH8 @ 20s`, `CH12`, `CH16`). `polarity_score` is INTEGER
so heatmap legends and tooltips render `+1` / `-1` rather than `1.0`.

## How to refresh

DIRECT_QUERY. No cache, no schedule. Every visual interaction issues a
fresh Athena query against `aws004.polarity_results`. Run
`scripts/aws/ingest_ibm_heron_results.py` (or future device ingests) to
push more records into S3; they appear in QuickSight immediately on next
visual interaction.

## How to add columns

Two-place edit:

1. Edit `infra/aws/polarity_dataset_sql.sql` — add the column to the
   `SELECT` list.
2. Add the matching `(name, QUICKSIGHT_TYPE)` entry to `POLARITY_COLUMNS`
   in `infra/aws/bootstrap_quicksight_dataset.py`.
3. Re-run `python infra/aws/bootstrap_quicksight_dataset.py`. Both
   datasets are updated in place; the regression dataset is unaffected.

The two-place edit is necessary because QuickSight requires the column
list explicitly — inferring from CustomSql is unreliable for derived /
nested fields.

## How to add a new device

The schema is forward-compatible for IonQ Aria, IQM Garnet, Rigetti, etc.

1. Write or extend an ingest script alongside
   `scripts/aws/ingest_ibm_heron_results.py`. Map the device's source data
   to the `polarity_results` schema; populate `mean_z_e` / `mean_z_i`
   arrays if the device captures per-qubit ⟨Z⟩.
2. If the new device introduces values not in the current enums, edit the
   `device` (and optionally `prompt_id`) lists in
   `infra/aws/polarity_glue_table_schema.json` (and the cross-table
   `glue_table_schema.json` for consistency).
3. Re-run `python infra/aws/bootstrap_data_layer.py` (idempotent).
4. Run the device-specific ingest.

QuickSight needs no changes — DIRECT_QUERY picks up the new rows on next
visual interaction.

## Tearing down

```bash
python infra/aws/teardown_quicksight_dataset.py                   # dry run
python infra/aws/teardown_quicksight_dataset.py --i-mean-it       # destroy both datasets + data source
python infra/aws/teardown_quicksight_dataset.py --i-mean-it --full-iam-cleanup
```

`--full-iam-cleanup` also detaches the
`QuickSightAthenaWriteToQdnuBucket` inline policy from the QuickSight
service role. Skip this flag if you'll re-bootstrap soon.

The AWS-004/006 data layer (Athena, Glue, S3) is untouched. Use
`infra/aws/teardown_data_layer.py` for that.

## Cost

QuickSight billing is per-user-per-month, charged separately and not
captured in `cost_log.md`. The data source and datasets have no
per-resource fees. Per-query Athena cost is logged under `athena_scan`
rows — DIRECT_QUERY visuals each trigger one Athena query.

At 28 rows of polarity data, every query scans kilobytes; cost is
functionally zero.

## Next step

The polarity dataset is now ready. Return to your Claude chat and ask
Claude to walk you through building the polarity analysis. The UI work is
intentionally not in this runbook because it's iterative and visual —
Claude in chat is the right tool for it, not Claude Code.
