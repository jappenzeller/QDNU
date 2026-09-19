-- PROMPT AWS-005 - Custom SQL the QuickSight dataset wraps.
-- Source of truth. infra/aws/bootstrap_quicksight_dataset.py reads this file
-- and embeds it in the dataset definition.
--
-- UNNEST'd to per-qubit granularity: each task row produces 8 rows
-- (one per channel/qubit pair). Scalar per-task metrics (tvd, costs)
-- repeat across the 8 rows; in visuals use AVG/MAX over qubit_index to
-- dedupe.

SELECT
  -- Identity / partition columns
  prompt_id,
  subject,
  device,
  channels,
  window_s,
  encoding,
  adapter_applied,
  CASE WHEN adapter_applied THEN 'Adapter ON' ELSE 'Adapter OFF' END AS adapter_label,
  shots,

  -- Pass status (passed for filters/aggregation, pass_label for display)
  "pass" AS passed,
  CASE WHEN "pass" THEN 'PASS' ELSE 'FAIL' END AS pass_label,

  -- Per-task scalar metrics (repeated 8x after UNNEST)
  aer.z_ancilla        AS aer_z_anc,
  braket.z_ancilla     AS bkt_z_anc,
  diff.tvd             AS tvd,
  diff.z_ancilla_abs   AS z_anc_diff,
  diff.z_e_max_abs     AS z_e_max_diff,
  diff.z_i_max_abs     AS z_i_max_diff,
  billable_seconds,
  estimated_cost_usd,
  actual_cost_usd,

  -- Per-qubit (UNNEST'd)
  qubit_index,
  aer_z_e_val,
  aer_z_i_val,
  bkt_z_e_val,
  bkt_z_i_val,
  ABS(aer_z_e_val - bkt_z_e_val) AS z_e_diff,
  ABS(aer_z_i_val - bkt_z_i_val) AS z_i_diff,

  -- Provenance
  task_arn,
  region,
  segment_id,

  -- Timestamp cast to DATETIME for QuickSight time-series binding
  CAST(from_iso8601_timestamp("timestamp") AS TIMESTAMP) AS run_timestamp

FROM aws004.braket_results
CROSS JOIN UNNEST(
  aer.z_e,
  aer.z_i,
  braket.z_e,
  braket.z_i
) WITH ORDINALITY AS t(
  aer_z_e_val,
  aer_z_i_val,
  bkt_z_e_val,
  bkt_z_i_val,
  qubit_index
)
