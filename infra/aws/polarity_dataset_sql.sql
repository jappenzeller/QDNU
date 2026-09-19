-- PROMPT AWS-007 - Custom SQL the polarity QuickSight dataset wraps.
-- Source of truth. infra/aws/bootstrap_quicksight_dataset.py reads this file
-- and embeds it in the dataset definition.
--
-- Per-task shape (no UNNEST). 28 rows: 7 subjects x 4 encoding configs.
-- Per-qubit visuals are intentionally deferred until source data carries
-- per-qubit <Z> arrays (mean_z_e / mean_z_i are null for all IBM Heron
-- records; see docs/aws/polarity_schema_spec.md). When future device
-- ingests populate those arrays, a sibling SQL with UNNEST goes here.

SELECT
  -- Identity / partition columns
  prompt_id,
  subject,
  device,
  channels,
  window_s,
  encoding,
  shots,

  -- Encoding config as a clean human-readable label
  CASE
    WHEN channels = 8 AND window_s = 1.95 THEN 'CH8 @ 1.95s'
    WHEN channels = 8 AND window_s = 20.0  THEN 'CH8 @ 20s'
    WHEN channels = 12 THEN 'CH12'
    WHEN channels = 16 THEN 'CH16'
    ELSE CONCAT('CH', CAST(channels AS VARCHAR), ' @ ', CAST(window_s AS VARCHAR), 's')
  END AS encoding_config,

  -- Polarity classification
  polarity_sign,
  polarity_strength,

  -- Numeric polarity score for visuals that need a scalar (heatmaps, averages)
  CASE polarity_sign
    WHEN 'positive' THEN 1
    WHEN 'negative' THEN -1
    ELSE 0
  END AS polarity_score,

  -- AUC pair plus calibration gain (the headline "calibration helped by X" metric)
  raw_auc,
  oracle_cal_auc,
  (oracle_cal_auc - raw_auc) AS calibration_gain,

  -- Noise-floor flag for "exclude chance-level rows from this visual"
  CASE
    WHEN ABS(raw_auc - 0.5) < 0.05 THEN TRUE
    ELSE FALSE
  END AS at_noise_floor,

  -- Provenance
  task_arn,
  region,

  -- Timestamp cast to DATETIME for time-series binding
  CAST(from_iso8601_timestamp("timestamp") AS TIMESTAMP) AS run_timestamp

FROM aws004.polarity_results
