-- ============================================================================
-- Athena DDL for the Braket A-Gate result lake.
--
-- Bucket: s3://qdnu-braket-results/runs/
-- Layout: runs/{run_id}/backend={backend}/results.parquet
--
-- Apply with the Athena console or `aws athena start-query-execution`.
-- The database name (qdnu_quantum) and bucket (qdnu-braket-results) are
-- placeholders; substitute before running.
-- ============================================================================

CREATE DATABASE IF NOT EXISTS qdnu_quantum
COMMENT 'Quantum dataset lake: A-Gate results across Braket backends';

-- ----------------------------------------------------------------------------
-- 1. Raw results table
-- ----------------------------------------------------------------------------
-- Mirrors the JSON schema in infra/result_schema.json. Partitioned by
-- run_id and backend for cheap filtering. Use partition projection so we
-- don't have to MSCK REPAIR after every job.

CREATE EXTERNAL TABLE IF NOT EXISTS qdnu_quantum.agate_results (
    run_id            STRING,
    backend           STRING,
    patient_id        STRING,
    window_id         STRING,
    window_state      STRING,
    channel_index     INT,
    z_e               DOUBLE,
    z_i               DOUBLE,
    polarity_e        INT,
    polarity_i        INT,
    fid_ictal         DOUBLE,
    fid_inter         DOUBLE,
    fid_score         DOUBLE,
    shots             INT,
    wall_time_seconds DOUBLE,
    error             STRING,
    schema_version    STRING
)
STORED AS PARQUET
LOCATION 's3://qdnu-braket-results/parquet/';

-- ----------------------------------------------------------------------------
-- 2. Per-(patient, window, backend) summary view
-- ----------------------------------------------------------------------------
-- Aggregates per-channel polarities into a vector-friendly shape suitable
-- for the QuickSight dashboard.

CREATE OR REPLACE VIEW qdnu_quantum.v_window_polarity AS
SELECT
    run_id,
    backend,
    patient_id,
    window_id,
    window_state,
    COUNT(*)                                                        AS n_channels,
    -- Per-qubit <Z> aggregates
    AVG(z_e)                                                        AS mean_z_e,
    AVG(z_i)                                                        AS mean_z_i,
    ABS(AVG(z_e))                                                   AS polarity_strength_e,
    ABS(AVG(z_i))                                                   AS polarity_strength_i,
    CASE WHEN AVG(z_e) >= 0 THEN 1 ELSE -1 END                      AS polarity_dir_e,
    CASE WHEN AVG(z_i) >= 0 THEN 1 ELSE -1 END                      AS polarity_dir_i,
    -- Hellinger-to-template observable (matches Paper 1's polarity pipeline)
    AVG(fid_ictal)                                                  AS fid_ictal,
    AVG(fid_inter)                                                  AS fid_inter,
    AVG(fid_score)                                                  AS fid_score,
    CASE WHEN AVG(fid_score) >= 0 THEN 1 ELSE -1 END                AS polarity_dir_fid,
    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END)              AS error_channels
FROM qdnu_quantum.agate_results
WHERE channel_index >= 0
GROUP BY run_id, backend, patient_id, window_id, window_state;

-- ----------------------------------------------------------------------------
-- 3. Cross-backend agreement view
-- ----------------------------------------------------------------------------
-- For each (patient, window), counts how many backends agree on the
-- per-window polarity direction. This is the central dashboard signal:
-- a patient with high agreement across backends is "polarity-robust";
-- a patient with low agreement is "platform-sensitive".

CREATE OR REPLACE VIEW qdnu_quantum.v_polarity_agreement AS
WITH per_run AS (
    SELECT
        patient_id,
        window_id,
        window_state,
        backend,
        polarity_dir_e
    FROM qdnu_quantum.v_window_polarity
)
SELECT
    patient_id,
    window_id,
    window_state,
    COUNT(DISTINCT backend)                                                          AS n_backends,
    COUNT(DISTINCT CASE WHEN polarity_dir_e =  1 THEN backend END)                   AS n_backends_positive,
    COUNT(DISTINCT CASE WHEN polarity_dir_e = -1 THEN backend END)                   AS n_backends_negative,
    -- Fraction that agree with the majority direction
    GREATEST(
        COUNT(DISTINCT CASE WHEN polarity_dir_e =  1 THEN backend END),
        COUNT(DISTINCT CASE WHEN polarity_dir_e = -1 THEN backend END)
    ) * 1.0 / NULLIF(COUNT(DISTINCT backend), 0)                                     AS agreement_fraction
FROM per_run
GROUP BY patient_id, window_id, window_state;

-- ----------------------------------------------------------------------------
-- 4. Backend coverage view
-- ----------------------------------------------------------------------------
-- One row per (backend, run_id) for the dashboard's "executions" tile.

CREATE OR REPLACE VIEW qdnu_quantum.v_runs AS
SELECT
    run_id,
    backend,
    COUNT(DISTINCT patient_id)                          AS n_patients,
    COUNT(DISTINCT window_id)                           AS n_windows,
    COUNT(*)                                            AS n_rows,
    SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END)  AS n_errors,
    AVG(wall_time_seconds)                              AS mean_wall_time_seconds
FROM qdnu_quantum.agate_results
GROUP BY run_id, backend;
