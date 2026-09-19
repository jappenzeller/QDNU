# `aws004.polarity_results` Schema Spec

Sister table to `aws004.braket_results`. Holds per-(subject × encoding-config)
polarity records from real-hardware quantum runs. Currently IBM Heron;
forward-compatible with IonQ, Rigetti, IQM.

Layout:

```
s3://qdnu-braket-{account}-us-east-1/polarity/
  prompt_id={...}/device={...}/subject={...}/{timestamp}__{config}.json
```

One NDJSON-compact record per file. Same OpenX SerDe as `braket_results`.

---

## Source contract

The IBM Heron data lives under `results/`. Inspection of the canonical
sources used by the AWS-006 ingest:

### `results/validation/polarity_consistency_audit.json` (primary source)

Top-level keys:

```
prompt              str    "PROMPT 029"
task                str    "polarity consistency audit"
timestamp           str    ISO8601, audit run time
runs_loaded         list   ["CH8_1.95s", "CH8_20s", "CH12", "CH16"]
auc_matrix          dict   subject -> { config -> raw_auc }
polarity_matrix     dict   subject -> { config -> "standard" | "inverted" | "chance" }
consistency         dict   subject -> { polarities, consistent, dominant, n_runs_with_signal }
summary             dict   prediction_met, chb14_always_inverted, ...
```

Sample (`auc_matrix['chb01']`):

```
{"CH8_1.95s": 0.6857, "CH8_20s": 0.5286, "CH12": 0.4786, "CH16": 0.5}
```

Sample (`polarity_matrix['chb01']`):

```
{"CH8_1.95s": "standard", "CH8_20s": "standard", "CH12": "inverted", "CH16": "chance"}
```

Sample (`consistency['chb03']`):

```
{
  "polarities": ["inverted", "inverted", "inverted"],
  "consistent": true,
  "dominant": "inverted",
  "n_runs_with_signal": 3
}
```

The audit aggregates four prior runs into a 4-config × 7-subject matrix.
This is the canonical source for AWS-006 ingestion.

### `results/window_analysis/quantum_20s_hardware.json` (CH8_20s enrichment)

Provides `calibrated_auc` for the CH8 @ 20s configuration that the audit
file does not carry. Only field used: `per_subject[subject].calibrated_auc`.

### Source data caveats

- **No per-channel `mean_z_e`, `mean_z_i`, `covariance_shift`.** Source
  files carry only aggregate AUC and polarity labels; per-channel `<Z>`
  values weren't preserved through the original IBM Quantum run pipeline.
  The schema fields exist but are written as `null` for every IBM Heron
  record. Future device runs (where per-channel `<Z>` is captured) populate
  these fields.
- **No IBM Quantum job IDs in the audit.** The original hardware-validation
  files have them but the audit-time aggregation drops them. `task_arn` is
  the empty string for IBM Heron records.
- **No `device_arn`.** IBM Heron isn't in the Braket device catalog, so no
  Braket-format ARN exists. Field is empty string.
- **Single audit timestamp.** All 28 records (7 subjects × 4 configs) carry
  the same `timestamp`: when the audit ran. The hardware tasks themselves
  ran earlier, but no per-task timestamp survives the audit aggregation.

---

## Target schema

Top-level columns (non-partition):

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | string | ISO8601 record creation time (audit timestamp for IBM) |
| `channels` | int | 8, 12, or 16 |
| `window_s` | double | EEG window in seconds (1.95 or 20.0) |
| `encoding` | string | e.g. `V3_PLV_theta_alpha` |
| `shots` | int | 1024 for IBM Heron |
| `raw_auc` | double | LOSO AUC, no calibration |
| `oracle_cal_auc` | double | LOSO AUC with oracle (true sign) calibration |
| `polarity_sign` | string | `positive` / `negative` / `null` |
| `polarity_strength` | string | `strong` / `weak` |
| `mean_z_e` | array<double> | per-channel mean ⟨Z_E_i⟩, length = channels (or null) |
| `mean_z_i` | array<double> | per-channel mean ⟨Z_I_i⟩, length = channels (or null) |
| `covariance_shift` | array<double> | per-channel scalar shift (or null) |
| `adapter_applied` | boolean | always `false` for IBM (workaround is Braket-only) |
| `notes` | string | free text from source JSON if present |
| `device_arn` | string | empty string for IBM (no Braket-catalog ARN) |
| `task_arn` | string | IBM Quantum job ID (opaque); empty if unknown |
| `region` | string | always `us-east-1` for downstream uniformity |

Partition columns (Hive-style, in the S3 path):

| Column | Type | Values (current) |
|---|---|---|
| `prompt_id` | string | `AWS-003`, `IBM-HERON` |
| `device` | string | `SV1`, `default_simulator`, `ibm_torino`, `Aria-1`, `Forte-1`, `Garnet` |
| `subject` | string | `chb01`, `chb03`, `chb05`, `chb07`, `chb11`, `chb14`, `chb21` |

---

## Derivation rules

### `polarity_sign` mapping

The source uses one set of labels; the schema uses another. The mapping is
exact:

| Source label | Schema value |
|---|---|
| `standard` | `positive` |
| `inverted` | `negative` |
| `chance` | `null` (literal string, not SQL NULL) |

`chance` corresponds to `n_runs_with_signal` excluding that config from the
consistency check — i.e., `\|raw_auc - 0.5\| < ε`. The audit uses
ε = 0.05 implicitly. We preserve this; if a future ingest needs a different
threshold, document it and bump the schema version.

### `polarity_strength` derivation

Lifted directly from the audit's `consistency.consistent` field:

| Audit value | Schema value |
|---|---|
| `consistent: true` | `strong` |
| `consistent: false` | `weak` |

The audit excludes `chance` configs from the consistency check, which
matches our preferred semantics: a subject with three signal-bearing configs
that all agree is `strong` even if a fourth config returned `chance`.

Per project memory, the expected strength assignments are:

| Subject | Expected strength |
|---|---|
| `chb03` | strong |
| `chb21` | strong |
| `chb01` | weak |
| `chb05` | weak |
| `chb07` | weak |
| `chb14` | weak |
| `chb11` | derive from data (not pre-classified) |

The ingest script prints a `DISCREPANCY` warning if a derived value
contradicts the expectation but writes the derived value anyway. Don't
silently override.

### `oracle_cal_auc` derivation

When the source carries an explicit `calibrated_auc` (currently only the
CH8_20s file), use it directly.

When absent, derive: `oracle_cal_auc = max(raw_auc, 1 - raw_auc)`. This is
the standard polarity-flip oracle calibration: under the assumption that
true polarity is known, scores from inverted-polarity subjects are flipped,
yielding `1 - raw_auc`.

---

## Adding a new device

When ingesting future IonQ Aria, IQM Garnet, Rigetti, etc. results:

1. **Update enums.** Edit
   `infra/aws/polarity_glue_table_schema.json` (and
   `infra/aws/glue_table_schema.json` for cross-table consistency) to append
   the new device value to `partition_projection.device.values`.

2. **Re-run bootstrap.**
   `python infra/aws/bootstrap_data_layer.py` is idempotent; it updates the
   table in place.

3. **Write a per-device ingest script.** Mirror
   `scripts/aws/ingest_ibm_heron_results.py`. Keep the schema mapping the
   same: source-specific code maps source fields to the schema columns.
   Don't add new columns unless the new device captures something
   genuinely new (per-channel `<Z>` is the obvious candidate; populate
   `mean_z_e`/`mean_z_i` arrays if they're available).

4. **Set `adapter_applied`** based on whether the bridge adapter from
   AWS-002 was used. For IBM hardware (uses Qiskit directly): `false`. For
   Braket hardware (uses qiskit-braket-provider via SV1, Aria, Forte,
   Garnet): `true` if `prepare_circuit_for_braket` was applied, else
   `false`.

5. **Pick a partition value** for `prompt_id`. Future Braket QPU
   submissions can use `AWS-007`, `AWS-008`, etc. Add to
   `partition_projection.prompt_id.values` and re-bootstrap.

---

## Cross-table joins

`braket_results` and `polarity_results` share `subject` (and on records
where applicable, `device`). Example: cross-check that adapter-on Braket
SV1 records show consistent polarity sign with the IBM Heron canonical:

```sql
SELECT
  pol.subject,
  pol.polarity_sign        AS ibm_polarity,
  pol.polarity_strength    AS ibm_strength,
  br.adapter_applied,
  br.tvd                   AS sv1_aer_tvd,
  br.actual_cost_usd       AS sv1_cost
FROM aws004.polarity_results pol
LEFT JOIN aws004.braket_results br
  ON pol.subject = br.subject
WHERE pol.prompt_id = 'IBM-HERON'
  AND pol.channels = 8
  AND pol.window_s = 1.95
  AND br.prompt_id = 'AWS-003'
ORDER BY pol.subject, br.adapter_applied DESC;
```

Joins like this don't yet show much because we have one Braket subject
(`chb01`) and seven IBM subjects, but the contract is in place for when
more devices are ingested.
