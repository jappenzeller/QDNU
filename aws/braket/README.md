# Braket Multi-Backend Polarity Experiment

Cross-platform A-Gate execution on Amazon Braket. Tests whether the polarity invariant
established on IBM Heron persists across IonQ (trapped-ion), Rigetti (superconducting,
different topology), IQM (yet another superconducting platform), and the Braket
managed simulators (SV1, DM1).

## Why this exists

Paper 2 established that polarity is "basis-dependent": IBM hardware, statevector
simulation, and classical Riemannian methods agree on polarity assignment 57-71% of
the time. The hypothesis is that each measurement basis imposes a different reference
orientation on the SPD manifold. Adding three more hardware platforms tests this
directly. Two outcomes are interesting:

1. **Patient-stable polarity** — chb03 and chb21 stay consistent across all platforms.
   That suggests polarity is a deeper geometric property than the IBM-only data showed.
2. **Platform-dependent polarity** — each backend produces its own polarity map. That
   tightens the basis-dependence claim and gives a falsifiable prediction for which
   patients are "robust" vs. "platform-sensitive".

## Architecture

```
existing PLV pipeline (results/)
        |
        v
prepare_input.py  --extracts (a,b,c) per (patient, channel, window)-->  job/input_data/
                                                                            |
                                                                            v
                                                                  Braket Hybrid Job
                                                                    (algorithm_script.py)
                                                                            |
                                                          per-backend execution
                                              ___________________|___________________
                                             |          |          |          |
                                             v          v          v          v
                                            SV1        DM1       IonQ      Rigetti
                                                                  Aria      Ankaa
                                             |__________|__________|__________|
                                                          |
                                                          v
                                                  S3 (qdnu-braket-results/)
                                                  results/{run_id}/
                                                    metadata.json
                                                    backend={name}/
                                                      patient={id}/
                                                        results.parquet
                                                          |
                                                          v
                                                    Glue crawler
                                                          |
                                                          v
                                                       Athena
                                                          |
                                                          v
                                                     QuickSight
```

## Cost ceiling

Default sweep is intentionally small to keep costs predictable:

- **7 patients** (chb01, chb03, chb05, chb07, chb11, chb14, chb21)
- **8 channels** (CH8 montage, 17 qubits)
- **1024 shots per circuit**
- **3 windows per patient** (1 interictal + 1 preictal + 1 ictal)

Per-backend cost (1024 shots * 3 windows * 7 patients = 21,504 shots):

| Backend       | Cost model                  | Estimated cost |
|---------------|-----------------------------|----------------|
| SV1           | $0.075 / minute             | ~$1            |
| DM1           | $0.075 / minute             | ~$1            |
| IonQ Aria     | $0.30/task + $0.03/shot     | ~$650          |
| IonQ Forte    | $0.30/task + $0.06/shot     | ~$1300         |
| Rigetti Ankaa | $0.30/task + $0.00090/shot  | ~$25           |
| IQM Garnet    | $0.30/task + $0.00145/shot  | ~$35           |

**Recommended first run: SV1 + DM1 + Rigetti only.** Total ~$30. Adds two new
backends to the existing IBM data without burning IonQ budget on a sweep that
might find nothing.

## Files

```
cdk/                       # CDK app: S3 + Glue + Athena workgroup + IAM role
  app.py
  cdk.json
  requirements.txt
  stacks/
    braket_storage_stack.py
    braket_iam_stack.py
job/
  algorithm_script.py      # Hybrid Job entry point; per-patient circuits + measurement
  agate_circuit.py         # Native Braket A-Gate circuit (no Qiskit dependency)
  requirements.txt         # Container deps for the Hybrid Job
  input_data/              # (gitignored) Pre-computed (a,b,c) parameters per patient
scripts/
  prepare_input.py         # Extract (a,b,c) from existing results/ artifacts
  submit_job.py            # Submit a Hybrid Job run with chosen backend
infra/
  athena_ddl.sql           # CREATE TABLE statements for the result schema
  result_schema.json       # JSON schema describing each row written to Parquet
```

## Deploy

### One-time infrastructure (CDK)

```bash
cd aws/braket/cdk
python -m venv .venv
. .venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on Unix
pip install -r requirements.txt

# First time per account/region:
cdk bootstrap

# Deploy both stacks:
cdk deploy --all
```

The deploy creates two stacks (`qdnu-dev-braket-storage`, `qdnu-dev-braket-iam`)
and prints the role ARN, bucket name, Glue database name, and Athena workgroup
name. Capture them; you'll feed the role ARN to `submit_job.py` and the rest
to Athena/QuickSight.

### Athena schema (one-time, after CDK deploy)

The CDK stack creates the Glue database but not the tables. Apply the DDL:

```bash
aws athena start-query-execution \
  --work-group qdnu-braket-dev \
  --query-string "$(cat aws/braket/infra/athena_ddl.sql)"
```

Or paste the file into the Athena console and run it once. Re-running is safe;
all statements use `CREATE ... IF NOT EXISTS` or `CREATE OR REPLACE VIEW`.

### Per-run workflow

```bash
# 1. Build the input bundle from existing PLV results
python aws/braket/scripts/prepare_input.py

# 2. Export the Hybrid Jobs role ARN (one time per shell)
export BRAKET_JOBS_ROLE_ARN="$(aws cloudformation list-exports \
    --query 'Exports[?Name==`qdnu-dev-braket-jobs-role-arn`].Value' \
    --output text)"

# 3. Submit (start with the cheap simulator)
python aws/braket/scripts/submit_job.py --backend sv1 --shots 1024

# 4. Watch progress in the Braket console; results land in
#    s3://qdnu-dev-braket-results-{account}/runs/{run_id}/
```

Once one good run is in, the Athena tables (with partition projection) pick it
up automatically and the QuickSight datasets refresh on next query.

## Out of scope for the CDK stack

- QuickSight subscription, datasets, dashboards. Stand these up via the
  console once the Athena views return rows.
- Cross-region Hybrid Jobs. Each QPU region (us-east-1 for IonQ, us-west-1
  for Rigetti, eu-north-1 for IQM) needs the storage+IAM stacks deployed
  there if you want the results bucket co-located. For an initial run, a
  single us-east-1 deployment with cross-region S3 writes is fine.
