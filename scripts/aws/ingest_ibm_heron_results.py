"""
PROMPT AWS-006 - Ingest IBM Heron polarity records into aws004.polarity_results.

Reads results/validation/polarity_consistency_audit.json (the canonical
4-config x 7-subject matrix) and enriches CH8_20s rows with calibrated AUC
from results/window_analysis/quantum_20s_hardware.json. Writes one
NDJSON-compact record per (subject, config) into the partitioned S3 layout:

    s3://qdnu-braket-{account}-us-east-1/polarity/
        prompt_id=IBM-HERON/device=ibm_torino/subject={subj}/{ts}.json

Field mappings:
  source 'standard' -> 'positive'   (polarity_sign)
  source 'inverted' -> 'negative'
  source 'chance'   -> 'null'

polarity_strength derives from polarity_consistency_audit.consistency:
  consistent=true  -> 'strong'
  consistent=false -> 'weak'

oracle_cal_auc derives as max(raw_auc, 1 - raw_auc) when not present in source.

Usage:
    python scripts/aws/ingest_ibm_heron_results.py
    python scripts/aws/ingest_ibm_heron_results.py --dry-run
    python scripts/aws/ingest_ibm_heron_results.py --limit 5
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
COST_LOG_PATH = REPO_ROOT / "docs" / "aws" / "cost_log.md"
DEFAULT_REGION = "us-east-1"
DEFAULT_AUDIT_PATH = REPO_ROOT / "results" / "validation" / "polarity_consistency_audit.json"
DEFAULT_20S_PATH = REPO_ROOT / "results" / "window_analysis" / "quantum_20s_hardware.json"

# Map config keys in the audit to (channels, window_s, encoding)
CONFIG_TABLE: dict[str, dict[str, Any]] = {
    "CH8_1.95s": {"channels": 8, "window_s": 1.95, "encoding": "V3_PLV_theta_alpha"},
    "CH8_20s":   {"channels": 8, "window_s": 20.0, "encoding": "V3_PLV_theta_alpha"},
    "CH12":      {"channels": 12, "window_s": 1.95, "encoding": "V3_PLV_theta_alpha"},
    "CH16":      {"channels": 16, "window_s": 1.95, "encoding": "V3_PLV_theta_alpha"},
}

POLARITY_MAP = {
    "standard": "positive",
    "inverted": "negative",
    "chance":   "null",
}

# Project-memory expectations. If derived strength contradicts these for the
# subjects with strong expectations (chb03, chb21) or any of the four weak
# expectations (chb01, chb05, chb07, chb14), warn but do not override.
EXPECTED_STRENGTH = {
    "chb03": "strong",
    "chb21": "strong",
    "chb01": "weak",
    "chb05": "weak",
    "chb07": "weak",
    "chb14": "weak",
    # chb11 not pre-classified per project memory; derive from data.
}

CANONICAL_SUBJECTS = {
    "chb01", "chb03", "chb05", "chb07", "chb11", "chb14", "chb21"
}


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _bucket_name(account_id: str) -> str:
    return f"qdnu-braket-{account_id}-{DEFAULT_REGION}"


def _safe_for_s3(s: str) -> str:
    return s.replace(":", "-").replace("+", "-")


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


def _load_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"polarity_consistency_audit not found at {path}; cannot ingest"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _load_20s(path: Path) -> dict[str, Any]:
    if not path.exists():
        log.warning(
            "quantum_20s_hardware.json not at %s; CH8_20s records will use "
            "derived oracle_cal_auc instead of the explicit calibrated_auc.",
            path,
        )
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _derive_oracle_cal_auc(raw_auc: float | None) -> float | None:
    if raw_auc is None:
        return None
    return float(max(raw_auc, 1.0 - raw_auc))


def _key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}:
            return False
        raise


def _build_record(
    *,
    subject: str,
    config_key: str,
    raw_auc: float | None,
    polarity_label: str,
    strength: str,
    timestamp: str,
    explicit_cal_auc: float | None = None,
    notes: str = "",
) -> dict[str, Any]:
    cfg = CONFIG_TABLE[config_key]
    polarity_sign = POLARITY_MAP.get(polarity_label, "null")
    cal_auc = (
        explicit_cal_auc if explicit_cal_auc is not None
        else _derive_oracle_cal_auc(raw_auc)
    )
    return {
        "timestamp": timestamp,
        "channels": cfg["channels"],
        "window_s": cfg["window_s"],
        "encoding": cfg["encoding"],
        "shots": 1024,
        "raw_auc": raw_auc,
        "oracle_cal_auc": cal_auc,
        "polarity_sign": polarity_sign,
        "polarity_strength": strength,
        "mean_z_e": None,
        "mean_z_i": None,
        "covariance_shift": None,
        "adapter_applied": False,
        "notes": notes,
        "device_arn": "",
        "task_arn": "",
        "region": DEFAULT_REGION,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-audit", type=Path, default=DEFAULT_AUDIT_PATH)
    ap.add_argument("--source-20s", type=Path, default=DEFAULT_20S_PATH)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of records to process (debug)")
    ap.add_argument("--region", default=DEFAULT_REGION)
    args = ap.parse_args()

    audit = _load_audit(args.source_audit)
    twenty_s = _load_20s(args.source_20s)
    audit_timestamp = audit.get("timestamp") or datetime.now(timezone.utc).isoformat()

    auc_matrix: dict[str, dict[str, float]] = audit.get("auc_matrix", {})
    polarity_matrix: dict[str, dict[str, str]] = audit.get("polarity_matrix", {})
    consistency: dict[str, dict[str, Any]] = audit.get("consistency", {})

    if not auc_matrix or not polarity_matrix:
        raise SystemExit("audit JSON missing auc_matrix or polarity_matrix")

    sts = boto3.client("sts", region_name=args.region)
    account_id = sts.get_caller_identity()["Account"]
    bucket = _bucket_name(account_id)
    s3 = boto3.client("s3", region_name=args.region) if not args.dry_run else None

    records_built = 0
    records_uploaded = 0
    records_skipped = 0
    discrepancies: list[str] = []

    for subject in sorted(auc_matrix.keys()):
        if subject not in CANONICAL_SUBJECTS:
            log.warning(
                "subject %s is not in the canonical 7-patient set; skipping",
                subject,
            )
            continue

        consist = consistency.get(subject, {})
        is_consistent = bool(consist.get("consistent", False))
        derived_strength = "strong" if is_consistent else "weak"

        expected = EXPECTED_STRENGTH.get(subject)
        if expected is not None and expected != derived_strength:
            msg = (
                f"DISCREPANCY: {subject} expected '{expected}' but derived "
                f"'{derived_strength}' from consistency.consistent="
                f"{is_consistent}. Writing derived value."
            )
            log.warning(msg)
            discrepancies.append(msg)

        for config_key, cfg in CONFIG_TABLE.items():
            raw_auc = auc_matrix.get(subject, {}).get(config_key)
            polarity_label = polarity_matrix.get(subject, {}).get(config_key)
            if raw_auc is None or polarity_label is None:
                log.info("skip %s/%s (missing data)", subject, config_key)
                continue

            explicit_cal = None
            notes = ""
            if config_key == "CH8_20s" and twenty_s:
                ps = twenty_s.get("per_subject", {}).get(subject, {})
                if "calibrated_auc" in ps:
                    explicit_cal = float(ps["calibrated_auc"])
                    notes = "calibrated_auc from quantum_20s_hardware.json"

            record = _build_record(
                subject=subject,
                config_key=config_key,
                raw_auc=float(raw_auc),
                polarity_label=polarity_label,
                strength=derived_strength,
                timestamp=audit_timestamp,
                explicit_cal_auc=explicit_cal,
                notes=notes,
            )

            key = (
                "polarity/"
                f"prompt_id=IBM-HERON/"
                f"device=ibm_torino/"
                f"subject={subject}/"
                f"{_safe_for_s3(audit_timestamp)}__{config_key}.json"
            )
            s3_uri = f"s3://{bucket}/{key}"
            records_built += 1

            if args.limit is not None and records_built > args.limit:
                log.info("hit --limit %d; stopping", args.limit)
                break

            if args.dry_run:
                log.info(
                    "DRY RUN  %s/%s  raw=%.4f cal=%.4f sign=%s strength=%s -> %s",
                    subject,
                    config_key,
                    record["raw_auc"],
                    record["oracle_cal_auc"],
                    record["polarity_sign"],
                    record["polarity_strength"],
                    s3_uri,
                )
                continue

            if _key_exists(s3, bucket, key):
                log.warning("EXISTS   %s -> skip", s3_uri)
                records_skipped += 1
                continue

            body = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
            s3.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="application/x-ndjson",
            )
            log.info(
                "UPLOAD   %s/%s  sign=%s strength=%s -> %s",
                subject,
                config_key,
                record["polarity_sign"],
                record["polarity_strength"],
                s3_uri,
            )
            records_uploaded += 1

        if args.limit is not None and records_built > args.limit:
            break

    if records_uploaded > 0 and not args.dry_run:
        _append_cost_row(
            action="ibm_heron_ingest",
            resource=f"s3://{bucket}/polarity/",
            quantity=records_uploaded,
            unit="put_objects",
            cost_usd=records_uploaded * 0.000005,
        )

    print()
    print("=" * 72)
    print("AWS-006 IBM Heron ingest summary")
    print("=" * 72)
    print(f"  bucket:                 s3://{bucket}/")
    print(f"  records built:          {records_built}")
    print(f"  records uploaded:       {records_uploaded}")
    print(f"  records skipped:        {records_skipped}")
    if discrepancies:
        print(f"  discrepancies flagged:  {len(discrepancies)}")
        for d in discrepancies:
            print(f"    {d}")
    if args.dry_run:
        print("  (dry run; no S3 calls made)")
    print("=" * 72)
    return 0 if not discrepancies else 0


if __name__ == "__main__":
    sys.exit(main())
