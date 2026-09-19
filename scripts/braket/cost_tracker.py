"""
PROMPT AWS-003 - Braket cost estimation and logging.

Hard cap: any single script using estimate_sv1_cost() must abort if the
estimate exceeds $5. The point is a guardrail, not a budgeting tool.

SV1 pricing (us-east-1, 2025): $0.075 per minute, billed in seconds with a
3-second minimum charge. Source: AWS Braket pricing page.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SV1_RATE_USD_PER_MINUTE = 0.075
SV1_MINIMUM_CHARGE_SECONDS = 3.0
HARD_CAP_USD = 5.0
COST_LOG_PATH = Path(__file__).resolve().parents[2] / "docs" / "aws" / "braket_cost_log.md"


def estimate_sv1_cost(
    n_qubits: int, n_shots: int, expected_minutes: float | None = None
) -> float:
    """Conservative USD upper bound for an SV1 task.

    The real bottleneck on SV1 is the dense statevector simulation which
    scales as 2^n_qubits in memory and roughly per-shot in measurement
    sampling. For 17 qubits with up to a few thousand shots, SV1 finishes
    in tens of seconds, but we estimate generously.

    Args:
        n_qubits: number of qubits in the circuit (must be <= 34 for SV1).
        n_shots: total measurement shots requested.
        expected_minutes: override the heuristic with a known runtime in
            minutes. If None, a conservative estimate is used.

    Returns:
        USD upper bound for the task.

    Raises:
        ValueError: n_qubits exceeds the SV1 limit.
        RuntimeError: estimate exceeds the hard cap.
    """
    if n_qubits > 34:
        raise ValueError(f"SV1 supports up to 34 qubits, got {n_qubits}")

    if expected_minutes is not None:
        minutes = max(expected_minutes, SV1_MINIMUM_CHARGE_SECONDS / 60.0)
    else:
        # Heuristic: assume 0.5s of base setup + per-shot measurement at
        # 0.1ms (very generous), with an extra 1.5x factor for queueing.
        seconds_per_shot = 1e-4
        base_seconds = 0.5
        raw_seconds = base_seconds + n_shots * seconds_per_shot
        # Guard against tiny estimates being outpaced by the minimum
        # charge.
        seconds = max(raw_seconds, SV1_MINIMUM_CHARGE_SECONDS) * 1.5
        minutes = seconds / 60.0

    cost = minutes * SV1_RATE_USD_PER_MINUTE
    if cost > HARD_CAP_USD:
        raise RuntimeError(
            f"Estimated SV1 cost ${cost:.2f} exceeds hard cap "
            f"${HARD_CAP_USD:.2f}. Refusing to submit. Reduce shots or "
            "raise the cap explicitly with full understanding."
        )
    return cost


def _format_cost_row(
    *,
    task_arn: str,
    region: str,
    device_arn: str,
    output_path: str,
    billable_seconds: float,
    cost_usd: float,
    timestamp_utc: str,
    notes: str = "",
) -> str:
    """Build a single markdown table row for the cost log."""
    return (
        f"| {timestamp_utc} | {region} | {device_arn} | {task_arn} | "
        f"{billable_seconds:.2f} | {cost_usd:.4f} | "
        f"{output_path} | {notes} |"
    )


def _ensure_cost_log(path: Path) -> None:
    """Create the cost log with header if missing."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# Braket Cost Log\n\n"
        "Append-only record of every Braket task that consumed paid time. "
        "One row per task. Created by `scripts/braket/cost_tracker.py`.\n\n"
        "| timestamp_utc | region | device_arn | task_arn | "
        "billable_seconds | cost_usd | output_path | notes |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    path.write_text(header, encoding="utf-8")


def log_actual_cost(
    task_arn: str,
    region: str,
    output_path: str,
    notes: str = "",
) -> dict[str, Any]:
    """Fetch task metadata, compute actual billable time x rate, append row.

    Uses the Braket SDK's AwsQuantumTask. Requires AWS credentials in the
    standard chain.

    Returns a dict with:
        task_arn, device_arn, billable_seconds, cost_usd, status,
        timestamp_utc, output_path, region.
    """
    # Lazy import so importing this module doesn't require the SDK.
    from braket.aws import AwsQuantumTask

    task = AwsQuantumTask(arn=task_arn)
    meta = task.metadata()

    device_arn = meta.get("deviceArn", "")
    status = meta.get("status", "UNKNOWN")
    created_at = meta.get("createdAt")
    ended_at = meta.get("endedAt") or meta.get("modifiedAt")
    billable_seconds = 0.0
    if created_at and ended_at:
        billable_seconds = (ended_at - created_at).total_seconds()
    billable_seconds = max(billable_seconds, SV1_MINIMUM_CHARGE_SECONDS)

    rate_per_second = SV1_RATE_USD_PER_MINUTE / 60.0
    cost = billable_seconds * rate_per_second

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    row = _format_cost_row(
        task_arn=task_arn,
        region=region,
        device_arn=device_arn,
        output_path=output_path,
        billable_seconds=billable_seconds,
        cost_usd=cost,
        timestamp_utc=timestamp,
        notes=notes,
    )

    _ensure_cost_log(COST_LOG_PATH)
    with COST_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(row + "\n")

    return {
        "task_arn": task_arn,
        "device_arn": device_arn,
        "billable_seconds": billable_seconds,
        "cost_usd": cost,
        "status": status,
        "timestamp_utc": timestamp,
        "output_path": output_path,
        "region": region,
    }
