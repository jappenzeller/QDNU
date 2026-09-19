"""
PROMPT AWS-002 - Isolate the CRy/CRz translation bug surfaced in AWS-001.

Three subroutines:
  A) Minimal 2-qubit CRy/CRz reproducer comparing Aer vs BraketLocalBackend
     against theoretical predictions. Tests H1 (operand swap) and H3 (phase
     convention).
  B) Emitted-circuit inspection: feed the AWS-001 8-channel circuit through
     qiskit_braket_provider.providers.adapter.to_braket and dump the result.
     Compare CRy unitary against the Braket-emitted equivalent. Tests H2
     (decomposition bug).
  C) Manual-decomposition workaround: rewrite each CRy/CRz as RY(t/2)-CX-
     RY(-t/2)-CX (or RZ analog). Output a transformed circuit ready for the
     Braket bridge.

No cloud spend. No threshold widening.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from math import cos, pi, sin
from pathlib import Path
from typing import Any

# Disable numba JIT so braket-default-simulator can handle 17-qubit reshapes.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np

# Preload braket SDK before any sys.path manipulation that puts the project's
# 'scripts/braket/' folder on the path (it would shadow the SDK).
import braket  # noqa: F401
import qiskit_braket_provider  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "QA1"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Operator  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402
from qiskit_braket_provider import BraketLocalBackend  # noqa: E402
from qiskit_braket_provider.providers.adapter import to_braket  # noqa: E402

from sagemaker.train_chbmit import (  # noqa: E402
    extract_segments_for_subject,
)
from QA1.multichannel_circuit import (  # noqa: E402
    create_multichannel_circuit,
    add_measurements,
)
from run_quantum_loso_v3 import (  # noqa: E402
    DATA_DIR,
    QUANTUM_CHANNELS,
    extract_plv_params,
    load_eeg_segment,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


SHOTS = 8192
RESULTS_DIR = REPO_ROOT / "results" / "braket"

# ============================================================================
# Helpers
# ============================================================================


def _run_aer(circuit: QuantumCircuit, shots: int, seed: int = 42) -> dict[str, int]:
    sim = AerSimulator(seed_simulator=seed)
    return sim.run(circuit, shots=shots, seed_simulator=seed).result().get_counts()


def _run_braket_local(circuit: QuantumCircuit, shots: int) -> dict[str, int]:
    backend = BraketLocalBackend()
    return backend.run(circuit, shots=shots).result().get_counts()


def _normalize_counts(counts: dict[str, int]) -> dict[str, float]:
    """Convert raw counts to probability fractions and strip whitespace."""
    total = sum(counts.values()) or 1
    return {k.replace(" ", ""): v / total for k, v in counts.items()}


# ============================================================================
# Subroutine A - minimal CRy reproducer
# ============================================================================


def _make_minimal_circuit(
    gate: str, control: int, target: int, theta: float = pi / 4
) -> QuantumCircuit:
    """H on q0, then controlled rotation. Measure all."""
    qc = QuantumCircuit(2, 2)
    qc.h(0)
    if gate == "cry":
        qc.cry(theta, control, target)
    elif gate == "crz":
        qc.crz(theta, control, target)
    else:
        raise ValueError(gate)
    qc.measure([0, 1], [0, 1])
    return qc


def _theoretical_cry(control: int, target: int, theta: float = pi / 4) -> dict[str, float]:
    """Theoretical probabilities for: H on q0, CRy(theta, control, target).

    Qiskit bitstring convention: '<q1><q0>' (q1 left, q0 right).

    After H on q0: |psi> = (|00> + |01>)/sqrt(2) (Qiskit ordering).
    Decompose into the {|control=0>, |control=1>} branches:
      branch with control=0: target unchanged
      branch with control=1: target gets Ry(theta) applied
    Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>
    """
    half = theta / 2.0
    c = cos(half)
    s = sin(half)
    if (control, target) == (0, 1):
        # control=q0, target=q1
        # control=0 branch (q0=0, prob 1/2): q1=0 -> bitstring '00'
        # control=1 branch (q0=1, prob 1/2): q1 = Ry|0> = cos|0> + sin|1>
        #   bitstring '01' has prob 1/2 * c^2; bitstring '11' has 1/2 * s^2
        return {"00": 0.5, "01": 0.5 * c * c, "11": 0.5 * s * s}
    if (control, target) == (1, 0):
        # control=q1, target=q0
        # H on q0 created (|q1=0,q0=0> + |q1=0,q0=1>)/sqrt(2)
        # control=q1=0 in BOTH branches, so CRy is a no-op.
        return {"00": 0.5, "01": 0.5}
    raise ValueError((control, target))


def _theoretical_crz(control: int, target: int, theta: float = pi / 4) -> dict[str, float]:
    """CRz only modifies phase, never population. Same probabilities as no-op."""
    if (control, target) == (0, 1):
        return {"00": 0.5, "01": 0.5}
    if (control, target) == (1, 0):
        return {"00": 0.5, "01": 0.5}
    raise ValueError((control, target))


def _max_prob_diff(
    expected: dict[str, float], counts: dict[str, int]
) -> tuple[float, dict[str, float]]:
    """Return (max abs diff, normalized observed)."""
    obs = _normalize_counts(counts)
    keys = set(expected) | set(obs)
    max_d = 0.0
    for k in keys:
        d = abs(expected.get(k, 0.0) - obs.get(k, 0.0))
        if d > max_d:
            max_d = d
    return max_d, obs


def subroutine_a() -> dict[str, Any]:
    """Run minimal reproducers for CRy and CRz with both operand orders."""
    log.info("Subroutine A - minimal controlled-rotation reproducer")
    configs: list[dict[str, Any]] = []
    for gate in ("cry", "crz"):
        for control, target in ((0, 1), (1, 0)):
            qc = _make_minimal_circuit(gate, control, target)
            aer = _run_aer(qc, SHOTS)
            brk = _run_braket_local(qc, SHOTS)
            theo = (_theoretical_cry if gate == "cry" else _theoretical_crz)(
                control, target
            )
            aer_diff, aer_obs = _max_prob_diff(theo, aer)
            brk_diff, brk_obs = _max_prob_diff(theo, brk)
            cfg = {
                "gate": gate,
                "control": control,
                "target": target,
                "theta": "pi/4",
                "shots": SHOTS,
                "theoretical_probs": theo,
                "aer_probs": aer_obs,
                "braket_probs": brk_obs,
                "aer_max_diff_vs_theory": aer_diff,
                "braket_max_diff_vs_theory": brk_diff,
                "aer_passes": aer_diff < 0.02,
                "braket_passes": brk_diff < 0.02,
            }
            configs.append(cfg)
            log.info(
                "  %s control=%d target=%d  aer_d=%.4f braket_d=%.4f",
                gate,
                control,
                target,
                aer_diff,
                brk_diff,
            )
    summary = {
        "prompt_id": "AWS-002",
        "subroutine": "A",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "shots": SHOTS,
        "threshold": 0.02,
        "configurations": configs,
    }
    return summary


# ============================================================================
# Subroutine B - emitted-circuit inspection
# ============================================================================


def _load_aws001_circuit() -> tuple[QuantumCircuit, str]:
    """Rebuild the same 17-qubit circuit AWS-001 used."""
    subject_dir = DATA_DIR / "chb01"
    segments = extract_segments_for_subject(subject_dir)
    ictal = [s for s in segments if s.label == "ictal"]
    if not ictal:
        raise RuntimeError("no ictal segments for chb01")
    for seg in ictal:
        eeg = load_eeg_segment(seg, subject_dir, QUANTUM_CHANNELS)
        if eeg is not None:
            params = extract_plv_params(eeg, fs=256.0, band=(4, 13))
            qc = create_multichannel_circuit(params)
            return qc, f"{seg.source_file}#start={seg.start_sec:.1f}s"
    raise RuntimeError("failed to load any chb01 ictal segment")


def _cry_unitary_qiskit(theta: float = pi / 4) -> np.ndarray:
    qc = QuantumCircuit(2)
    qc.cry(theta, 0, 1)
    return np.asarray(Operator(qc).data)


def _cry_decomposed_unitary(theta: float = pi / 4) -> np.ndarray:
    qc = QuantumCircuit(2)
    half = theta / 2.0
    qc.ry(half, 1)
    qc.cx(0, 1)
    qc.ry(-half, 1)
    qc.cx(0, 1)
    return np.asarray(Operator(qc).data)


def subroutine_b() -> dict[str, Any]:
    log.info("Subroutine B - inspect provider-emitted circuit + unitaries")
    qc_full, seg_id = _load_aws001_circuit()
    measured = qc_full.copy()
    measured = add_measurements(measured, num_channels=8, measure_all=True)
    braket_circ = to_braket(measured)
    text = str(braket_circ)
    (RESULTS_DIR / "provider_emitted_circuit.txt").write_text(
        f"AWS-002 Subroutine B emitted circuit\n"
        f"Source segment: {seg_id}\n"
        f"\n"
        f"=== Full Braket circuit (provider output) ===\n{text}\n",
        encoding="utf-8",
    )

    # Unitary check: compare Qiskit's CRy(pi/4) vs the standard 4-gate
    # decomposition (RY(t/2)-CX-RY(-t/2)-CX). If these match, the bug is
    # in the provider's translation, not in the math.
    u_qiskit = _cry_unitary_qiskit()
    u_decomp = _cry_decomposed_unitary()
    u_diff_max = float(np.max(np.abs(u_qiskit - u_decomp)))

    # Count CRy/CRz occurrences in the original Qiskit circuit so the
    # output text shows what we were comparing against.
    cry_count = sum(1 for instr in qc_full.data if instr.operation.name == "cry")
    crz_count = sum(1 for instr in qc_full.data if instr.operation.name == "crz")

    summary = {
        "prompt_id": "AWS-002",
        "subroutine": "B",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "segment": seg_id,
        "qiskit_cry_count": cry_count,
        "qiskit_crz_count": crz_count,
        "cry_unitary_qiskit_vs_decomposed_max_abs": u_diff_max,
        "emitted_circuit_path": str(
            (RESULTS_DIR / "provider_emitted_circuit.txt").relative_to(REPO_ROOT)
        ),
        "emitted_gate_types": _summarize_braket_gates(braket_circ),
    }
    return summary


def _summarize_braket_gates(braket_circ) -> dict[str, int]:
    counts: dict[str, int] = {}
    for instr in braket_circ.instructions:
        # Each instruction wraps a Gate
        gate_name = type(instr.operator).__name__
        counts[gate_name] = counts.get(gate_name, 0) + 1
    return counts


# ============================================================================
# Subroutine C - manual decomposition workaround
# ============================================================================


def replace_controlled_rotations_with_native(qc: QuantumCircuit) -> QuantumCircuit:
    """Walk the circuit; rewrite cry/crz using the standard 4-gate identity.

    cry(t, c, t) -> ry(t/2, t), cx(c, t), ry(-t/2, t), cx(c, t)
    crz(t, c, t) -> rz(t/2, t), cx(c, t), rz(-t/2, t), cx(c, t)

    All other gates pass through unchanged. The transformed circuit is
    semantically identical to the input under the standard CRy/CRz
    definitions in Qiskit.
    """
    new = QuantumCircuit(*qc.qregs, *qc.cregs)
    for instr in qc.data:
        op = instr.operation
        qargs = instr.qubits
        cargs = instr.clbits
        if op.name == "cry":
            theta = float(op.params[0])
            half = theta / 2.0
            ctrl, tgt = qargs[0], qargs[1]
            new.ry(half, tgt)
            new.cx(ctrl, tgt)
            new.ry(-half, tgt)
            new.cx(ctrl, tgt)
        elif op.name == "crz":
            theta = float(op.params[0])
            half = theta / 2.0
            ctrl, tgt = qargs[0], qargs[1]
            new.rz(half, tgt)
            new.cx(ctrl, tgt)
            new.rz(-half, tgt)
            new.cx(ctrl, tgt)
        else:
            new.append(op, qargs, cargs)
    return new


def subroutine_c_quick_check() -> dict[str, Any]:
    """Quick sanity check that the workaround actually removes the divergence
    on the minimal CRy reproducer. Doesn't replace the AWS-001 v2 run.
    """
    log.info("Subroutine C - manual-decomposition workaround sanity check")
    results: list[dict[str, Any]] = []
    for gate, control, target in (
        ("cry", 0, 1),
        ("cry", 1, 0),
        ("crz", 0, 1),
        ("crz", 1, 0),
    ):
        qc = _make_minimal_circuit(gate, control, target)
        qc_decomp = replace_controlled_rotations_with_native(qc)
        aer = _run_aer(qc_decomp, SHOTS)
        brk = _run_braket_local(qc_decomp, SHOTS)
        theo = (_theoretical_cry if gate == "cry" else _theoretical_crz)(
            control, target
        )
        aer_d, _ = _max_prob_diff(theo, aer)
        brk_d, _ = _max_prob_diff(theo, brk)
        results.append(
            {
                "gate": gate,
                "control": control,
                "target": target,
                "aer_max_diff_vs_theory": aer_d,
                "braket_max_diff_vs_theory": brk_d,
                "aer_passes": aer_d < 0.02,
                "braket_passes": brk_d < 0.02,
            }
        )
        log.info(
            "  decomposed %s control=%d target=%d  aer_d=%.4f braket_d=%.4f",
            gate,
            control,
            target,
            aer_d,
            brk_d,
        )
    return {
        "prompt_id": "AWS-002",
        "subroutine": "C",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "minimal_results_after_workaround": results,
    }


# ============================================================================
# Driver
# ============================================================================


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    a = subroutine_a()
    b = subroutine_b()
    c = subroutine_c_quick_check()

    bundle = {
        "subroutine_a": a,
        "subroutine_b": b,
        "subroutine_c": c,
    }
    out = RESULTS_DIR / "cry_minimal_reproducer.json"
    out.write_text(json.dumps(bundle, indent=2))
    log.info("wrote %s", out)

    # Print a one-screen verdict
    print()
    print("=" * 72)
    print("AWS-002 Diagnostic Summary")
    print("=" * 72)
    print()
    print("[A] Minimal CRy/CRz reproducer (Aer & BraketLocal vs theory):")
    for cfg in a["configurations"]:
        marker_aer = "PASS" if cfg["aer_passes"] else "FAIL"
        marker_brk = "PASS" if cfg["braket_passes"] else "FAIL"
        print(
            f"    {cfg['gate']:>3s}({cfg['control']}->{cfg['target']})  "
            f"aer_d={cfg['aer_max_diff_vs_theory']:.4f} {marker_aer}    "
            f"braket_d={cfg['braket_max_diff_vs_theory']:.4f} {marker_brk}"
        )
    print()
    print("[B] Provider-emitted circuit:")
    print(f"    qiskit cry count: {b['qiskit_cry_count']}")
    print(f"    qiskit crz count: {b['qiskit_crz_count']}")
    print(f"    cry unitary (qiskit vs std decomp) max abs diff: {b['cry_unitary_qiskit_vs_decomposed_max_abs']:.2e}")
    print(f"    emitted circuit -> {b['emitted_circuit_path']}")
    print(f"    braket gate counts: {b['emitted_gate_types']}")
    print()
    print("[C] After manual-decomposition workaround on minimal reproducer:")
    for r in c["minimal_results_after_workaround"]:
        marker_brk = "PASS" if r["braket_passes"] else "FAIL"
        print(
            f"    {r['gate']:>3s}({r['control']}->{r['target']})  "
            f"braket_d={r['braket_max_diff_vs_theory']:.4f} {marker_brk}"
        )
    print()
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
