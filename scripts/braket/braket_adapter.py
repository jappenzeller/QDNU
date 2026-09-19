"""
PROMPT AWS-002 - Adapter for Qiskit -> Braket bridge issues.

The qiskit-braket-provider 0.14.3 translates Qiskit's CRz to Braket's native
ZZ gate. The 2-gate `Rz(t/2)_target + ZZ(-t/2)` decomposition is unitarily
equivalent to CRz, but in the 17-qubit multichannel A-Gate context (PROMPT
AWS-001), the ZZ-based emission produces measurement statistics that diverge
from Qiskit Aer's native simulation by TVD ~0.55 with |Z_I|max diff ~0.51.

Workaround: rewrite each `cry`/`crz` instruction with the standard 4-gate
decomposition before submission, so the provider sees only `ry`/`rz`/`cx`
primitives, no controlled rotations. This is a depth-trade-off: 8 ZZ gates
become 16 Rz + 16 CNOT, but it eliminates the divergence.

This adapter is opt-in. The Aer-and-IBM-Heron path (run_quantum_loso_v3.py
and friends) is unaffected. Apply it only when targeting the Braket bridge.
"""

from __future__ import annotations

from qiskit import QuantumCircuit


def replace_controlled_rotations_with_native(qc: QuantumCircuit) -> QuantumCircuit:
    """Rewrite cry/crz using the standard 4-gate identity.

    cry(t, c, t) -> ry(t/2, t), cx(c, t), ry(-t/2, t), cx(c, t)
    crz(t, c, t) -> rz(t/2, t), cx(c, t), rz(-t/2, t), cx(c, t)

    All other gates pass through unchanged. The transformed circuit is
    semantically identical to the input under the Qiskit definitions of
    CRy/CRz.
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


def prepare_circuit_for_braket(qc: QuantumCircuit) -> QuantumCircuit:
    """Public entry point. Apply all known Qiskit -> Braket bridge workarounds.

    Currently delegates to replace_controlled_rotations_with_native. Add new
    transformations here as the bridge evolves.
    """
    return replace_controlled_rotations_with_native(qc)
