"""
Native Braket A-Gate circuit, no Qiskit dependency.

Mirrors the canonical 14-gate single-channel circuit from qdnu/quantum_agate.py
and the multi-channel ring topology from qdnu/multichannel_circuit.py, but
expressed in the Braket SDK so it can run on any Braket device (SV1, DM1,
IonQ, Rigetti, IQM) without translation.

The encoding parameters (a, b, c) are computed upstream by the existing PLV
pipeline; this module only consumes them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from braket.circuits import Circuit


PI_4 = math.pi / 4.0


@dataclass(frozen=True)
class ChannelParams:
    """Per-channel A-Gate encoding parameters."""

    a: float  # excitatory amplitude in [0, 1]
    b: float  # phase parameter in [0, 2pi]
    c: float  # inhibitory amplitude in [0, 1]


def single_channel_agate(params: ChannelParams, e_qubit: int, i_qubit: int) -> Circuit:
    """Build a single-channel 14-gate A-Gate on the given qubit pair.

    Layout (matches qdnu/quantum_agate.py):
        E qubit:  H -> P(b) -> Rx(2a) -> P(b) -> H
        I qubit:  H -> P(b) -> Ry(2c) -> P(b) -> H
        coupling: CRy(pi/4) E->I, CRz(pi/4) I->E
    """
    a = params.a
    b = params.b
    c = params.c

    circ = Circuit()

    # Per-qubit encoding sandwich (E)
    circ.h(e_qubit)
    circ.phaseshift(e_qubit, b)
    circ.rx(e_qubit, 2.0 * a)
    circ.phaseshift(e_qubit, b)
    circ.h(e_qubit)

    # Per-qubit encoding sandwich (I)
    circ.h(i_qubit)
    circ.phaseshift(i_qubit, b)
    circ.ry(i_qubit, 2.0 * c)
    circ.phaseshift(i_qubit, b)
    circ.h(i_qubit)

    # E-I coupling (controlled rotations)
    # Braket exposes cphaseshift; CRy/CRz are decomposed via standard identities.
    _apply_cry(circ, control=e_qubit, target=i_qubit, theta=PI_4)
    _apply_crz(circ, control=i_qubit, target=e_qubit, theta=PI_4)

    return circ


def multichannel_agate(channel_params: Sequence[ChannelParams]) -> tuple[Circuit, dict]:
    """Build the full M-channel A-Gate with ancilla and ring topology.

    Returns (circuit, qubit_map) where qubit_map records the role of each
    qubit so the measurement layer and downstream readers know which qubits
    correspond to which channel.

    Qubit layout for M channels:
        qubit 0: ancilla (global sync)
        qubit 2k+1: excitatory qubit for channel k
        qubit 2k+2: inhibitory qubit for channel k
    Total qubits: 2M + 1
    """
    m = len(channel_params)
    if m < 1:
        raise ValueError("need at least one channel")

    ancilla = 0
    e_qubits = [2 * k + 1 for k in range(m)]
    i_qubits = [2 * k + 2 for k in range(m)]

    circ = Circuit()

    # Layer 1: per-channel A-Gate
    for k, params in enumerate(channel_params):
        sub = single_channel_agate(params, e_qubits[k], i_qubits[k])
        # Compose by replaying instructions onto the same Circuit
        for instr in sub.instructions:
            circ.add_instruction(instr)

    # Layer 2: ring coupling on excitatory qubits, then on inhibitory qubits
    for k in range(m):
        circ.cnot(e_qubits[k], e_qubits[(k + 1) % m])
    for k in range(m):
        circ.cnot(i_qubits[k], i_qubits[(k + 1) % m])

    # Layer 3: global sync via ancilla -- H, CZ to each E qubit, H
    circ.h(ancilla)
    for q in e_qubits:
        circ.cz(ancilla, q)
    circ.h(ancilla)

    qubit_map = {
        "ancilla": ancilla,
        "e_qubits": e_qubits,
        "i_qubits": i_qubits,
        "num_channels": m,
        "total_qubits": 2 * m + 1,
    }
    return circ, qubit_map


def _apply_cry(circ: Circuit, control: int, target: int, theta: float) -> None:
    """Decompose CRy(theta) into native gates available on most Braket backends.

    Identity: CRy(theta) = Ry(theta/2) on target, then CNOT, Ry(-theta/2) on
    target, then CNOT.
    """
    half = theta / 2.0
    circ.ry(target, half)
    circ.cnot(control, target)
    circ.ry(target, -half)
    circ.cnot(control, target)


def _apply_crz(circ: Circuit, control: int, target: int, theta: float) -> None:
    """Decompose CRz(theta) into native gates.

    Identity: CRz(theta) = Rz(theta/2) on target, CNOT, Rz(-theta/2) on
    target, CNOT.
    """
    half = theta / 2.0
    circ.rz(target, half)
    circ.cnot(control, target)
    circ.rz(target, -half)
    circ.cnot(control, target)


def expectation_z(counts: dict[str, int], qubit_index: int, total_qubits: int) -> float:
    """Compute <Z> on a single qubit from a Braket measurement-counts dict.

    Braket bitstring convention: position 0 in the string is qubit 0.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    weighted = 0.0
    for bitstring, n in counts.items():
        # +1 for |0>, -1 for |1>
        bit = bitstring[qubit_index]
        weighted += n * (1.0 if bit == "0" else -1.0)
    return weighted / total


def polarity(z_value: float) -> int:
    """sgn(<Z>), with 0 mapped to +1 by convention."""
    return -1 if z_value < 0 else 1
