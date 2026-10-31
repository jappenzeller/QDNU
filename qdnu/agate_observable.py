"""Classical (numpy-only) implementation of the A-Gate fixed observables.

R2 section 1.9 deliverable (PROMPT P2REV-03 Job 2): the submitted circuit is a
product-state encoding followed by fixed unitaries (docs/AGATE_OBSERVABLE_NOTES.md
section 4), so every expectation value is a fixed function of the 24 classical
inputs (a_k, b_k, c_k), k = 1..8. This module evaluates two of them with no
qiskit import:

- ``agate_ancilla_z(a, b, c)``: the ancilla <Z_0> of the multichannel A-Gate
  exactly as built by ``QA1/multichannel_circuit.py`` (per-channel
  H-P(b)-Rx(2a)-P(b)-H / H-P(b)-Ry(2c)-P(b)-H sandwiches, CRy(pi/4) E->I,
  CRz(pi/4) I->E, OPEN-CHAIN inter-channel CNOTs on the E and I registers,
  ancilla H - CZ from every E qubit - H, measure Z on the ancilla). Evaluated
  by direct numpy tensor simulation of the 2M+1-qubit state (M = 8 ->
  131072 amplitudes, ~milliseconds).

- ``agate_balance(a, b, c)``: the model-derived E/I balance observable
  Pi_b^coupled of the notes (section 3), per channel, mean over channels. By
  the exact identity <U Pi_b U^dagger>_coupled = <Pi_b>_uncoupled =
  sin(2a) - sin(2c) (verified to machine precision in the notes and by
  ``verify_balance_identity`` below), the closed form is sin(2a_k) -
  sin(2c_k) averaged over channels. The b-dependence enters only through the
  measurement gains 1/sin(b), 1/cos(b); the ideal expectation is b-free, but
  windows with |sin b| or |cos b| < 0.3 are unreliable to *estimate* on
  hardware (the notes' dead zones), and the sign relation to a - c needs
  a, c <= pi/4.

Inputs: ``a``, ``b``, ``c`` are array-likes of length M (one triple per EEG
channel), with a, c in [0, 1] radians and b in [0, 2*pi] from the PLV
extraction (``extract_plv_params`` in the quantum run scripts).
"""

from __future__ import annotations

import numpy as np

SQ2 = 1.0 / np.sqrt(2.0)
H_GATE = np.array([[SQ2, SQ2], [SQ2, -SQ2]], dtype=complex)


def _p(phi):
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)


def _rx(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0],
                     [0, np.exp(1j * theta / 2)]], dtype=complex)


def _controlled(u):
    """4x4 controlled-U, control = first tensor factor."""
    out = np.eye(4, dtype=complex)
    out[2:, 2:] = u
    return out


def _apply1(state, u, q):
    state = np.tensordot(u, state, axes=([1], [q]))
    return np.moveaxis(state, 0, q)


def _apply2(state, u4, q_ctrl, q_tgt):
    u = u4.reshape(2, 2, 2, 2)      # (ctrl_out, tgt_out, ctrl_in, tgt_in)
    state = np.tensordot(u, state, axes=([2, 3], [q_ctrl, q_tgt]))
    return np.moveaxis(state, [0, 1], [q_ctrl, q_tgt])


_CZ = np.diag([1, 1, 1, -1]).astype(complex)


def agate_ancilla_z(a, b, c):
    """<Z_0> of the multichannel A-Gate ancilla for one window.

    Parameters: per-channel triples (arrays of length M, M = 8 for CH8).
    Returns the exact statevector expectation, a float in [-1, 1].
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    m = len(a)
    n = 2 * m + 1
    state = np.zeros((2,) * n, dtype=complex)
    state[(0,) * n] = 1.0

    cry = _controlled(_ry(np.pi / 4))
    crz = _controlled(_rz(np.pi / 4))
    cx = _controlled(np.array([[0, 1], [1, 0]], dtype=complex))

    for i in range(m):
        q_e, q_i = 1 + 2 * i, 2 + 2 * i
        for u in (H_GATE, _p(b[i]), _rx(2 * a[i]), _p(b[i]), H_GATE):
            state = _apply1(state, u, q_e)
        for u in (H_GATE, _p(b[i]), _ry(2 * c[i]), _p(b[i]), H_GATE):
            state = _apply1(state, u, q_i)
        state = _apply2(state, cry, q_e, q_i)
        state = _apply2(state, crz, q_i, q_e)

    # Open-chain inter-channel CNOTs, exactly as QA1/multichannel_circuit.py
    # (NOT the closed ring of qdnu/multichannel_circuit.py).
    for i in range(m - 1):
        state = _apply2(state, cx, 1 + 2 * i, 1 + 2 * (i + 1))
    for i in range(m - 1):
        state = _apply2(state, cx, 2 + 2 * i, 2 + 2 * (i + 1))

    state = _apply1(state, H_GATE, 0)
    for i in range(m):
        state = _apply2(state, _CZ, 1 + 2 * i, 0)
    state = _apply1(state, H_GATE, 0)

    probs = np.abs(state) ** 2
    p0 = probs[0].sum()
    p1 = probs[1].sum()
    return float(p0 - p1)


def agate_balance(a, b, c):
    """Mean over channels of the Pi_b^coupled balance observable.

    Exact closed form per channel: sin(2a) - sin(2c) (notes section 3
    identity; b-free in expectation). Sign tracks sign(a - c) when
    a, c <= pi/4.
    """
    a = np.asarray(a, dtype=float)
    c = np.asarray(c, dtype=float)
    return float(np.mean(np.sin(2 * a) - np.sin(2 * c)))


def dead_zone_flags(a, b, c, gain_floor=0.3):
    """The notes' section 5(2) reliability flags for one window.

    Returns (amp_out_of_range, phase_dead_zone): True when any channel has
    a or c > pi/4, resp. |sin b| or |cos b| < gain_floor.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    amp = bool(np.any(a > np.pi / 4) or np.any(c > np.pi / 4))
    phase = bool(np.any(np.abs(np.sin(b)) < gain_floor) or
                 np.any(np.abs(np.cos(b)) < gain_floor))
    return amp, phase


def verify_balance_identity(n_random=2000, seed=0):
    """Check <U Pi_b U^dagger>_coupled == sin(2a) - sin(2c) on the 2-qubit
    coupled pair for random (a, b, c); returns the max absolute error."""
    rng = np.random.default_rng(seed)
    x_ = np.array([[0, 1], [1, 0]], dtype=complex)
    max_err = 0.0
    for _ in range(n_random):
        a = rng.uniform(0, np.pi / 4)
        c = rng.uniform(0, np.pi / 4)
        b = rng.uniform(0, 2 * np.pi)
        if abs(np.sin(b)) < 1e-3 or abs(np.cos(b)) < 1e-3:
            continue
        e_sandwich = H_GATE @ _p(b) @ _rx(2 * a) @ _p(b) @ H_GATE
        i_sandwich = H_GATE @ _p(b) @ _ry(2 * c) @ _p(b) @ H_GATE
        psi = np.kron(e_sandwich @ np.array([1, 0], dtype=complex),
                      i_sandwich @ np.array([1, 0], dtype=complex))
        # coupling, E = first factor: CRy(pi/4) E->I then CRz(pi/4) I->E
        cry = _controlled(_ry(np.pi / 4))                 # ctrl=E, tgt=I
        swap = np.eye(4)[[0, 2, 1, 3]]
        crz = swap @ _controlled(_rz(np.pi / 4)) @ swap   # ctrl=I, tgt=E
        u = crz @ cry
        psi_c = u @ psi
        x_e = np.kron(x_, np.eye(2))
        x_i = np.kron(np.eye(2), x_)
        obs = u @ (x_e / np.sin(b) + x_i / np.cos(b)) @ u.conj().T
        val = float(np.real(psi_c.conj() @ obs @ psi_c))
        max_err = max(max_err, abs(val - (np.sin(2 * a) - np.sin(2 * c))))
    return max_err


if __name__ == '__main__':
    err = verify_balance_identity()
    print(f"balance identity max |err| over random (a,b,c): {err:.3e}")
    z = agate_ancilla_z([0.5] * 8, [0.3] * 8, [0.7] * 8)
    print(f"agate_ancilla_z(uniform example) = {z:.12f}")
