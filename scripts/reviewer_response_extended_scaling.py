#!/usr/bin/env python3
"""
REV-001 Task 4: extended transpiled scaling for M in {2,4,6,8,12,16,24,32}.

Uses the same circuit-construction code (create_multichannel_circuit), the
same backend (ibm_torino via FakeTorino, basis cz/rz/sx/x), the same
optimization_level=3, and the same seed_transpiler as the original Table 6
run (SEED=42 in scripts/ibm_hardware_validation.py).

Outputs:
    results/reviewer_response/extended_scaling.csv
    results/reviewer_response/extended_scaling.json
"""

import sys
import json
import csv
import logging
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from QA1.multichannel_circuit import create_multichannel_circuit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
log = logging.getLogger('rev001-scale')

OUT_DIR = ROOT / 'results' / 'reviewer_response'
SEED = 42  # matches scripts/ibm_hardware_validation.py
M_VALUES = [2, 4, 6, 8, 12, 16, 24, 32]
ORIGINAL_M = [2, 4, 6, 8]


def count_two_qubit_gates(circuit, gate_names=None):
    if gate_names is None:
        gate_names = ['cz', 'ecr', 'cx', 'cnot', 'swap', 'iswap', 'rzz', 'rxx', 'ryy']
    ops = circuit.count_ops()
    return sum(ops.get(g, 0) for g in gate_names)


def main():
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    try:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        backend = FakeTorino()
        backend_name = 'FakeTorino (ibm_torino)'
    except ImportError:
        from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
        backend = FakeSherbrooke()
        backend_name = 'FakeSherbrooke (fallback)'
    log.info(f'Backend: {backend_name}; basis_gates={list(backend.basis_gates)}; n_qubits={backend.num_qubits}')

    pm = generate_preset_pass_manager(
        optimization_level=3,
        backend=backend,
        seed_transpiler=SEED,
    )

    rows = []
    log.info('Transpiling at M ∈ ' + str(M_VALUES) + ' with measurements (matches Table 6 run)')
    for M in M_VALUES:
        if 2 * M + 1 > backend.num_qubits:
            log.warning(f'M={M} requires {2*M+1} qubits, backend has {backend.num_qubits}. Skipping.')
            continue
        params = [(0.5, 1.0, 0.5)] * M  # synchronized; matches scaling experiment
        qc = create_multichannel_circuit(params)
        qc.measure_all()
        t_qc = pm.run(qc)
        two_q = count_two_qubit_gates(t_qc)
        ops = dict(t_qc.count_ops())
        cz = int(ops.get('cz', 0))
        rows.append({
            'M': M,
            'qubits': 2 * M + 1,
            'original_depth': qc.depth(),
            'transpiled_depth': t_qc.depth(),
            'transpiled_total_gates': t_qc.size(),
            'transpiled_cz_gates': cz,
            'transpiled_two_qubit_total': two_q,
            'transpiled_ops': ops,
        })
        log.info(f'  M={M}: qubits={2*M+1}, cz={cz}, two_q={two_q}, depth={t_qc.depth()}, total={t_qc.size()}')

    # Combined refit over all M (CZ vs M)
    Ms = np.array([r['M'] for r in rows], dtype=float)
    czs = np.array([r['transpiled_cz_gates'] for r in rows], dtype=float)
    slope, intercept = np.polyfit(Ms, czs, 1)
    pred = slope * Ms + intercept
    ss_res = np.sum((czs - pred) ** 2)
    ss_tot = np.sum((czs - czs.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    log.info(f'Combined fit: CZ = {slope:.3f}*M + {intercept:.3f}, R^2 = {r2:.5f}')

    # Also report the original-only fit for comparison
    mask_orig = np.array([m_val in ORIGINAL_M for m_val in Ms.astype(int)])
    s_o, i_o = np.polyfit(Ms[mask_orig], czs[mask_orig], 1)
    p_o = s_o * Ms[mask_orig] + i_o
    r2_o = 1.0 - np.sum((czs[mask_orig] - p_o) ** 2) / np.sum((czs[mask_orig] - czs[mask_orig].mean()) ** 2)
    log.info(f'Original M only ({ORIGINAL_M}): CZ = {s_o:.3f}*M + {i_o:.3f}, R^2 = {r2_o:.5f}')

    # Sanity check: original M (2,4,6,8) CZ counts should match the published
    # {14, 34, 67, 97} numbers if Qiskit/passes haven't drifted.
    published_cz = {2: 14, 4: 34, 6: 67, 8: 97}
    drift = []
    for r in rows:
        if r['M'] in published_cz:
            pub = published_cz[r['M']]
            if r['transpiled_cz_gates'] != pub:
                drift.append((r['M'], pub, r['transpiled_cz_gates']))
    if drift:
        log.warning(f'Drift from published Table 6 CZ counts: {drift}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / 'extended_scaling.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['M', 'qubits', 'transpiled_depth', 'CZ_gates', 'total_gates'])
        for r in rows:
            writer.writerow([r['M'], r['qubits'], r['transpiled_depth'], r['transpiled_cz_gates'], r['transpiled_total_gates']])
    log.info(f'Wrote {csv_path}')

    json_path = OUT_DIR / 'extended_scaling.json'
    with open(json_path, 'w') as f:
        json.dump({
            'backend': backend_name,
            'basis_gates': list(backend.basis_gates),
            'seed_transpiler': SEED,
            'optimization_level': 3,
            'M_values': [r['M'] for r in rows],
            'rows': rows,
            'combined_fit': {'slope': float(slope), 'intercept': float(intercept), 'R2': float(r2)},
            'original_only_fit': {'slope': float(s_o), 'intercept': float(i_o), 'R2': float(r2_o),
                                  'M_subset': ORIGINAL_M},
            'published_cz_check': {str(k): v for k, v in published_cz.items()},
            'cz_drift_from_published': drift,
        }, f, indent=2)
    log.info(f'Wrote {json_path}')


if __name__ == '__main__':
    main()
