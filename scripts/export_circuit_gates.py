#!/usr/bin/env python3
"""
Export gate sequence from circuit_trajectories.json for visualization circuit bar.

Creates visualization_data/circuit_gates.json with the logical circuit gate sequence
in a format suitable for rendering the Qiskit-style circuit bar.
"""

import json
from pathlib import Path

# Channel labels for the 8 EEG channel pairs (logical qubits)
CHANNEL_LABELS = [
    "FP1-F7", "F7-T7", "FP1-F3", "F3-C3",
    "FP2-F8", "F8-T8", "FP2-F4", "F4-C4"
]


def main():
    # Load trajectory data
    traj_path = Path(__file__).parent.parent / "visualization_data" / "circuit_trajectories.json"
    with open(traj_path, 'r') as f:
        traj_data = json.load(f)

    # Extract gate sequence from chb01 ictal trajectory (all trajectories use same circuit)
    trajectory = traj_data["patients"]["chb01"]["ictal"]

    # Find the number of qubits used in the transpiled circuit
    all_qubits = set()
    for step in trajectory:
        all_qubits.update(step["gate_qubits"])
    n_qubits = max(all_qubits) + 1  # Qubit indices are 0-based

    # Use all transpiled qubits (17 for IBM Heron)
    n_logical_qubits = n_qubits

    # Build gate list with parallel column positions
    # Track next available column for each qubit
    qubit_next_col = {q: 0 for q in range(n_qubits)}
    gates = []

    for i, step_data in enumerate(trajectory):
        gate_type = step_data["gate_type"].upper()
        gate_qubits = step_data["gate_qubits"]
        is_two_qubit = len(gate_qubits) == 2

        # Find the earliest column where all qubits are free
        if is_two_qubit:
            # For two-qubit gates, need both qubits free at same column
            col = max(qubit_next_col[q] for q in gate_qubits)
        else:
            col = qubit_next_col[gate_qubits[0]]

        gate_entry = {
            "type": gate_type,
            "index": i,
            "step": col,
        }

        if is_two_qubit:
            gate_entry["qubits"] = gate_qubits
            gate_entry["control"] = gate_qubits[0]
            gate_entry["target"] = gate_qubits[1]
            # Block both qubits for next column
            for q in gate_qubits:
                qubit_next_col[q] = col + 1
        else:
            gate_entry["qubit"] = gate_qubits[0]
            qubit_next_col[gate_qubits[0]] = col + 1

        gates.append(gate_entry)

    n_columns = max(g["step"] for g in gates) + 1

    # Create output structure
    output = {
        "n_qubits": n_logical_qubits,
        "n_gates": len(trajectory),
        "n_columns": n_columns,
        "channel_labels": CHANNEL_LABELS[:n_logical_qubits],
        "gates": gates,
        "gate_types": sorted(list(set(g["type"] for g in gates)))
    }

    # Write output
    output_path = Path(__file__).parent.parent / "visualization_data" / "circuit_gates.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Exported {len(gates)} gates in {n_columns} columns for {n_logical_qubits} qubits to {output_path}")
    print(f"Gate types: {output['gate_types']}")


if __name__ == "__main__":
    main()
