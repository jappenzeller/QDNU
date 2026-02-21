# Channel Scaling Comparison

## Results

| Method          | 8ch AUC | 12ch AUC   | 16ch AUC   | Trend       |
|-----------------|---------|------------|------------|-------------|
| Classical Tier2 | 0.7234  | 0.7116     | 0.7184     | flat        |
| Quantum QPNN    | 0.4595  | INFEASIBLE | INFEASIBLE | N/A         |

## Channel Sets (Nested)

- **CH8**: ['FP1-F7', 'F7-T7', 'FP1-F3', 'F3-C3', 'FP2-F8', 'F8-T8', 'FP2-F4', 'F4-C4']
- **CH12**: ['FP1-F7', 'F7-T7', 'FP1-F3', 'F3-C3', 'FP2-F8', 'F8-T8', 'FP2-F4', 'F4-C4', 'C3-P3', 'C4-P4', 'CZ-PZ', 'FZ-CZ']
- **CH16**: ['FP1-F7', 'F7-T7', 'FP1-F3', 'F3-C3', 'FP2-F8', 'F8-T8', 'FP2-F4', 'F4-C4', 'C3-P3', 'C4-P4', 'CZ-PZ', 'FZ-CZ', 'P3-O1', 'P4-O2', 'P7-O1', 'P8-O2']

## Qubit Requirements and Computational Scaling

| Channels | Qubits | State Size | Time/Subject | Feasibility  |
|----------|--------|------------|--------------|--------------|
| 8        | 17     | ~2 MB      | ~2 min       | OK           |
| 12       | 25     | ~512 MB    | >60 min      | Infeasible   |
| 16       | 33     | ~128 GB    | N/A          | Infeasible   |

## Configuration

- **Classical**: Tier 2 MAX (correlation eigenvalues with MAX pooling), XGBoost 5-bag ensemble
- **Quantum**: PLV encoding (theta-alpha 4-13 Hz), MAX aggregation, fidelity-based classification
- **CV**: Leave-One-Subject-Out

## Key Finding: Exponential Scaling Gap

The quantum circuit's qubit count scales as **n = 2M + 1** for M channels, requiring **O(2^n)**
complex amplitudes for statevector simulation. This results in:

- **8 channels**: 17 qubits, 2^17 = 131,072 amplitudes (~2 MB)
- **12 channels**: 25 qubits, 2^25 = 33,554,432 amplitudes (~512 MB)
- **16 channels**: 33 qubits, 2^33 = 8,589,934,592 amplitudes (~128 GB)

In contrast, classical methods scale **O(M^2)** with channel count.

This exponential computational gap is a fundamental limitation of quantum statevector simulation
and demonstrates why NISQ-era quantum advantage remains elusive for EEG classification tasks.

## Performance Comparison (8 channels)

| Metric      | Classical | Quantum | Delta           |
|-------------|-----------|---------|-----------------|
| AUC         | 0.7234    | 0.4595  | -0.2639         |
| Trend       | flat      | N/A     | -               |
| Scalability | O(M^2)    | O(2^2M) | Exponential gap |

Generated: 2026-02-21 11:50:00
